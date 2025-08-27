import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import Config
from model.multi_VAE_no_joint import MultimodalVAE_NoJoint
from utils.datasets import ExpressionDataset, MultiOmicsDataset

def read_ids(split_txt):
    """Read sample IDs from text file"""
    with open(split_txt) as f:
        return [line.strip() for line in f if line.strip()]

def load_model(config, device, model_path):
    """Load trained MultimodalVAE_NoJoint model"""
    model = MultimodalVAE_NoJoint(config)
    
    # Load state dict
    state = torch.load(model_path, map_location=device)
    
    # Handle different state dict formats
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def compute_per_sample_metrics(true_vals, pred_vals):
    """
    Compute per-sample Pearson & cosine similarity (same as compute_similarity_metrics).
    Shape: (n_samples, n_features)
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)

    if true_vals.shape[1] != pred_vals.shape[1]:
        n = true_vals.shape[0]
        return np.full(n, np.nan), np.full(n, np.nan)

    n_samples = true_vals.shape[0]
    correlations = np.full(n_samples, np.nan)
    cosine_sims = np.full(n_samples, np.nan)

    for i in range(n_samples):
        y_true = true_vals[i]
        y_pred = pred_vals[i]

        # Pearson correlation
        std_true, std_pred = y_true.std(), y_pred.std()
        if std_true > 0 and std_pred > 0:
            cov = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
            correlations[i] = cov / (std_true * std_pred)

        # Cosine similarity
        norm_true, norm_pred = np.linalg.norm(y_true), np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            cosine_sims[i] = np.dot(y_true, y_pred) / (norm_true * norm_pred)

    return correlations, cosine_sims


def compute_per_isoform_metrics(true_vals, pred_vals):
    """
    Compute per-isoform Pearson & cosine similarity across samples
    (same as compute_isoform_metrics).
    """
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)

    if true_vals.shape != pred_vals.shape:
        raise ValueError(f"Shape mismatch: {true_vals.shape} vs {pred_vals.shape}")

    n_samples, n_isoforms = true_vals.shape
    pearsons = np.full(n_isoforms, np.nan)
    cosines = np.full(n_isoforms, np.nan)

    for j in range(n_isoforms):
        y_true = true_vals[:, j]
        y_pred = pred_vals[:, j]

        # Pearson
        if y_true.std() > 0 and y_pred.std() > 0:
            pearsons[j] = np.corrcoef(y_true, y_pred)[0, 1]

        # Cosine
        norm_true, norm_pred = np.linalg.norm(y_true), np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            cosines[j] = np.dot(y_true, y_pred) / (norm_true * norm_pred)

    return pd.DataFrame({
        "isoform_idx": np.arange(n_isoforms),
        "pearson": pearsons,
        "cosine": cosines
    })


def extract_prediction(outputs, config):
    """
    Extract the appropriate prediction from model outputs
    """
    if config.recon_loss_type == 'gaussian':
        # Try cross-modal prediction first, then direct
        if outputs.get('recon_iso_cross_mu') is not None:
            return outputs['recon_iso_cross_mu']
        elif outputs.get('recon_iso_mu') is not None:
            return outputs['recon_iso_mu']
    else:
        # Try cross-modal prediction first, then direct
        if outputs.get('recon_iso_cross') is not None:
            return outputs['recon_iso_cross']
        elif outputs.get('recon_iso') is not None:
            return outputs['recon_iso']
    
    # Fallback: try to find any tensor that might be the prediction
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.dim() == 2 and value.size(1) == config.n_isoforms:
            return value
    
    raise ValueError("Could not extract prediction from model outputs")

def run_inference(config, device, model_path, save_predictions=True):
    """Main inference workflow for MultimodalVAE_NoJoint"""
    working_dir = config.working_dir

    # Load test samples
    ids_path = f'{working_dir}/samples/by_study/by_study_test_samples.txt'
    test_ids = read_ids(ids_path)
    print(f"Loaded {len(test_ids)} test samples", flush=True)

    if config.scaler:
        scaler_genes = 'scaler_genes.pkl'
        scaler_isof = 'scaler_isoforms.pkl'
    else:
        scaler_genes = None
        scaler_isof = None

    # Load datasets
    test_ds_genes = ExpressionDataset(f'{working_dir}/h5/genes.h5', test_ids, scaler_genes)
    test_ds_iso = ExpressionDataset(f'{working_dir}/h5/isoforms.h5', test_ids, scaler_isof)
    test_dataset = MultiOmicsDataset(test_ds_genes, test_ds_iso)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )

    model = load_model(config, device, model_path)
    print(f"Loaded model for inference on {device}")

    all_predictions = []
    all_true_isoforms = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch_idx, (bg, bi) in enumerate(test_loader):
            bg = bg.to(device)
            bi = bi.to(device)
            
            outputs = model(bg, bi)
            
            pred_iso = extract_prediction(outputs, config)
            
            all_predictions.append(pred_iso.cpu().numpy())
            all_true_isoforms.append(bi.cpu().numpy())
            
            # Store sample IDs for this batch
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(test_ids))
            all_sample_ids.extend(test_ids[start_idx:end_idx])
            
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}", flush=True)
    
    all_predictions = np.vstack(all_predictions)
    all_true_isoforms = np.vstack(all_true_isoforms)
    
    output_dir = f"results/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_corrs, sample_cosines = compute_per_sample_metrics(all_true_isoforms, all_predictions)
    sample_metrics = pd.DataFrame({
        'sample_id': all_sample_ids,
        'pearson': sample_corrs,
        'cosine': sample_cosines
    })
    sample_metrics.to_csv(f"{output_dir}/per_sample_similarity.csv", index=False)
    
    # Compute and save per-isoform metrics
    isoform_metrics = compute_per_isoform_metrics(all_true_isoforms, all_predictions)
    isoform_metrics.to_csv(f"{output_dir}/per_isoform_similarity.csv", index=False)
    
    if save_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_df.columns = [f"isoform_{i}" for i in range(pred_df.shape[1])]
        pred_df.insert(0, 'sample_id', all_sample_ids)
        pred_df.to_csv(f"{output_dir}/predictions.csv", index=False)
        
        true_df = pd.DataFrame(all_true_isoforms)
        true_df.columns = [f"isoform_{i}" for i in range(true_df.shape[1])]
        true_df.insert(0, 'sample_id', all_sample_ids)
        true_df.to_csv(f"{output_dir}/true_values.csv", index=False)
    
    print(f"Average sample-level Pearson correlation: {sample_corrs.mean():.4f}")
    print(f"Average sample-level cosine similarity: {sample_cosines.mean():.4f}")
    print(f"Average isoform-level Pearson correlation: {isoform_metrics['pearson'].mean():.4f}")
    



if __name__ == "__main__":
    config = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    model_path = "path/to/.pt"
    
    # Run inference
    results = run_inference(config, device, model_path, save_predictions=False)
    
    print("Inference completed successfully!", flush=True)