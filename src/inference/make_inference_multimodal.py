import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union
from model.multi_VAE import MultimodalVAE
from utils.config import Config
from data_preprocess.hash_files import ExpressionDataset, MultiOmicsDataset
from inference.metrics import calculate_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_ids(split_txt: str) -> List[str]:
    """Read sample IDs from text file"""
    try:
        with open(split_txt) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"File {split_txt} not found")
        raise
    except Exception as e:
        logger.error(f"Error reading {split_txt}: {e}")
        raise
def load_model(config, device):
    if config.model_type == 'multimodal_vae':
        model_path = "/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/model_resources/best_model_w_feat_complete_loss_10.pt"
        model = MultimodalVAE(config)
    
    elif ocnfig.model_type=="multimodal_no_joint":
        model_path="/home/projects2/kvs_students/2025/fl_mVAE/best_model_repro_check_multimodal_nojoiny.pt"
        model=MultimodalVAE_NoJoint(config)
    
    state = torch.load(model_path, map_location=device)

        # Handle checkpoints that wrap weights in 'state_dict'
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']

        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    else:
        raise ValueError(f"Unsupported model_type: {config.model_type}")


def compute_similarity_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample Pearson & cosine similarity metrics"""
    if true_vals.shape != pred_vals.shape:
        logger.warning(f"Shape mismatch: true_vals {true_vals.shape}, pred_vals {pred_vals.shape}")
        n = min(true_vals.shape[0], pred_vals.shape[0])
        true_vals = true_vals[:n]
        pred_vals = pred_vals[:n]
    
    n = true_vals.shape[0]
    corrs = np.full(n, np.nan)
    cos_sims = np.full(n, np.nan)
    
    for i in range(n):
        y_true, y_pred = true_vals[i], pred_vals[i]
        
        # Check for constant values
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            continue
            
        # Pearson correlation
        try:
            corrs[i] = np.corrcoef(y_true, y_pred)[0, 1]
        except:
            pass
            
        # Cosine similarity
        norm_true, norm_pred = np.linalg.norm(y_true), np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            cos_sims[i] = np.dot(y_true, y_pred) / (norm_true * norm_pred)
    
    return corrs, cos_sims

def compute_isoform_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> pd.DataFrame:
    """
    Compute per-isoform Pearson & cosine across samples.
    Shape: (n_samples, n_isoforms)
    """
    if true_vals.shape != pred_vals.shape:
        logger.error(f"Shape mismatch in isoform metrics: {true_vals.shape} vs {pred_vals.shape}")
        # Align shapes
        min_samples = min(true_vals.shape[0], pred_vals.shape[0])
        min_isoforms = min(true_vals.shape[1], pred_vals.shape[1])
        true_vals = true_vals[:min_samples, :min_isoforms]
        pred_vals = pred_vals[:min_samples, :min_isoforms]
    
    n_samples, n_isoforms = true_vals.shape
    pearsons = np.full(n_isoforms, np.nan)
    cosines = np.full(n_isoforms, np.nan)
    
    for j in range(n_isoforms):
        y_true = true_vals[:, j]
        y_pred = pred_vals[:, j]
        
        # Skip if constant values
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            continue
            
        # Pearson correlation
        try:
            pearsons[j] = np.corrcoef(y_true, y_pred)[0, 1]
        except:
            pass
            
        # Cosine similarity
        norm_true, norm_pred = np.linalg.norm(y_true), np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            cosines[j] = np.dot(y_true, y_pred) / (norm_true * norm_pred)
    
    return pd.DataFrame({
        "isoform_idx": np.arange(n_isoforms),
        "pearson": pearsons,
        "cosine": cosines
    })

def extract_tensor_candidate(obj: Union[torch.Tensor, Dict, List]) -> Optional[torch.Tensor]:
    """Try to pull out a tensor from nested structures or model outputs."""
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for e in obj:
            t = extract_tensor_candidate(e)
            if t is not None:
                return t
    if isinstance(obj, dict):
        # Priority order for isoform reconstruction keys
        iso_keys_priority = [
            'recon_iso_cross_mu', 'recon_iso_cross', 'recon_iso_mu', 'recon_iso',
            'recon_iso_joint_mu', 'recon_iso_joint'
        ]
        for k in iso_keys_priority:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
                
        # If no isoform keys found, try any tensor
        for v in obj.values():
            t = extract_tensor_candidate(v)
            if t is not None:
                return t
                
    return None

def save_results(metrics: List[Dict], predictions_batches: List[Dict], config: Config) -> None:
    """Save metrics and predictions to CSV"""
    output_dir = f"results/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if metrics:
            # Flatten metrics if they contain nested dictionaries
            flat_metrics = []
            for m in metrics:
                if isinstance(m, dict):
                    flat_metrics.append(m)
                else:
                    # Handle non-dict metrics
                    flat_metrics.append({'value': m})
            
            pd.DataFrame(flat_metrics).to_csv(f"{output_dir}/metrics.csv", index=False)
        
        if predictions_batches:
            for i, batch_pred in enumerate(predictions_batches):
                batch_df = pd.DataFrame()
                for key, arr in batch_pred.items():
                    arr = np.asarray(arr)
                    if arr.ndim == 1:
                        batch_df[key] = arr
                    elif arr.ndim == 2:
                        for feat_idx in range(arr.shape[1]):
                            batch_df[f"{key}_feat_{feat_idx}"] = arr[:, feat_idx]
                    else:
                        flat = arr.reshape(arr.shape[0], -1)
                        for feat_idx in range(flat.shape[1]):
                            batch_df[f"{key}_flat_{feat_idx}"] = flat[:, feat_idx]
                batch_df.to_csv(f"{output_dir}/predictions_batch_{i}.csv", index=False)
        
        logger.info(f"Saved results to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def run_inference(config: Config, device: torch.device, save_predictions: bool = False) -> Dict:
    """Main inference workflow"""
    try:
        working_dir = config.working_dir
        ids_path = f'{working_dir}/samples/by_study/by_study_test_samples.txt'
        test_ids = read_ids(ids_path)
        logger.info(f"Loaded {len(test_ids)} test samples")
        
        # Load datasets
        test_ds_genes = ExpressionDataset(f'{working_dir}/h5/genes.h5', test_ids, None)
        test_ds_iso = ExpressionDataset(f'{working_dir}/h5/isoforms.h5', test_ids, None)
        test_dataset = MultiOmicsDataset(test_ds_genes, test_ds_iso)
        
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                                pin_memory=True, num_workers=4)
        
        # Load model
        print("Loading model...", flush=True)
        model = load_model(config, device)
        print("Loaded model", flush=True)
        
        # Initialize result containers
        all_metrics, predictions_batches = [], [] if save_predictions else None
        per_sample_ids, per_sample_pearson, per_sample_cosine = [], [], []
        all_true, all_pred = [], []
        id_ptr = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                bg, bi = batch[0].to(device), batch[1].to(device)
                
                # Forward pass
                try:
                    outputs = model(bg, bi)
                except TypeError:
                    # Fallback if model has different signature
                    try:
                        outputs = model(bg)
                    except Exception as e:
                        logger.error(f"Model forward pass failed: {e}")
                        continue
                
                # Calculate metrics
                try:
                    batch_metrics = calculate_metrics(outputs, bg, bi, config)
                    all_metrics.append(batch_metrics)
                except Exception as e:
                    logger.warning(f"calculate_metrics failed: {e}")
                    all_metrics.append({})
                
                # Extract predictions
                pred_iso_tensor = extract_tensor_candidate(outputs)
                
                # Fallback to infer_isoforms if no tensor found
                if pred_iso_tensor is None:
                    try:
                        pred_iso_tensor = model.infer_isoforms(bg, deterministic=True)
                    except Exception as e:
                        logger.warning(f"infer_isoforms failed: {e}")
                        pred_iso_tensor = None
                
                # Compute similarity metrics if we have predictions
                if pred_iso_tensor is not None:
                    pred_np = pred_iso_tensor.detach().cpu().numpy()
                    true_np = bi.detach().cpu().numpy()
                    all_true.append(true_np)
                    all_pred.append(pred_np)
                    
                    batch_corrs, batch_cosines = compute_similarity_metrics(true_np, pred_np)
                else:
                    batch_size = bi.shape[0]
                    batch_corrs = np.full(batch_size, np.nan)
                    batch_cosines = np.full(batch_size, np.nan)
                    logger.warning(f"No predictions extracted for batch {batch_idx}")
                
                # Store results
                ids_batch = test_ids[id_ptr: id_ptr + bi.shape[0]]
                per_sample_ids.extend(ids_batch)
                per_sample_pearson.extend(batch_corrs.tolist())
                per_sample_cosine.extend(batch_cosines.tolist())
                
                if save_predictions:
                    preds = {
                        "gene_input": bg.cpu().numpy(), 
                        "iso_true": bi.cpu().numpy()
                    }
                    if pred_iso_tensor is not None:
                        preds["iso_pred"] = pred_iso_tensor.cpu().numpy()
                    predictions_batches.append(preds)
                
                id_ptr += bi.shape[0]
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed batch {batch_idx+1}/{len(test_loader)}")
        
        # Save results
        out_dir = f"results/{config.run_name}"
        os.makedirs(out_dir, exist_ok=True)
        
        # Per-sample metrics
        per_sample_df = pd.DataFrame({
            "sample_id": per_sample_ids,
            "pearson": per_sample_pearson,
            "cosine": per_sample_cosine
        })
        per_sample_df.to_csv(os.path.join(out_dir, "per_sample_similarity.csv"), index=False)
        logger.info(f"Saved per-sample similarity to {out_dir}/per_sample_similarity.csv")
        
        # Summary statistics
        mean_pearson = np.nanmean(per_sample_df['pearson'])
        mean_cosine = np.nanmean(per_sample_df['cosine'])
        logger.info(f"Mean Pearson (samples): {mean_pearson:.4f}")
        logger.info(f"Mean Cosine (samples): {mean_cosine:.4f}")
        
        # Per-isoform metrics
        if all_true and all_pred:
            all_true_np = np.vstack(all_true)
            all_pred_np = np.vstack(all_pred)
            isoform_df = compute_isoform_metrics(all_true_np, all_pred_np)
            isoform_df.to_csv(os.path.join(out_dir, "per_isoform_similarity.csv"), index=False)
            logger.info(f"Saved per-isoform similarity to {out_dir}/per_isoform_similarity.csv")
            
            mean_iso_pearson = np.nanmean(isoform_df['pearson'])
            mean_iso_cosine = np.nanmean(isoform_df['cosine'])
            logger.info(f"Mean Pearson (isoforms): {mean_iso_pearson:.4f}")
            logger.info(f"Mean Cosine (isoforms): {mean_iso_cosine:.4f}")
        
        # Save other results if requested
        if save_predictions:
            save_results(all_metrics, predictions_batches, config)
        
        return {
            "per_sample": per_sample_df,
            "metrics_list": all_metrics,
            "per_isoform": isoform_df if all_true and all_pred else None
        }
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    # Example config - adapt paths & dims to match your trained model and data
    config = Config(
        run_name='multimodal_vae',
        working_dir='/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/data',
        n_genes=50988,
        n_isoforms=214483,
        latent_dim=64,
        hidden_dims=[1024],
        recon_loss_type='mse',
        iso_from_gene=True,
        gene_from_iso=True,
        scaler=False,
        batch_size=64
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        results = run_inference(config, device, save_predictions=False)
        logger.info("Inference completed successfully")
    except Exception as e:
        logger.error(f"Inference failed: {e}")