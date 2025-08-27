#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.multi_VAE import MultimodalVAE
from model.baseline import GeneToIsoformModel
from model.baseline_VAE import BaselineVAE
from utils.config import Config
from data_preprocess.hash_files import ExpressionDataset, MultiOmicsDataset
from inference.metrics import calculate_metrics


def read_ids(split_txt):
    """Read sample IDs from text file"""
    with open(split_txt) as f:
        return [line.strip() for line in f if line.strip()]


def load_model(config, device):
    """Load trained model based on config"""
    if config.model_type == 'baseline':
        model_path = "/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/model_resources/best_baseline_baseline_model.pt"
        model = GeneToIsoformModel(
            config.n_genes,
            config.n_isoforms,
            config.hidden_dims,
            config.latent_dim
        )
    elif config.model_type == 'baseline_vae':
        model_path = "/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/model_resources/best_baseline_vae_baseline_VAE_model.pt"
        model = BaselineVAE(config)
    else:
        model_path = "/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/model_resources/best_model_w_feat_complete_loss_10.pt"
        model = MultimodalVAE(config)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_similarity_metrics(true_vals, pred_vals):
    """Per-sample Pearson & cosine"""
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)

    if true_vals.shape[1] != pred_vals.shape[1]:
        n = true_vals.shape[0]
        return np.full(n, np.nan), np.full(n, np.nan)

    n = true_vals.shape[0]
    corrs = np.full(n, np.nan)
    cos_sims = np.full(n, np.nan)

    for i in range(n):
        y_true, y_pred = true_vals[i], pred_vals[i]
        std_true, std_pred = y_true.std(), y_pred.std()

        if std_true > 0 and std_pred > 0:
            cov = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
            corrs[i] = cov / (std_true * std_pred)

        norm_true, norm_pred = np.linalg.norm(y_true), np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            cos_sims[i] = np.dot(y_true, y_pred) / (norm_true * norm_pred)

    return corrs, cos_sims


def compute_isoform_metrics(true_vals, pred_vals):
    """
    Compute per-isoform Pearson & cosine across samples.
    Shape: (n_samples, n_isoforms)
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

        if y_true.std() > 0 and y_pred.std() > 0:
            pearsons[j] = np.corrcoef(y_true, y_pred)[0, 1]

        norm_true, norm_pred = np.linalg.norm(y_true), np.linalg.norm(y_pred)
        if norm_true > 0 and norm_pred > 0:
            cosines[j] = np.dot(y_true, y_pred) / (norm_true * norm_pred)

    return pd.DataFrame({
        "isoform_idx": np.arange(n_isoforms),
        "pearson": pearsons,
        "cosine": cosines
    })


def extract_tensor_candidate(obj):
    """Try to pull out a tensor from nested structures"""
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
        for k in ['recon', 'mu', 'pred', 'recon_iso', 'recon_iso_mu']:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
    return None


def save_results(metrics, predictions_batches, config):
    """Save metrics and predictions to CSV"""
    output_dir = f"results/{config.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    if metrics:
        pd.DataFrame(metrics).to_csv(f"{output_dir}/metrics.csv", index=False)

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

    print(f"Saved results to {output_dir}", flush=True)


def run_inference(config, device, save_predictions=False):
    """Main inference workflow"""
    working_dir = config.working_dir
    ids_path = f'{working_dir}/samples/by_study/by_study_test_samples.txt'
    test_ids = read_ids(ids_path)
    print(f"Loaded {len(test_ids)} test samples", flush=True)

    test_ds_genes = ExpressionDataset(f'{working_dir}/h5/genes.h5', test_ids, None)
    test_ds_iso = ExpressionDataset(f'{working_dir}/h5/isoforms.h5', test_ids, None)
    test_dataset = MultiOmicsDataset(test_ds_genes, test_ds_iso)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    model = load_model(config, device)
    print(f"Loaded model type '{config.model_type}' on {device}")

    all_metrics, predictions_batches = [], [] if save_predictions else None
    per_sample_ids, per_sample_pearson, per_sample_cosine = [], [], []
    all_true, all_pred = [], []
    id_ptr = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            bg, bi = batch[0].to(device), batch[1].to(device)

            if config.model_type == 'baseline':
                out = model(bg)
                outputs = {'recon_iso': out} if torch.is_tensor(out) else {}
            else:
                try:
                    candidate = model(bg, bi)
                except TypeError:
                    candidate = model(bg)
                outputs = candidate if isinstance(candidate, dict) else {'recon_iso': extract_tensor_candidate(candidate)}

            try:
                batch_metrics = calculate_metrics(outputs, bg, bi, config)
                all_metrics.append(batch_metrics)
            except Exception as e:
                print(f"Warning: calculate_metrics failed: {e}", flush=True)
                all_metrics.append({})

            pred_iso_tensor = None
            if isinstance(outputs, dict):
                for k in ['recon_iso', 'recon_iso_mu', 'iso_pred']:
                    if k in outputs and outputs[k] is not None:
                        pred_iso_tensor = extract_tensor_candidate(outputs[k])
                        if pred_iso_tensor is not None:
                            break

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

            ids_batch = test_ids[id_ptr: id_ptr + bi.shape[0]]
            per_sample_ids.extend(ids_batch)
            per_sample_pearson.extend(batch_corrs.tolist())
            per_sample_cosine.extend(batch_cosines.tolist())

            if save_predictions:
                preds = {"gene_input": bg.cpu().numpy(), "iso_true": bi.cpu().numpy()}
                if pred_iso_tensor is not None:
                    preds["iso_pred"] = pred_iso_tensor.cpu().numpy()
                predictions_batches.append(preds)

            id_ptr += bi.shape[0]
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}", flush=True)

    out_dir = f"results/{config.run_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Save per-sample metrics ---
    per_sample_df = pd.DataFrame({
        "sample_id": per_sample_ids,
        "pearson": per_sample_pearson,
        "cosine": per_sample_cosine
    })
    per_sample_df.to_csv(os.path.join(out_dir, "per_sample_similarity.csv"), index=False)
    print(f"Saved per-sample similarity to {out_dir}/per_sample_similarity.csv", flush=True)

    print(f"Mean Pearson (samples): {np.nanmean(per_sample_df['pearson']):.4f}",flush=True)
    print(f"Mean Cosine (samples): {np.nanmean(per_sample_df['cosine']):.4f}",flush=True)

    # --- Save per-isoform metrics ---
    if all_true and all_pred:
        all_true_np = np.vstack(all_true)
        all_pred_np = np.vstack(all_pred)
        isoform_df = compute_isoform_metrics(all_true_np, all_pred_np)
        isoform_df.to_csv(os.path.join(out_dir, "per_isoform_similarity.csv"), index=False)
        print(f"Saved per-isoform similarity to {out_dir}/per_isoform_similarity.csv", flush=True)

        print(f"Mean Pearson (isoforms): {np.nanmean(isoform_df['pearson']):.4f}",flush=True)
        print(f"Mean Cosine (isoforms): {np.nanmean(isoform_df['cosine']):.4f}",flush=True)

    if save_predictions:
        save_results(all_metrics, predictions_batches, config)

    return {"per_sample": per_sample_df, "metrics_list": all_metrics}


if __name__ == "__main__":
    config = Config(
        run_name='updated_experiment_baseline_vae',
        working_dir='/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/data',
        model_type='baseline_vae',
        n_genes=50988,
        n_isoforms=214483,
        latent_dim=16,#64,
        hidden_dims=[512],#[1024],
        scaler=False,
        batch_size=512#64
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    run_inference(config, device, save_predictions=False)
