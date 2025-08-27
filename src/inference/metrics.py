import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

def calculate_metrics(outputs, bg, bi, config):
    """Calculate evaluation metrics for a batch"""
    metrics = {}
    
    # Function to calculate correlation and cosine
    def calc_metrics(pred, true):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
            true = true.cpu().numpy()
        
        # Flatten for overall metrics
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        
        # Pearson correlation
        corr = np.corrcoef(pred_flat, true_flat)[0, 1]
        
        # Cosine similarity
        cos_sim = np.dot(pred_flat, true_flat) / (
            np.linalg.norm(pred_flat) * np.linalg.norm(true_flat) + 1e-8)
        
        # MSE
        mse = np.mean((pred_flat - true_flat) ** 2)
        
        return corr, cos_sim, mse
    
    # Model-specific metrics
    if config.model_type == 'baseline':
        pred_iso = outputs['recon_iso']
        corr, cos_sim, mse = calc_metrics(pred_iso, bi)
        metrics = {
            'iso_corr': corr,
            'iso_cosine': cos_sim,
            'iso_mse': mse
        }
    
    elif config.model_type == 'baseline_vae':
        # Gene reconstruction
        recon_gene = outputs.get('recon_gene', outputs.get('recon_gene_mu'))
        g_corr, g_cos, g_mse = calc_metrics(recon_gene, bg)
        
        # Isoform prediction
        recon_iso = outputs.get('recon_iso', outputs.get('recon_iso_mu'))
        i_corr, i_cos, i_mse = calc_metrics(recon_iso, bi)
        
        metrics = {
            'gene_corr': g_corr,
            'gene_cosine': g_cos,
            'gene_mse': g_mse,
            'iso_corr': i_corr,
            'iso_cosine': i_cos,
            'iso_mse': i_mse
        }
    
    else:  # Multimodal VAE
        # Joint reconstructions
        recon_gene_joint = outputs.get('recon_gene_joint', outputs.get('recon_gene_mu'))
        g_corr, g_cos, g_mse = calc_metrics(recon_gene_joint, bg)
        
        recon_iso_joint = outputs.get('recon_iso_joint', outputs.get('recon_iso_joint_mu'))
        i_corr, i_cos, i_mse = calc_metrics(recon_iso_joint, bi)
        
        metrics = {
            'gene_joint_corr': g_corr,
            'gene_joint_cosine': g_cos,
            'gene_joint_mse': g_mse,
            'iso_joint_corr': i_corr,
            'iso_joint_cosine': i_cos,
            'iso_joint_mse': i_mse
        }
        
        # Cross-modal reconstructions
        if 'recon_iso_cross' in outputs or 'recon_iso_cross_mu' in outputs:
            recon_iso_cross = outputs.get('recon_iso_cross', outputs.get('recon_iso_cross_mu'))
            i_corr, i_cos, i_mse = calc_metrics(recon_iso_cross, bi)
            metrics.update({
                'iso_cross_corr': i_corr,
                'iso_cross_cosine': i_cos,
                'iso_cross_mse': i_mse
            })
        
        if 'recon_gene_cross' in outputs or 'recon_gene_cross_mu' in outputs:
            recon_gene_cross = outputs.get('recon_gene_cross', outputs.get('recon_gene_cross_mu'))
            g_corr, g_cos, g_mse = calc_metrics(recon_gene_cross, bg)
            metrics.update({
                'gene_cross_corr': g_corr,
                'gene_cross_cosine': g_cos,
                'gene_cross_mse': g_mse
            })
    
    return metrics