import torch
import numpy as np
import joblib
import math
import torch.nn.functional as F
from losses.loss import gaussian_nll
def multimodal_loss_no_joint(outputs, x_gene, x_iso, config):
    """Calculate multimodal VAE loss without joint latent space"""
    device = x_gene.device
    
    # Initialize cross losses as tensors
    cross_loss_iso = torch.tensor(0.0, device=device)
    cross_loss_gene = torch.tensor(0.0, device=device)

    # Reconstruction losses (within modality)
    if config.recon_loss_type == 'mse':
        recon_loss_gene = F.mse_loss(outputs['recon_gene'], x_gene, reduction='mean')
        recon_loss_iso = F.mse_loss(outputs['recon_iso'], x_iso, reduction='mean')
        
        # Cross-modal reconstruction losses
        if config.iso_from_gene and outputs['recon_iso_cross'] is not None:
            cross_loss_iso = F.mse_loss(outputs['recon_iso_cross'], x_iso, reduction='mean')
        
        if config.gene_from_iso and outputs['recon_gene_cross'] is not None:
            cross_loss_gene = F.mse_loss(outputs['recon_gene_cross'], x_gene, reduction='mean')

    elif config.recon_loss_type == 'gaussian':
        recon_loss_gene = gaussian_nll(
            outputs['recon_gene_mu'], outputs['recon_gene_logvar'], x_gene, 'mean'
        )
        recon_loss_iso = gaussian_nll(
            outputs['recon_iso_mu'], outputs['recon_iso_logvar'], x_iso, 'mean'
        )
        
        # Cross-modal reconstruction losses
        if config.iso_from_gene and outputs.get('recon_iso_cross_mu') is not None:
            cross_loss_iso = gaussian_nll(
                outputs['recon_iso_cross_mu'], outputs['recon_iso_cross_logvar'], x_iso, 'mean'
            )
        
        if config.gene_from_iso and outputs.get('recon_gene_cross_mu') is not None:
            cross_loss_gene = gaussian_nll(
                outputs['recon_gene_cross_mu'], outputs['recon_gene_cross_logvar'], x_gene, 'mean'
            )
    else:
        raise ValueError(f"Invalid recon_loss_type: {config.recon_loss_type}")

    # KL divergences (separate for each modality)
    kl_gene = -0.5 * torch.mean(1 + outputs['logvar_gene'] - 
                               outputs['mu_gene'].pow(2) - outputs['logvar_gene'].exp())
    kl_iso = -0.5 * torch.mean(1 + outputs['logvar_iso'] - 
                              outputs['mu_iso'].pow(2) - outputs['logvar_iso'].exp())
    
    # Set gamma values based on config
    gamma_g = config.gamma_g if config.gene_from_iso else 0
    gamma_i = config.gamma_i if config.iso_from_gene else 0
    
    # Compute weighted terms
    weighted_recon_gene = recon_loss_gene
    weighted_recon_iso = recon_loss_iso
    weighted_cross_iso = gamma_i * cross_loss_iso
    weighted_cross_gene = gamma_g * cross_loss_gene
    weighted_kl_gene = config.beta * kl_gene
    weighted_kl_iso = config.beta * kl_iso

    total_loss = (
        weighted_recon_gene + 
        weighted_recon_iso + 
        weighted_cross_iso + 
        weighted_cross_gene +
        weighted_kl_gene + 
        weighted_kl_iso
    )
    
    # Return loss components
    loss_components = {
        'total': total_loss,
        'weighted_recon_gene': weighted_recon_gene,
        'weighted_recon_iso': weighted_recon_iso,
        'weighted_cross_iso': weighted_cross_iso,
        'weighted_cross_gene': weighted_cross_gene,
        'weighted_kl_gene': weighted_kl_gene,
        'weighted_kl_iso': weighted_kl_iso,
        'raw_recon_gene': recon_loss_gene,
        'raw_recon_iso': recon_loss_iso,
        'raw_cross_iso': cross_loss_iso,
        'raw_cross_gene': cross_loss_gene,
        'raw_kl_gene': kl_gene,
        'raw_kl_iso': kl_iso,
    }
    
    return total_loss, loss_components