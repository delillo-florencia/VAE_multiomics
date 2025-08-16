
import torch
import numpy as np
import joblib
import math
import torch.nn.functional as F

# Gaussian negative log-likelihood loss
def gaussian_nll(mu, logvar, target, reduction='mean'):
    """Numerically stable negative log-likelihood for Gaussian distribution"""
    logvar = torch.clamp(logvar, min=-10, max=10)  # Prevents extreme variance values
    var = torch.exp(logvar)
    
    # Compute negative log-likelihood 
    nll = 0.5 * (logvar + (target - mu).pow(2) / var + np.log(2 * np.pi))
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll

def multimodal_loss(outputs, x_gene, x_iso, config):
    """Calculate multimodal VAE loss with mean reduction"""
    # Get device from inputs
    device = x_gene.device
    
    # Initialize as tensors instead of floats
    cross_loss_iso = torch.tensor(0.0, device=device)
    cross_loss_gene = torch.tensor(0.0, device=device)

    # Reconstruction losses
    if config.recon_loss_type == 'mse':
        recon_loss_gene = F.mse_loss(outputs['recon_gene_joint'], x_gene, reduction='mean')
        recon_loss_iso = F.mse_loss(outputs['recon_iso_joint'], x_iso, reduction='mean')
        
        # Handle cross-modal reconstructions
        if config.iso_from_gene and outputs['recon_iso_cross'] is not None:
            cross_loss_iso = F.mse_loss(outputs['recon_iso_cross'], x_iso, reduction='mean')
        
        if config.gene_from_iso and outputs['recon_gene_cross'] is not None:
            cross_loss_gene = F.mse_loss(outputs['recon_gene_cross'], x_gene, reduction='mean')

    elif config.recon_loss_type == 'gaussian':
        recon_loss_gene = gaussian_nll(
            outputs['recon_gene_mu'], outputs['recon_gene_logvar'], x_gene, 'mean'
        )
        recon_loss_iso = gaussian_nll(
            outputs['recon_iso_joint_mu'], outputs['recon_iso_joint_logvar'], x_iso, 'mean'
        )
        
        # Handle cross-modal reconstructions
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

    # KL divergences (mean-reduced)
    kl_joint = -0.5 * torch.mean(1 + outputs['logvar_joint'] - 
                                 outputs['mu_joint'].pow(2) - outputs['logvar_joint'].exp())
    kl_gene = -0.5 * torch.mean(1 + outputs['logvar_gene'] - 
                                outputs['mu_gene'].pow(2) - outputs['logvar_gene'].exp())
    kl_iso = -0.5 * torch.mean(1 + outputs['logvar_iso'] - 
                               outputs['mu_iso'].pow(2) - outputs['logvar_iso'].exp())
    
    # Set gamma values based on config
    gamma_g = config.gamma_g if config.gene_from_iso else 0
    gamma_i = config.gamma_i if config.iso_from_gene else 0
    gamma_joint = 0 if config.include_joint else 1
                
    # Compute weighted terms
    weighted_recon_gene = gamma_joint * recon_loss_gene
    weighted_recon_iso = gamma_joint * recon_loss_iso
    weighted_kl_joint = gamma_joint * config.beta * kl_joint
    weighted_cross_iso = gamma_i * cross_loss_iso
    weighted_cross_gene = gamma_g * cross_loss_gene
    weighted_kl_gene = 0.1 * config.beta * kl_gene
    weighted_kl_iso = 0.1 * config.beta * kl_iso

    total_loss = (
        weighted_recon_gene + 
        weighted_recon_iso + 
        weighted_kl_joint +
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
        'weighted_kl_joint': weighted_kl_joint,
        'weighted_cross_iso': weighted_cross_iso,
        'weighted_cross_gene': weighted_cross_gene,
        'weighted_kl_gene': weighted_kl_gene,
        'weighted_kl_iso': weighted_kl_iso,
        'raw_recon_gene': recon_loss_gene,
        'raw_recon_iso': recon_loss_iso,
        'raw_cross_iso': cross_loss_iso,
        'raw_cross_gene': cross_loss_gene,
        'raw_kl_joint': kl_joint,
        'raw_kl_gene': kl_gene,
        'raw_kl_iso': kl_iso,
    }
    
    return total_loss, loss_components