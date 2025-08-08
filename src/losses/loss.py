
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
    if config.recon_loss_type == 'mse':
        # Reconstruction losses with MSE
        recon_loss_gene = F.mse_loss(outputs['recon_gene'], x_gene, reduction='mean')
        recon_loss_iso = F.mse_loss(outputs['recon_iso_joint'], x_iso, reduction='mean')
        cross_loss = F.mse_loss(outputs['recon_iso_cross'], x_iso, reduction='mean')
    elif config.recon_loss_type == 'gaussian':
        # Reconstruction losses with Gaussian NLL
        recon_loss_gene = gaussian_nll(
            outputs['recon_gene_mu'], outputs['recon_gene_logvar'], x_gene, 'mean'
        )
        recon_loss_iso = gaussian_nll(
            outputs['recon_iso_joint_mu'], outputs['recon_iso_joint_logvar'], x_iso, 'mean'
        )
        cross_loss = gaussian_nll(
            outputs['recon_iso_cross_mu'], outputs['recon_iso_cross_logvar'], x_iso, 'mean'
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
    
    # Weighted total loss
    total_loss = (recon_loss_gene +
                  recon_loss_iso +
                  config.beta * kl_joint +
                  config.gamma * cross_loss +
                  0.1 * config.beta * kl_gene +
                  0.1 * config.beta * kl_iso)

    return total_loss