import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model.layers import Encoder,Decoder

class BaselineVAE(nn.Module):
    """Unimodal VAE using only gene data to reconstruct both genes and isoforms"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder for genes only
        self.encoder = Encoder(
            config.n_genes, config.hidden_dims, config.latent_dim
        )
        
        # Decoders for both modalities
        self.decoder_gene = Decoder(
            config.latent_dim, config.hidden_dims, config.n_genes, 
            gaussian_output=(config.recon_loss_type == 'gaussian')
        )
        self.decoder_iso = Decoder(
            config.latent_dim, config.hidden_dims, config.n_isoforms,
            gaussian_output=(config.recon_loss_type == 'gaussian')
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_gene):
        # Encode genes only
        mu, logvar = self.encoder(x_gene)
        z = self.reparameterize(mu, logvar)
        
        # Reconstruct genes and predict isoforms
        if self.config.recon_loss_type == 'gaussian':
            recon_gene_mu, recon_gene_logvar = self.decoder_gene(z)
            recon_iso_mu, recon_iso_logvar = self.decoder_iso(z)
            return {
                'mu': mu,
                'logvar': logvar,
                'recon_gene_mu': recon_gene_mu,
                'recon_gene_logvar': recon_gene_logvar,
                'recon_iso_mu': recon_iso_mu,
                'recon_iso_logvar': recon_iso_logvar
            }
        else:
            recon_gene = self.decoder_gene(z)
            recon_iso = self.decoder_iso(z)
            return {
                'mu': mu,
                'logvar': logvar,
                'recon_gene': recon_gene,
                'recon_iso': recon_iso
            }
    
    def infer_isoforms(self, x_gene, deterministic=True):
        """Predict isoforms from gene expression"""
        mu, logvar = self.encoder(x_gene)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        
        if self.config.recon_loss_type == 'gaussian':
            mu, _ = self.decoder_iso(z)
            return mu
        else:
            return self.decoder_iso(z)
        
