import torch
import torch.nn as nn
from model.layers import Encoder, Decoder

class MultimodalVAE_NoJoint(nn.Module):
    """Multimodal VAE without joint latent space"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.iso_from_gene = config.iso_from_gene
        self.gene_from_iso = config.gene_from_iso

        # Encoders
        self.encoder_gene = Encoder(
            config.n_genes, config.hidden_dims, config.latent_dim
        )
        self.encoder_iso = Encoder(
            config.n_isoforms, config.hidden_dims, config.latent_dim
        )
        
        # Decoders with Gaussian output 
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

    def forward(self, x_gene, x_iso):
        # Encode both modalities
        mu_g, logvar_g = self.encoder_gene(x_gene)
        mu_i, logvar_i = self.encoder_iso(x_iso)
        
        # Sample from each latent space
        z_g = self.reparameterize(mu_g, logvar_g)
        z_i = self.reparameterize(mu_i, logvar_i)
        
        # Reconstruction from own latent space
        if self.config.recon_loss_type == 'gaussian':
            recon_gene_mu, recon_gene_logvar = self.decoder_gene(z_g)
            recon_iso_mu, recon_iso_logvar = self.decoder_iso(z_i)
        else:
            recon_gene = self.decoder_gene(z_g)
            recon_iso = self.decoder_iso(z_i)

        # Cross-modal reconstruction
        if self.gene_from_iso:
            if self.config.recon_loss_type == 'gaussian':
                recon_g_cross_mu, recon_g_cross_logvar = self.decoder_gene(z_i)
            else:
                recon_g_cross = self.decoder_gene(z_i)
        else:
            recon_g_cross = None
            recon_g_cross_mu = None
            recon_g_cross_logvar = None
            
        if self.iso_from_gene:
            if self.config.recon_loss_type == 'gaussian':
                recon_iso_cross_mu, recon_iso_cross_logvar = self.decoder_iso(z_g)
            else:
                recon_iso_cross = self.decoder_iso(z_g)
        else:
            recon_iso_cross = None
            recon_iso_cross_mu = None
            recon_iso_cross_logvar = None
            
        # Outputs
        outputs = {
            'mu_gene': mu_g,
            'logvar_gene': logvar_g,
            'mu_iso': mu_i,
            'logvar_iso': logvar_i
        }
        
        if self.config.recon_loss_type == 'gaussian':
            outputs.update({
                'recon_gene_mu': recon_gene_mu,
                'recon_gene_logvar': recon_gene_logvar,
                'recon_iso_mu': recon_iso_mu,
                'recon_iso_logvar': recon_iso_logvar,
                'recon_iso_cross_mu': recon_iso_cross_mu,
                'recon_iso_cross_logvar': recon_iso_cross_logvar,
                'recon_gene_cross_mu': recon_g_cross_mu,
                'recon_gene_cross_logvar': recon_g_cross_logvar
            })
        else:
            outputs.update({
                'recon_gene': recon_gene,
                'recon_gene_cross': recon_g_cross,
                'recon_iso': recon_iso,
                'recon_iso_cross': recon_iso_cross
            })
            
        return outputs

    def infer_isoforms(self, x_gene, deterministic=True):
        """Infer isoforms from gene expression only"""
        mu_g, logvar_g = self.encoder_gene(x_gene)
        z = mu_g if deterministic else self.reparameterize(mu_g, logvar_g)
        
        if self.config.recon_loss_type == 'gaussian':
            mu, _ = self.decoder_iso(z)
            return mu
        else:
            return self.decoder_iso(z)

    def infer_gene(self, x_iso, deterministic=True):
        """Infer genes from isoform expression only"""
        mu_i, logvar_i = self.encoder_iso(x_iso)
        z = mu_i if deterministic else self.reparameterize(mu_i, logvar_i)
        
        if self.config.recon_loss_type == 'gaussian':
            mu, _ = self.decoder_gene(z)
            return mu
        else:
            return self.decoder_gene(z)