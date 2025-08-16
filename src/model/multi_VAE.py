import torch
import torch.nn as nn
from model.layers import Encoder,Decoder

class MultimodalVAE(nn.Module):
    """Multimodal VAE for gene and isoform data """
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

        # Joint latent space (average)
        mu_joint = (mu_g + mu_i) / 2
        logvar_joint = (logvar_g + logvar_i) / 2
        z_joint = self.reparameterize(mu_joint, logvar_joint)

        # Reconstruction from joint
        if self.config.recon_loss_type == 'gaussian':
            recon_gene_mu, recon_gene_logvar = self.decoder_gene(z_joint)
            recon_iso_joint_mu, recon_iso_joint_logvar = self.decoder_iso(z_joint)
        else:
            recon_gene = self.decoder_gene(z_joint)
            recon_iso_joint = self.decoder_iso(z_joint)

        # Cross-modal reconstruction
        if self.gene_from_iso:
            # Use isoform encoder outputs to predict genes 
            z_i = self.reparameterize(mu_i, logvar_i)
            if self.config.recon_loss_type == 'gaussian':
                recon_g_cross_mu, recon_g_cross_logvar = self.decoder_gene(z_i)
            else:
                recon_g_cross = self.decoder_gene(z_i)
        else:
            recon_g_cross=None
            recon_g_cross_mu=None
            recon_g_cross_logvar=None
        if self.iso_from_gene:
            # Use gene encoder to predict isoforms
            z_g = self.reparameterize(mu_g, logvar_g)
            if self.config.recon_loss_type == 'gaussian':
                recon_iso_cross_mu, recon_iso_cross_logvar = self.decoder_iso(z_g)
            else:
                recon_iso_cross = self.decoder_iso(z_g)
        else:
            recon_iso_cross=None
            recon_iso_cross_mu=None
            recon_iso_cross_logvar=None
            

        #  outputs
        outputs = {
            'mu_joint': mu_joint,
            'logvar_joint': logvar_joint,
            'mu_gene': mu_g,
            'logvar_gene': logvar_g,
            'mu_iso': mu_i,
            'logvar_iso': logvar_i
        }
        
        if self.config.recon_loss_type == 'gaussian':
            outputs.update({
                'recon_gene_mu': recon_gene_mu,
                'recon_gene_logvar': recon_gene_logvar,
                'recon_iso_joint_mu': recon_iso_joint_mu,
                'recon_iso_joint_logvar': recon_iso_joint_logvar,
                'recon_iso_cross_mu': recon_iso_cross_mu,
                'recon_iso_cross_logvar': recon_iso_cross_logvar,
                'recon_gene_cross_mu': recon_g_cross_mu,
                'recon_gene_cross_logvar': recon_g_cross_logvar
            })
        else:
            outputs.update({
                'recon_gene_joint': recon_gene,
                'recon_gene_cross':recon_g_cross,
                'recon_iso_joint': recon_iso_joint,
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

    def infer_gene(self, x_isof, deterministic=True):
        """Infer isoforms from gene expression only"""
        mu_i, logvar_i = self.encoder_iso(x_isof)

        z = mu_i if deterministic else self.reparameterize(mu_i, logvar_i)

        if self.config.recon_loss_type == 'gaussian':
            mu, _ = self.decoder_gene(z)
            return mu
        else:
            return self.decoder_gene(z)