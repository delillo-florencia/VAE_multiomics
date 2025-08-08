import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder network for a single modality"""
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            input_dim = h_dim
        
        self.features = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        x = self.features(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for a single modality"""
    def __init__(self, latent_dim, hidden_dims, output_dim, gaussian_output=False):
        super().__init__()
        self.gaussian_output = gaussian_output
        layers = []
        hidden_dims = hidden_dims[::-1]
        
        # First layer
        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Intermediate layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU()
            ])
        
        # Final output layer
        self.hidden = nn.Sequential(*layers)
        if gaussian_output:
            self.fc_mu = nn.Linear(hidden_dims[-1], output_dim)
            self.fc_logvar = nn.Linear(hidden_dims[-1], output_dim)
        else:
            self.fc = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, z):
        hidden = self.hidden(z)
        if self.gaussian_output:
            mu = self.fc_mu(hidden)
            logvar = self.fc_logvar(hidden)
            return mu, logvar
        else:
            return self.fc(hidden)


class MultimodalVAE(nn.Module):
    """Multimodal VAE for gene and isoform data """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_joint_for_isoform_prediction = config.use_joint_for_isoform_prediction

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

        # Reconstruction 
        if self.config.recon_loss_type == 'gaussian':
            recon_gene_mu, recon_gene_logvar = self.decoder_gene(z_joint)
            recon_iso_joint_mu, recon_iso_joint_logvar = self.decoder_iso(z_joint)
        else:
            recon_gene = self.decoder_gene(z_joint)
            recon_iso_joint = self.decoder_iso(z_joint)

        # Cross-modal reconstruction
        if self.use_joint_for_isoform_prediction:
            # Use only gene modality to compute joint latent
            z_fake_joint = self.reparameterize(mu_g, logvar_g)
            if self.config.recon_loss_type == 'gaussian':
                recon_iso_cross_mu, recon_iso_cross_logvar = self.decoder_iso(z_fake_joint)
            else:
                recon_iso_cross = self.decoder_iso(z_fake_joint)
        else:
            # Use gene encoder directly
            z_g = self.reparameterize(mu_g, logvar_g)
            if self.config.recon_loss_type == 'gaussian':
                recon_iso_cross_mu, recon_iso_cross_logvar = self.decoder_iso(z_g)
            else:
                recon_iso_cross = self.decoder_iso(z_g)

        # Prepare outputs
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
                'recon_iso_cross_logvar': recon_iso_cross_logvar
            })
        else:
            outputs.update({
                'recon_gene': recon_gene,
                'recon_iso_joint': recon_iso_joint,
                'recon_iso_cross': recon_iso_cross
            })
            
        return outputs

    def infer_isoforms(self, x_gene, deterministic=True, use_joint=False):
        """Infer isoforms from gene expression only"""
        mu_g, logvar_g = self.encoder_gene(x_gene)

        if use_joint:
            z = mu_g if deterministic else self.reparameterize(mu_g, logvar_g)
        else:
            z = mu_g if deterministic else self.reparameterize(mu_g, logvar_g)

        if self.config.recon_loss_type == 'gaussian':
            mu, _ = self.decoder_iso(z)
            return mu
        else:
            return self.decoder_iso(z)
