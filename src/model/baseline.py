import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GeneToIsoformModel(nn.Module):
    """Simple baseline: Encode genes, decode isoforms"""
    def __init__(self, n_genes, n_isoforms, hidden_dims, latent_dim):
        super().__init__()
        self.n_genes = n_genes
        self.n_isoforms = n_isoforms
        
        # Encoder for genes
        encoder_layers = []
        input_dim = n_genes
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            input_dim = h_dim
        
        # Latent layer
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder for isoforms
        decoder_layers = []
        decoder_dims = hidden_dims[::-1]  # Reverse hidden dims
        input_dim = latent_dim
        
        for h_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            input_dim = h_dim
        
        # Final output layer
        decoder_layers.append(nn.Linear(decoder_dims[-1], n_isoforms))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x_gene):
        # Encode genes
        latent = self.encoder(x_gene)
        # Decode to isoforms
        recon_iso = self.decoder(latent)
        return recon_iso
    
    def predict(self, x_gene):
        """Predict isoforms from gene expression"""
        self.eval()
        with torch.no_grad():
            return self.forward(x_gene)
