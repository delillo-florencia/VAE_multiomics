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
