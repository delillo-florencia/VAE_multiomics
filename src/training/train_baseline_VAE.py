import torch
import torch.optim as optim
import numpy as np
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from losses.loss import multimodal_loss 
import os
import csv
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def train_baseline_vae(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create CSV logger
    csv_path = f"baseline_vae_metrics_{config.run_name}.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,"
                "gene_recon_loss,iso_recon_loss,kl_loss,"
                "gene_corr,gene_cosine,iso_corr,iso_cosine\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        gene_recon_loss = 0.0
        iso_recon_loss = 0.0
        kl_loss = 0.0
        
        for bg, bi in train_loader:
            bg, bi = bg.to(device), bi.to(device)
            optimizer.zero_grad()
            
            # Only gene input used
            outputs = model(bg)
            
            # Calculate losses
            if config.recon_loss_type == 'mse':
                gene_loss = F.mse_loss(outputs['recon_gene'], bg)
                iso_loss = F.mse_loss(outputs['recon_iso'], bi)
            else:
                gene_loss = gaussian_nll(
                    outputs['recon_gene_mu'], outputs['recon_gene_logvar'], bg, 'mean'
                )
                iso_loss = gaussian_nll(
                    outputs['recon_iso_mu'], outputs['recon_iso_logvar'], bi, 'mean'
                )
                
            kl = -0.5 * torch.mean(1 + outputs['logvar'] - 
                                    outputs['mu'].pow(2) - outputs['logvar'].exp())
            
            total_loss = gene_loss + iso_loss + config.beta * kl
            
            # Backprop
            total_loss.backward()
            optimizer.step()
            
            # Accumulate
            batch_size = bg.size(0)
            train_loss += total_loss.item() * batch_size
            gene_recon_loss += gene_loss.item() * batch_size
            iso_recon_loss += iso_loss.item() * batch_size
            kl_loss += kl.item() * batch_size
        
        # Calculate epoch averages
        n_train = len(train_loader.dataset)
        train_loss /= n_train
        gene_recon_loss /= n_train
        iso_recon_loss /= n_train
        kl_loss /= n_train
        
        # Validation
        model.eval()
        val_metrics = {
            'loss': 0.0, 
            'gene_corr': 0.0, 'gene_cosine': 0.0,
            'iso_corr': 0.0, 'iso_cosine': 0.0
        }
        
        with torch.no_grad():
            for bg, bi in val_loader:
                bg, bi = bg.to(device), bi.to(device)
                outputs = model(bg)
                
                # Calculate loss
                if config.recon_loss_type == 'mse':
                    gene_loss = F.mse_loss(outputs['recon_gene'], bg)
                    iso_loss = F.mse_loss(outputs['recon_iso'], bi)
                else:
                    gene_loss = gaussian_nll(
                        outputs['recon_gene_mu'], outputs['recon_gene_logvar'], bg, 'mean'
                    )
                    iso_loss = gaussian_nll(
                        outputs['recon_iso_mu'], outputs['recon_iso_logvar'], bi, 'mean'
                    )
                
                kl = -0.5 * torch.mean(1 + outputs['logvar'] - 
                                        outputs['mu'].pow(2) - outputs['logvar'].exp())
                
                total_val_loss = gene_loss + iso_loss + config.beta * kl
                val_metrics['loss'] += total_val_loss.item() * bg.size(0)
                
                # Calculate metrics
                gene_corr, gene_cos = calc_metrics(
                    outputs['recon_gene_mu'] if config.recon_loss_type == 'gaussian' else outputs['recon_gene'],
                    bg
                )
                iso_corr, iso_cos = calc_metrics(
                    outputs['recon_iso_mu'] if config.recon_loss_type == 'gaussian' else outputs['recon_iso'],
                    bi
                )
                
                val_metrics['gene_corr'] += gene_corr * bg.size(0)
                val_metrics['gene_cosine'] += gene_cos * bg.size(0)
                val_metrics['iso_corr'] += iso_corr * bg.size(0)
                val_metrics['iso_cosine'] += iso_cos * bg.size(0)
        
        # Final validation metrics
        n_val = len(val_loader.dataset)
        val_loss = val_metrics['loss'] / n_val
        gene_corr = val_metrics['gene_corr'] / n_val
        gene_cosine = val_metrics['gene_cosine'] / n_val
        iso_corr = val_metrics['iso_corr'] / n_val
        iso_cosine = val_metrics['iso_cosine'] / n_val
        
        # Save to CSV
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{gene_recon_loss:.6f},{iso_recon_loss:.6f},{kl_loss:.6f},"
                    f"{gene_corr:.6f},{gene_cosine:.6f},{iso_corr:.6f},{iso_cosine:.6f}\n")
        
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Gene Corr: {gene_corr:.4f} | Iso Corr: {iso_corr:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_baseline_vae_{config.run_name}.pt")
    
    return model