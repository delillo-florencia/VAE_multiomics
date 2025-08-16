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

def train_baseline(model, train_loader, val_loader, config, device):
    """Train gene-to-isoform baseline model"""
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    best_model_path = f"best_baseline_{config.run_name}.pt"
    
    # Create CSV for logging
    csv_path = f"baseline_metrics_{config.run_name}.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,corr,cosine\n")
    
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            bg, bi = batch
            bg, bi = bg.to(device), bi.to(device)
            
            optimizer.zero_grad()
            recon_iso = model(bg)
            loss = criterion(recon_iso, bi)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * bg.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                bg, bi = batch
                bg, bi = bg.to(device), bi.to(device)
                recon_iso = model(bg)
                loss = criterion(recon_iso, bi)
                val_loss += loss.item() * bg.size(0)
                
                all_preds.append(recon_iso)
                all_targets.append(bi)
        
        val_loss /= len(val_loader.dataset)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # Calculate metrics
        corr, cosine = calc_metrics(all_preds, all_targets)
        
        # Save metrics
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{corr:.6f},{cosine:.6f}\n")
        
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Corr: {corr:.4f} | Cosine: {cosine:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
    
    # Load best model for final metrics
    model.load_state_dict(torch.load(best_model_path))
    return model

def calc_metrics(pred, target):
    """Calculate correlation and cosine similarity"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Flatten all samples and features
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    # Pearson correlation
    cov = np.cov(pred_flat, target_flat)
    corr = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]) + 1e-8)
    
    # Cosine similarity
    cos_sim = np.dot(pred_flat, target_flat) / (
        np.linalg.norm(pred_flat) * np.linalg.norm(target_flat) + 1e-8)
    
    return corr, cos_sim