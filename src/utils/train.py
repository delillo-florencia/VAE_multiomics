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


def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(model, train_loader, val_loader, config, device):
    set_seed(config.seed)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    csv_path = f"training_metrics_{config.run_name}.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_path = f"best_model_{config.run_name}.pt"

    cfg_fields = [
        "run_name", "isof_from_gene","gene_from_iso","include_joint", "scaler", "n_genes", "n_isoforms",
        "latent_dim", "hidden_dims", "batch_size", "learning_rate", "epochs",
        "beta", "gamma_i","gamma_g", "recon_loss_type"
    ]
    metric_fields = [
        "epoch", "train_loss", "val_loss",
        "gene_recon_corr", "gene_recon_cosine",
        "iso_joint_recon_corr", "iso_joint_recon_cosine",
        "iso_cross_recon_corr", "iso_cross_recon_cosine",
        "gene_cross_recon_corr", "gene_cross_recon_cosine"
    ]
    header = cfg_fields + metric_fields
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")

    metrics = {m: [] for m in metric_fields}


    os.makedirs('logs', exist_ok=True)

    # Loss csv
    csv_columns = [
        'epoch', 
        'total_loss',
        'weighted_recon_gene', 'weighted_recon_iso', 'weighted_kl_joint',
        'weighted_cross_iso', 'weighted_cross_gene',
        'weighted_kl_gene', 'weighted_kl_iso',
        'raw_recon_gene', 'raw_recon_iso', 'raw_cross_iso', 'raw_cross_gene',
        'raw_kl_joint', 'raw_kl_gene', 'raw_kl_iso'
    ]


    # Create loss tracking file
    loss_csv_path = f'logs/loss_tracking_{config.run_name}.csv'
    with open(loss_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        epoch_loss_components = {key: 0.0 for key in csv_columns[1:]}  # Initialize all components
        
        for bg, bi in train_loader:
            bg, bi = bg.to(device), bi.to(device)
            optimizer.zero_grad()
            out = model(bg, bi)
            loss, loss_components = multimodal_loss(out, bg, bi, config)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss components
            batch_size = bg.size(0)
            total_train_loss += loss.item() * batch_size
            
            # Accumulate each loss component
            for key in epoch_loss_components:
                if key in loss_components:
                    epoch_loss_components[key] += loss_components[key].item() * batch_size
                    
        # Calculate average losses for the epoch
        n_train = len(train_loader.dataset)
        train_loss = total_train_loss / n_train
        epoch_losses = {'epoch': epoch, 'total_loss': train_loss}
        for key in epoch_loss_components:
            epoch_losses[key] = epoch_loss_components[key] / n_train
            
        # Write to loss tracking CSV
        with open(loss_csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(epoch_losses)
            
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        sums = {m: 0.0 for m in metric_fields if m not in ("epoch", "train_loss", "val_loss")}
        
        with torch.no_grad():
            for bg, bi in val_loader:
                bg, bi = bg.to(device), bi.to(device)
                out = model(bg, bi)
                loss, _ = multimodal_loss(out, bg, bi, config)
                total_val_loss += loss.item() * bg.size(0)  # Multiply by batch size

                bsz = bg.size(0)
                n_val += bsz

                # Get predictions based on loss type
                if config.recon_loss_type == 'gaussian':
                    pred_gene = out['recon_gene_mu']
                    pred_iso_joint = out['recon_iso_joint_mu']
                    pred_iso_cross = out['recon_iso_cross_mu']
                    pred_gene_cross = out['recon_gene_cross_mu']
                else:
                    pred_gene = out['recon_gene_joint']
                    pred_iso_joint = out['recon_iso_joint']
                    pred_iso_cross = out['recon_iso_cross']
                    pred_gene_cross = out['recon_gene_cross']

                # Only compute metrics if predictions exist
                if pred_gene is not None:
                    gc, gcos = calc(pred_gene, bg,bsz)
                else:
                    gc, gcos = 0.0, 0.0
                    
                if pred_iso_joint is not None:
                    icj, icosj = calc(pred_iso_joint, bi,bsz)
                else:
                    icj, icosj = 0.0, 0.0
                    
                if pred_iso_cross is not None:
                    icx, icosx = calc(pred_iso_cross, bi,bsz)
                else:
                    icx, icosx = 0.0, 0.0
                    
                if pred_gene_cross is not None:
                    gcx, gcosx = calc(pred_gene_cross, bg,bsz)
                else:
                    gcx, gcosx = 0.0, 0.0





                sums["gene_recon_corr"] += gc * bsz
                sums["gene_recon_cosine"] += gcos * bsz
                sums["iso_joint_recon_corr"] += icj * bsz
                sums["iso_joint_recon_cosine"] += icosj * bsz
                sums["iso_cross_recon_corr"] += icx * bsz
                sums["iso_cross_recon_cosine"] += icosx * bsz
                sums["gene_cross_recon_corr"] += gcx * bsz
                sums["gene_cross_recon_cosine"] += gcosx * bsz

        val_loss = total_val_loss / len(val_loader.dataset)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        for m in sums:
            row[m] = sums[m] / n_val
            metrics[m].append(row[m])
        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)

        cfg_vals = [
            config.run_name,
            str(config.iso_from_gene),
            str(config.gene_from_iso),
            str(config.include_joint),
            str(config.scaler),
            str(config.n_genes),
            str(config.n_isoforms),
            str(config.latent_dim),
            json.dumps(config.hidden_dims),
            str(config.batch_size),
            str(config.learning_rate),
            str(config.epochs),
            str(config.beta),
            str(config.gamma_i),
            str(config.gamma_g),
            config.recon_loss_type
        ]
        row_vals = [f"{row[m]:.6f}" if isinstance(row[m], float) else str(row[m])
                    for m in metric_fields]
        with open(csv_path, "a") as f:
            f.write(",".join(cfg_vals + row_vals) + "\n")
        with open(f'logs/loss_tracking_{config.run_name}.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(epoch_losses)
        print(f"Epoch {epoch}/{config.epochs} "
              f"Train {train_loss:.4f}  Val {val_loss:.4f}  "
              f"GeneCorr {row['gene_recon_corr']:.4f}")

        if val_loss + config.early_stopping_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss = {best_val_loss:.6f} (epoch {best_epoch})")
                break
    
    # Load best model
    if config.feat_metrics:
        
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        all_gene_joint_preds = []
        all_gene_joint_targets = []
        all_iso_joint_preds = []
        all_iso_joint_targets = []
        all_gene_cross_preds = []
        all_gene_cross_targets = []
        all_iso_cross_preds = []
        all_iso_cross_targets = []
        
        with torch.no_grad():
            for bg, bi in val_loader:
                bg, bi = bg.to(device), bi.to(device)
                out = model(bg, bi)
                
                # Get predictions
                if config.recon_loss_type == 'gaussian':
                    gene_joint_pred = out['recon_gene_mu']
                    iso_joint_pred = out['recon_iso_joint_mu']
                    gene_cross_pred = out['recon_gene_cross_mu']
                    iso_cross_pred = out['recon_iso_cross_mu']
                else:
                    gene_joint_pred = out['recon_gene_joint']
                    iso_joint_pred = out['recon_iso_joint']
                    gene_cross_pred = out['recon_gene_cross']
                    iso_cross_pred = out['recon_iso_cross']
                
                # Store predictions and targets
                all_gene_joint_preds.append(gene_joint_pred)
                all_gene_joint_targets.append(bg)
                all_iso_joint_preds.append(iso_joint_pred)
                all_iso_joint_targets.append(bi)
                all_gene_cross_preds.append(gene_cross_pred)
                all_gene_cross_targets.append(bg)
                all_iso_cross_preds.append(iso_cross_pred)
                all_iso_cross_targets.append(bi)
        
        # Concatenate all batches
        gene_joint_preds = torch.cat(all_gene_joint_preds, dim=0)
        gene_joint_targets = torch.cat(all_gene_joint_targets, dim=0)
        iso_joint_preds = torch.cat(all_iso_joint_preds, dim=0)
        iso_joint_targets = torch.cat(all_iso_joint_targets, dim=0)
        gene_cross_preds = torch.cat(all_gene_cross_preds, dim=0)
        gene_cross_targets = torch.cat(all_gene_cross_targets, dim=0)
        iso_cross_preds = torch.cat(all_iso_cross_preds, dim=0)
        iso_cross_targets = torch.cat(all_iso_cross_targets, dim=0)
        
        # Compute per-feature metrics
        print("Computing per-feature metrics for gene joint reconstruction...")
        gene_joint_corr, gene_joint_cos = compute_per_feature_metrics(gene_joint_preds, gene_joint_targets)
        
        print("Computing per-feature metrics for isoform joint reconstruction...")
        iso_joint_corr, iso_joint_cos = compute_per_feature_metrics(iso_joint_preds, iso_joint_targets)
        
        print("Computing per-feature metrics for gene cross reconstruction...")
        gene_cross_corr, gene_cross_cos = compute_per_feature_metrics(gene_cross_preds, gene_cross_targets)
        
        print("Computing per-feature metrics for isoform cross reconstruction...")
        iso_cross_corr, iso_cross_cos = compute_per_feature_metrics(iso_cross_preds, iso_cross_targets)
        
        # Save per-feature metrics to CSV
        per_feature_csv = f"best_model_per_feature_metrics_{config.run_name}.csv"
        with open(per_feature_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'feature_idx', 'feature_type', 'reconstruction_type',
                'correlation', 'cosine_similarity'
            ])
            
            # Write gene joint metrics
            for i in range(config.n_genes):
                writer.writerow([
                    i, 'gene', 'joint',
                    gene_joint_corr[i], gene_joint_cos[i]
                ])
                
            # Write isoform joint metrics
            for i in range(config.n_isoforms):
                writer.writerow([
                    i, 'isoform', 'joint',
                    iso_joint_corr[i], iso_joint_cos[i]
                ])
                
            # Write gene cross metrics
            for i in range(config.n_genes):
                writer.writerow([
                    i, 'gene', 'cross',
                    gene_cross_corr[i], gene_cross_cos[i]
                ])
                
            # Write isoform cross metrics
            for i in range(config.n_isoforms):
                writer.writerow([
                    i, 'isoform', 'cross',
                    iso_cross_corr[i], iso_cross_cos[i]
                ])
        
        print(f"Saved per-feature metrics to {per_feature_csv}")
    
    #----------Plotting-----------
    epochs = metrics["epoch"]
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.axvline(best_epoch, color='red', linestyle='--', label='Best Epoch')
    plt.title("Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, metrics["gene_recon_corr"], label="Gene")
    plt.plot(epochs, metrics["iso_joint_recon_corr"], label="IsoJoint")
    plt.plot(epochs, metrics["iso_cross_recon_corr"], label="IsoCross")
    plt.axvline(best_epoch, color='red', linestyle='--')
    plt.title("Pearson Corr")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, metrics["gene_recon_cosine"], label="Gene")
    plt.plot(epochs, metrics["iso_joint_recon_cosine"], label="IsoJoint")
    plt.plot(epochs, metrics["iso_cross_recon_cosine"], label="IsoCross")
    plt.axvline(best_epoch, color='red', linestyle='--')
    plt.title("Cosine Sim")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_metrics_{config.run_name}.png")
    plt.close()

    return model