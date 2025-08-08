import torch
import torch.optim as optim
import numpy as np
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from losses.loss import multimodal_loss 

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
        "run_name", "use_joint_for_isoform_prediction", "scaler", "n_genes", "n_isoforms",
        "latent_dim", "hidden_dims", "batch_size", "learning_rate", "epochs",
        "beta", "gamma", "recon_loss_type"
    ]
    metric_fields = [
        "epoch", "train_loss", "val_loss",
        "gene_recon_corr", "gene_recon_cosine",
        "iso_joint_recon_corr", "iso_joint_recon_cosine",
        "iso_cross_recon_corr", "iso_cross_recon_cosine"
    ]
    header = cfg_fields + metric_fields
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")

    metrics = {m: [] for m in metric_fields}

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for bg, bi in train_loader:
            bg, bi = bg.to(device), bi.to(device)
            optimizer.zero_grad()
            out = model(bg, bi)
            loss = multimodal_loss(out, bg, bi, config)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * bg.size(0)  # Multiply by batch size
            
        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        sums = {m: 0.0 for m in metric_fields if m not in ("epoch", "train_loss", "val_loss")}
        n_val = 0

        with torch.no_grad():
            for bg, bi in val_loader:
                bg, bi = bg.to(device), bi.to(device)
                out = model(bg, bi)
                loss = multimodal_loss(out, bg, bi, config)
                total_val_loss += loss.item() * bg.size(0)  # Multiply by batch size

                bsz = bg.size(0)
                n_val += bsz

                # Get predictions based on loss type
                if config.recon_loss_type == 'gaussian':
                    pred_gene = out['recon_gene_mu']
                    pred_iso_joint = out['recon_iso_joint_mu']
                    pred_iso_cross = out['recon_iso_cross_mu']
                else:
                    pred_gene = out['recon_gene']
                    pred_iso_joint = out['recon_iso_joint']
                    pred_iso_cross = out['recon_iso_cross']

                def calc(pred, tgt):
                    p = pred.view(bsz, -1)
                    t = tgt.view(bsz, -1)
                    pc = p - p.mean(1, keepdim=True)
                    tc = t - t.mean(1, keepdim=True)
                    num = (pc * tc).sum(1)
                    den = (pc.norm(2, 1) * tc.norm(2, 1)).clamp(min=1e-6)
                    corr = (num / den).mean().item()
                    cos = F.cosine_similarity(p, t, dim=1).mean().item()
                    return corr, cos

                gc, gcos = calc(pred_gene, bg)
                icj, icosj = calc(pred_iso_joint, bi)
                icx, icosx = calc(pred_iso_cross, bi)

                sums["gene_recon_corr"] += gc * bsz
                sums["gene_recon_cosine"] += gcos * bsz
                sums["iso_joint_recon_corr"] += icj * bsz
                sums["iso_joint_recon_cosine"] += icosj * bsz
                sums["iso_cross_recon_corr"] += icx * bsz
                sums["iso_cross_recon_cosine"] += icosx * bsz

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
            str(config.use_joint_for_isoform_prediction),
            str(config.scaler),
            str(config.n_genes),
            str(config.n_isoforms),
            str(config.latent_dim),
            json.dumps(config.hidden_dims),
            str(config.batch_size),
            str(config.learning_rate),
            str(config.epochs),
            str(config.beta),
            str(config.gamma),
            config.recon_loss_type
        ]
        row_vals = [f"{row[m]:.6f}" if isinstance(row[m], float) else str(row[m])
                    for m in metric_fields]
        with open(csv_path, "a") as f:
            f.write(",".join(cfg_vals + row_vals) + "\n")

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
    model.load_state_dict(torch.load(best_model_path))

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