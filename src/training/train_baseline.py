import torch
import torch.optim as optim
from utils.seeds import set_seed
from metrics.metrics import calc_metrics

def train_baseline(model, train_loader, val_loader, config, device):
    """Train gene-to-isoform baseline model"""
    set_seed(config.seed)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    
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
            recon_iso = model(bg) # decode isoforms from gene data
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

