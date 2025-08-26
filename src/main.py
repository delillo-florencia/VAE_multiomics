from torch.utils.data import DataLoader, Dataset
#import itertool
import os
# -----Data imports---------------
from utils.config import Config
from data_preprocess.hash_files import *
from utils.datasets import *
# ----- Model imports ------------
from model.multi_VAE import *
from model.baseline import GeneToIsoformModel
from model.baseline_VAE import BaselineVAE
from training.train import train_model
from training.train_baseline import train_baseline
from training.train_baseline_VAE import train_baseline_vae
from model.multi_VAE_no_joint import MultimodalVAE_NoJoint
from training.train_no_joint import train_model_no_joint

def read_ids(split_txt):
    with open(split_txt) as f:
        return [line.strip() for line in f if line.strip()]

def run_experiment(config):
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    working_dir=config.working_dir
    
    samples_train=config.samples_train
    samples_val=config.samples_val
    
    #Read train/val/test splits
    train_ids = read_ids(f'{working_dir}/samples/{samples_train}')
    val_ids   = read_ids(f'{working_dir}/samples/{samples_val}')
    
    print(f"Loading {len(train_ids)} samples for training & {len(val_ids)} for validation ")
    
    # Fit scaler on training gene expression data
    if config.scaler:
        scaler_genes='scaler_genes.pkl'
        scaler_isof='scaler_isoforms.pkl'
    else:
        scaler_genes=None
        scaler_isof=None
        
    print("Building Dataset...",flush=True)
    train_ds_genes = ExpressionDataset(f'{working_dir}/h5/genes.h5', train_ids, scaler_genes )
    val_ds_genes   = ExpressionDataset(f'{working_dir}/h5/genes.h5', val_ids,  scaler_genes)

    train_ds_iso   = ExpressionDataset(f'{working_dir}/h5/isoforms.h5', train_ids, scaler_isof)
    val_ds_iso   = ExpressionDataset(f'{working_dir}/h5/isoforms.h5', val_ids, scaler_isof)

    print("Building Dataloaders...",flush=True)
    train_dataset = MultiOmicsDataset(train_ds_genes, train_ds_iso)
    val_dataset = MultiOmicsDataset(val_ds_genes, val_ds_iso)

    n_samples = len(train_ds_genes)
    n_gene_feats = train_ds_genes[0].shape[0]
    n_iso_feats  = train_ds_iso[0].shape[0]

    print(f"Train genes  : ({n_samples}, {n_gene_feats})",flush=True)
    print(f"Train isoform: ({n_samples}, {n_iso_feats})",flush=True)

    n_samples = len(val_ds_genes)
    n_gene_feats = val_ds_genes[0].shape[0]
    n_iso_feats  = val_ds_iso[0].shape[0]

    print(f"Val genes  : ({n_samples}, {n_gene_feats})",flush=True)
    print(f"Val isoform: ({n_samples}, {n_iso_feats})",flush=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    if config.model_type == 'baseline':
        
        print("Initializing baseline model...", flush=True)
        model = GeneToIsoformModel(
            n_genes=config.n_genes,
            n_isoforms=config.n_isoforms,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim
        ).to(device)
        
        # Train baseline
        print("Training baseline model...", flush=True)
        train_baseline(model, train_loader, val_loader, config, device)
        
    elif config.model_type == 'baseline_vae':
        print("Initializing baseline VAE...", flush=True)
        model = BaselineVAE(config).to(device)
        print("Training baseline VAE...", flush=True)
        model = train_baseline_vae(model, train_loader, val_loader, config, device)

    elif config.model_type == 'multimodal_no_joint':
        print("Initializing multimodal VAE, NO joint...", flush=True)
        model = MultimodalVAE_NoJoint(config).to(device)
        
        # Train multimodal VAE without joint latent space
        print("Training multimodal VAE without joint latent space...", flush=True)
        model = train_model_no_joint(model, train_loader, val_loader, config, device)

    elif config.model_type == 'multimodal_vae':
        print("Initializing multimodal VAE...", flush=True)
        model = MultimodalVAE(config).to(device)
        
        # Train multimodal VAE
        print("Training multimodal VAE...", flush=True)
        train_model(model, train_loader, val_loader, config, device)    

    else:
        print("No model with that name")
        return False
    # Save resources
    print("Saving resources...",flush=True)
    os.makedirs("model_resources", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("model_resources", "model.pth"))

    print("Workflow completed successfully!")
    
    return True





def main():
    latent_dims = [64, 32]
    hidden_dims_list =[[1024], [1024, 512, 256], [512, 256]]
    batch_sizes=[64, 256]
    betas = [0.01]
    gammas = [0.6]
    loss_types = ["mse", "gaussian"]
    joint_flags = [False]  

    # All combinations
    combinations = list(itertools.product(latent_dims,batch_sizes, hidden_dims_list, betas, gammas, loss_types, joint_flags))

    for i, (ld, b_size,h_dims, b, g, loss, joint_flag) in enumerate(combinations):
        model_name = "isof_from_joint" if joint_flag else "isof_from_ge"
        run_name = f"run_{i}_{model_name}_ld{ld}_b{b}_g{g}_loss{loss}_hd{len(h_dims)}"

        config = Config(
            run_name=run_name,
            latent_dim=ld,
            hidden_dims=h_dims,
            batch_size=b_size,
            beta=b,
            gamma=g,
            recon_loss_type=loss,
        )
        try:
            print("Starting new combination",flush=True)
            run_experiment(config)
        except:
            print(f"Error when running combination{i}",flush=True)

if __name__ == "__main__":
    config=Config()
    run_experiment(config)
    #main()