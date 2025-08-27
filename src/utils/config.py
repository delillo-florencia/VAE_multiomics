from dataclasses import dataclass, field
import itertools

@dataclass
class Config:

    working_dir: str = "VAE_multiomics/data"  
    run_name: str = "run_name"
    model_type: str = "multimodal_no_joint"
    samples_train:str= "by_study/by_study_train_samples.txt"
    samples_val:str= "by_study/by_study_val_samples.txt"

    iso_from_gene : bool = True
    gene_from_iso : bool = True
    include_joint : bool = False
    feat_metrics: bool = False
    scaler: bool = False
    seed: int = 42
    loss_type="corr"
    recon_loss_type: str = "mse"
    n_genes: int = 50988
    n_isoforms: int = 214483
    latent_dim: int = 16
    hidden_dims: list = field(default_factory=lambda: [512])
    batch_size: int = 512
    learning_rate: float = 1e-2
    epochs: int = 50
    beta: float = 0.001
    gamma_i: float = 0.6
    gamma_g: float = 0.6
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    model_save_path: str = field(init=False)
    gene_scaler_path: str = "gene_scaler.pkl"
    iso_scaler_path: str = "iso_scaler.pkl"

    def __post_init__(self):
        self.model_save_path = f"multimodal_vae_{self.run_name}.pth"