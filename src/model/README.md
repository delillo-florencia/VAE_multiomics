
This repository contains implementations of four models for joint analysis of gene and isoform expression data:

1. **Multimodal Variational Autoencoder (VAE)**
2. **Baseline VAE (Unimodal)**
3. **Baseline Feedforward Network (FFN)**

## Table of Contents
- [Model Overview](#model-overview)
- [Multimodal VAE](#1-multimodal-vae)
- [Baseline VAE](#2-baseline-vae-unimodal)
- [Baseline FFN](#3-baseline-ffn-feedforward-network)
- [Model Comparison](#model-comparison)
- [Usage Recommendations](#usage-recommendations)
- [Getting Started](#getting-started)

## Model Overview

### 1. Multimodal VAE

![Model Architecture Diagram](docs/architecture.png)


**Architecture**:  
Dual encoders + Joint latent space + Dual decoders with cross-modal pathways  

**Input**:  
Both gene and isoform expression data  

**Output**:  
Joint reconstructions and cross-modal predictions  

**Key Features**:
- Learns rich joint representations of both modalities
- Models uncertainty through variational inference
- Supports both MSE and Gaussian reconstruction losses
- Optional cross-modal prediction pathways
- Flexible configuration via hyperparameters

**Use Cases**:
- Joint analysis of gene and isoform data
- Cross-modal prediction (genes → isoforms and isoforms → genes)
- Learning biologically meaningful latent representations
- Handling missing modalities during inference

**Hyperparameters**:
- `latent_dim`: Dimension of joint latent space (default: 64)
- `beta`: KL divergence weight (default: 0.01)
- `gamma_i`, `gamma_g`: Cross-modal loss weights (default: 0.6)
- `recon_loss_type`: 'mse' or 'gaussian'
- `hidden_dims`: Encoder/decoder hidden dimensions (default: [1024, 512])

### 2. Baseline VAE (Unimodal)

**Architecture**:  
Single encoder + Dual decoders  

**Input**:  
Gene expression data only  

**Output**:  
Gene reconstruction and isoform prediction  

**Key Features**:
- Uses only gene data as input
- Simpler architecture than multimodal VAE
- Probabilistic modeling of latent space
- Direct gene→isoform mapping
- Same reconstruction loss options as multimodal VAE

**Use Cases**:
- When isoform data is unavailable at inference time
- Resource-constrained environments
- Establishing baseline for gene→isoform prediction
- Probabilistic modeling with simpler architecture

**Hyperparameters**:
- `latent_dim`: Latent space dimension (default: 64)
- `beta`: KL divergence weight (default: 0.01)
- `gamma`: Isoform prediction loss weight (default: 1.0)
- `hidden_dims`: Hidden layer dimensions (default: [1024, 512])

### 3. Baseline FFN (Feedforward Network)

**Architecture**:  
Simple encoder-decoder feedforward network  

**Input**:  
Gene expression data  

**Output**:  
Isoform predictions  

**Key Features**:
- Minimalist deterministic architecture
- No latent space representation
- Fast training and inference
- Easy to interpret and debug
- Serves as performance floor

**Use Cases**:
- Establishing minimum performance baseline
- Quick prototyping and sanity checks
- Extremely resource-constrained scenarios
- When probabilistic modeling is not required

**Hyperparameters**:
- `hidden_dims`: Hidden layer dimensions (default: [512, 256])
- `learning_rate`: Optimization rate (default: 1e-3)
- `batch_size`: Training batch size (default: 256)

## Model Comparison

| Feature                | Multimodal VAE | Baseline VAE | Baseline FFN |
|------------------------|----------------|--------------|--------------|
| **Input Modalities**   | Gene + Isoform | Gene only    | Gene only    |
| **Probabilistic**      | ✓              | ✓            | ✗            |
| **Latent Space**       | Joint multimodal | Standard   | None         |
| **Cross-Modal Prediction** | ✓           | ✗            | ✗            |
| **Parameter Count**    | High           | Medium       | Low          |
| **Training Speed**     | Slow           | Medium       | Fast         |
| **Best Use Case**      | Joint analysis | Gene→isoform prediction | Quick baseline |

## Usage Recommendations

1. **Use Multimodal VAE when**:
   - You have both gene and isoform data
   - You need to model cross-modal relationships
   - Uncertainty modeling is important
   - Joint representation learning is required

2. **Use Baseline VAE when**:
   - Only gene data is available
   - You need probabilistic predictions
   - Computational resources are limited
   - Isolating gene→isoform mapping

3. **Use Baseline FFN when**:
   - You need a quick performance baseline
   - Resources are extremely constrained
   - Simplicity is prioritized
   - Probabilistic modeling isn't required


The baseline model (GeneToIsoformModel) is a simple deterministic feedforward neural network that directly maps gene expression to isoform expression using a straightforward encoder-decoder architecture. Unlike the multimodal VAE, it lacks probabilistic modeling, variational inference, joint latent space learning, or cross-modal reconstruction capabilities. While the multimodal VAE uses complex loss functions with KL divergence terms and can reconstruct both modalities bidirectionally, the baseline employs a basic MSE loss and only predicts isoforms from genes. The baseline serves as a performance floor - it shows how much isoform information can be extracted from genes alone without the multimodal VAE's sophisticated architecture that learns joint representations and models uncertainty through variational inference.

