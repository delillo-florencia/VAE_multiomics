# VAE_multiomics 

This project was developed for the **DTU Research Immersion 3-week course**.   

The main objective was to build a flexible and efficient framework to train multimodal and baseline autoencoders, with the ultimate goal to predict isoform expression levels from gene expression data.
 

This repository provides everything you need to **preprocess data, create splits, configure experiments, and train machine learning models** for transcriptomics and multimodal analysis.  


Follow the steps below to prepare your data, select a model, configure hyperparameters, and launch your training!

# How to train a model

##  0. Set up Miniconda 
### Create a directory for Miniconda

```bash
mkdir -p ~/miniconda3
```
### Download the latest Miniconda installer
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
```
### Run the installer in silent mode
```bash
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
### Remove the installer to save space
```bash
rm ~/miniconda3/miniconda.sh
```
### Activate Miniconda
```bash
source ~/miniconda3/bin/activate
```
## 2. Configure Conda 

To ensure Conda is correctly initialized in future terminal sessions:

```bash
# Initialize Conda in your shell
conda init

# Reload your shell configuration
source ~/.bashrc
```

---

##  Create a Python Environment

To create a custom **Python 3.10** environment inside your home directory:

```bash
# Replace <your-username> with your DTU username
conda create --prefix=/home/people/<your-username>/your_environment python=3.10
```

**Activate the environment:**

```bash
conda activate /home/people/<your-username>/your_environment
```

**Verify Python installation:**

```bash
python --version
```

---

##  4. Install Python Package Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Then, go to src and run:

```bash
pip install e .
```

## 1. Prepare Your Data

Ensure your dataset follows this format:  
- **Rows = samples**  
- **Columns = features**

Preprocessing steps used in our experiments:
- Log-transform expression data.  
- Filter out transcripts with very low expression.  
- Retain only the relevant associated genes.  

To do so, you may use some of the code in the preprocess module
To speed up loading and reduce memory usage, we recommend **hashing the gene and isoform data** using:

```bash
 data_preprocess/hash_files.py
```
This will enable you to load the data super fast and without consuming too much memory.

## 2. Create splits

Now that your data is ready, it is time to create the train, test and validations splits. Be mindful of your splitting strategy! Incorrect splits may introduce bias, overfitting and may inflate metrics. To create splits, you can use:
```bash
 utils/generate_splits.py
```

Available strategies include:
* by_study
* clustering
* random

## 3. Select a model

As you may see in the models module, there are several options for training. A basic autoencoder, a baseline Variational autoencoder and 2 different multimodal autoencoders. Please refer to the module README and check the codes for details about their differences..

## 4. Prepare your config file
Use  ```utils/config.py``` as a template for your config file.
You will need to specify:

* Working directory
* Model type
* Model hyperparameters
* & More

⚠️ Note: Some config arguments may not apply to every model. Double-check the model requirements.

## 5. Train the model: 
Now everything is ready. You can run:

```bash
 main.py
```

This will launch training with the selected model and hyperparameters.

### Submit job in SLURM 

If working on a SLURM cluster, you can submit and manage jobs with:
```bash
sbatch run_slurm.sh        # Submit a job
squeue -u $USER            # Check your jobs
scancel <job-id>           # Cancel a job
scontrol show job <job-id> # Show job details
chmod +x run_slurm.sh      # Make script executable
```
