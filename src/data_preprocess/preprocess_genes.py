import pandas as pd
import gc
from data_preprocess.utils import *


def preprocess_genes(path_to_genes, path_to_tx, path_to_anot, needs_split=True):
    """
    Preprocess gene expression data from a TSV file and select relevant genes based on isoform list.

    Steps performed:
    1. Reads gene expression TSV file.
    2. Optionally normalizes column names to remove suffixes.
    3. Reads relevant isoforms list and annotation file.
    4. Maps isoforms to corresponding genes using `gene_from_tx`.
    5. Filters gene dataframe to retain only relevant genes + 'sample_id'.
    6. Converts gene expression values to numeric and sets 'sample_id' as index.
    7. Applies log transformation to filtered gene data using `log_transform`.
    8. Saves the processed file as 'filt_log_genes.tsv'.

    Parameters:
    ----------
    path_to_genes : str
        Path to raw gene expression TSV file (samples in rows, genes in columns).
    path_to_tx : str
        Path to text file containing relevant isoform IDs (one per line).
    path_to_anot : str
        Path to gene-transcript annotation CSV file mapping isoforms to genes.
    needs_split : bool, optional
        If True, column names will be split at '.' to remove suffixes. Default is True.

    Outputs:
    -------
    - 'filt_log_genes.tsv': filtered and log-transformed gene expression data
    """
    
    print("Reading genes", flush=True)
    df = pd.read_csv(path_to_genes, sep="\t")
    
    # Clean column names if needed
    if needs_split:
        df.columns = [col.split('.')[0] for col in df.columns]
    
    print(df.head(), flush=True)
    print("Files loaded", flush=True)

    # Load isoform list
    with open(path_to_tx, 'r') as file:
        txs = [line.strip() for line in file.readlines()]
    print("Isoforms loaded", flush=True)

    # Load annotation file
    anot_file = pd.read_csv(path_to_anot)
    
    # Map isoforms to genes
    rel_genes = gene_from_tx(anot_file, txs)
    print("Relevant genes mapped", flush=True)
    print(rel_genes[:10], flush=True)

    # Prepare list of relevant genes with sample_id
    rel_genes_sample = list(set(["sample_id"] + rel_genes))
    print("Number of relevant genes:", len(rel_genes_sample), flush=True)

    # Subset gene dataframe
    filt_genes = df[rel_genes_sample]

    # Convert to numeric and set index
    filt_genes_numeric = filt_genes.set_index("sample_id")
    filt_genes = filt_genes_numeric.apply(pd.to_numeric, errors='coerce')
    del filt_genes_numeric
    gc.collect()

    print(filt_genes.head(), flush=True)
    print(filt_genes.shape, flush=True)

    # Log transform
    print("Log transform in progress...", flush=True)
    transformed_genes = log_transform(filt_genes)

    # Save output
    print("Saving file...", flush=True)
    transformed_genes.to_csv("filt_log_genes.tsv", sep="\t")
