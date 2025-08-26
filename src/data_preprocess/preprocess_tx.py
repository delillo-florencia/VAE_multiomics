import pandas as pd
import numpy as np
import gc
from data_preprocess.utils import *


def preprocess_tx(path_to_isof,needs_transpose=False):
    """
    Preprocess isoform expression data from a TSV file.

    Steps performed:
    1. Reads the isoform TSV file.
    2. Optionally transposes the dataframe if samples are in columns.
    3. Saves sample names to 'samples_names_tx.txt'.
    4. Filters isoforms based on TPM thresholds (default 1-100) using `filter_tpm_thres`.
    5. Saves relevant transcripts to 'rel_tx.txt'.
    6. Subsets the dataframe to filtered transcripts and sets 'sample_id' as the index.
    7. Applies log transformation to filtered data using `log_transform`.
    8. Saves the final processed file as 'filt_log_isoforms.tsv'.

    Parameters:
    ----------
    path_to_isof : str
        Path to the isoform TSV file. The file should have samples in rows and isoforms in columns,
        unless `needs_transpose` is True.
    needs_transpose : bool, optional
        If True, the function will transpose the input dataframe to have samples in rows and isoforms in columns.
        Default is False.

    Outputs:
    -------
    - 'samples_names_tx.txt': text file with sample names
    - 'rel_tx.txt': text file with relevant transcript IDs
    - 'filt_log_isoforms.tsv': filtered and log-transformed isoform expression data
    """
    
    print("Reading tx", flush=True)
    df = pd.read_csv(path_to_isof, sep="\t")
    
    # Save sample names
    samples = df.columns
    with open("samples_names_tx.txt", "w") as file:
        for sample in samples:
            file.write(sample + "\n")

    if needs_transpose:
        # Refactor expression
        df_isof = refactor_expression(df)
    else:
        df_isof=df.copy()
        
    del df 
        
    print(df_isof.head(), flush=True)
    print(df_isof.shape, flush=True)

    # Filter data
    print("Filtering", flush=True)
    filter_tpm_tx = filter_tpm_thres(df_isof, 1, 100)
    print("Shape after filtering", flush=True)
    print(filter_tpm_tx.shape)

    # Save relevant transcripts
    txs = filter_tpm_tx.columns.to_list()
    with open("rel_tx.txt", "w") as file:
        for i in txs:
            file.write(i + "\n")

    # Subset and transform data
    cols_subj = ["sample_id"] + txs
    df_isof_filtered = df_isof[cols_subj].set_index("sample_id")
    del df_isof  # Free up memory
    gc.collect()

    print("Log transform in progress...", flush=True)
    transformed_isof = log_transform(df_isof_filtered)

    # Save output
    print("Saving file...", flush=True)
    transformed_isof.to_csv("filt_log_isoforms.tsv", sep="\t")