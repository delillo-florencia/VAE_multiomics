import pandas as pd
import numpy as np
import gc

def refactor_expression(df):
    df_expression = df.transpose()
    df_expression = df_expression.reset_index().rename(columns={'index': 'sample_id'})
    return df_expression

def filter_tpm_thres(df, thres, min_samples):
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    passing_samples = (df_numeric > thres).sum(axis=0)
    filtered_feats = df.loc[:, passing_samples >= min_samples]
    del df_numeric, passing_samples  # Free up memory
    gc.collect()
    return filtered_feats

def log_transform(df_filtered_tpm):
    filt_log2 = np.log2(df_filtered_tpm + 1)
    del df_filtered_tpm # Free up memory
    gc.collect()
    return filt_log2


def gene_from_tx(anot,txs):
    genes=anot[anot["transcript_id"].isin(txs)]["gene_id"].to_list()
    return genes
    