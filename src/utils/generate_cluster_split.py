import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import random

def cluster_split_from_distance(dist_matrix, sample_ids, n_splits=(0.7,0.15,0.15), n_clusters=None, random_seed=0):
    """
    dist_matrix: NxN symmetric distance matrix (numpy)
    sample_ids: list of sample IDs corresponding to matrix rows
    n_splits: tuple fractions for train/val/test (must sum to 1)
    n_clusters: if None, choose reasonable default (sqrt(N) or you can pass)
    returns: dict with keys 'train','val','test' with lists of sample IDs
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    N = dist_matrix.shape[0]
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

    clust_to_members = {}
    for i, c in enumerate(clusters):
        clust_to_members.setdefault(c, []).append(sample_ids[i])

    cluster_ids = list(clust_to_members.keys())
    random.shuffle(cluster_ids)

    target_counts = [int(frac * N) for frac in n_splits]
    diff = N - sum(target_counts)
    target_counts[0] += diff

    splits = {'train': [], 'val': [], 'test': []}
    targets = dict(zip(['train','val','test'], target_counts))
    
    for cid in cluster_ids:
        members = clust_to_members[cid]
        best_split = max(targets.keys(), key=lambda k: targets[k])
        splits[best_split].extend(members)
        targets[best_split] -= len(members)

    return splits

# ----------------------Run example----------------

# print("Loading data...",flush=True)
expr_df = pd.read_csv("VAE_multiomics/data/filtered/filt_log_genes.tsv", sep="\t")
sample_ids = expr_df['sample_id'].tolist()
genes = [col for col in expr_df.columns if col != 'sample_id']
X = expr_df[genes].values

print("Generating distance matrix...",flush=True)
dist_matrix = pairwise_distances(X, metric='euclidean')  # Choose appropriate metric
np.save(f"distance_matrix.npy", dist_matrix)

print("Generating splits...",flush=True)

splits = cluster_split_from_distance(dist_matrix, sample_ids)

print("Writing splits ...",flush=True)

for split_name, samples in splits.items():
    with open(f'{split_name}_samples.txt', 'w') as f:
        f.write('\n'.join(samples))