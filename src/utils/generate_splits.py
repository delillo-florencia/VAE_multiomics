import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import os

def create_dataset_splits(gene_expression_file, study_relation_file, 
                          split_method='random', train_ratio=0.7, val_ratio=0.15, 
                          test_ratio=0.15, n_clusters=10, random_state=None, 
                          distance_metric='euclidean', output_prefix='split',
                          plot_clusters=True, projection_method='pca'):
    """
    Creates train, validation, and test splits based on specified method.
    
    Args:
        gene_expression_file (str): Path to gene expression TSV (sample_id, genes...)
        study_relation_file (str): Path to study relation TSV (sample_id, study_id)
        split_method (str): 'random', 'by_study', or 'clustering'
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        n_clusters (int): Number of clusters for clustering method
        random_state (int): Seed for reproducibility
        distance_metric (str): Distance metric for clustering ('euclidean', 'correlation', etc.)
    
    Returns:
        dict: Sample IDs for each set {'train': [], 'val': [], 'test': []}
    """
    # Load and merge data
    print("Loading data....",flush=True)
    expr_df = pd.read_csv(gene_expression_file, sep='\t')
    print(expr_df.head)
    study_df = pd.read_csv(study_relation_file)
    print(study_df.head)
    print("Merging...",flush=True)
    merged_df = pd.merge(expr_df, study_df, on='sample_id', how='left')
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=0.01):
        raise ValueError("Ratios must sum to approximately 1.0")
    
    if split_method == 'random':
        # Random shuffle and split
        print("Shuffling data..",flush=True)
        shuffled = merged_df.sample(frac=1, random_state=random_state)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        print("Building sets...",flush=True)
        train = shuffled.iloc[:n_train]
        val = shuffled.iloc[n_train:n_train+n_val]
        test = shuffled.iloc[n_train+n_val:]
        
    elif split_method == 'by_study':
        # Ensure no missing study IDs
        if merged_df['study'].isnull().any():
            raise ValueError("Study IDs missing for some samples")
        
        # Group by study and assign entire studies to splits
        studies = merged_df['study'].unique()
        np.random.seed(random_state)
        np.random.shuffle(studies)
        
        n_studies = len(studies)
        n_train = int(n_studies * train_ratio)
        n_val = int(n_studies * val_ratio)
        
        train_studies = studies[:n_train]
        val_studies = studies[n_train:n_train+n_val]
        test_studies = studies[n_train+n_val:]
        
        train = merged_df[merged_df['study'].isin(train_studies)]
        val = merged_df[merged_df['study'].isin(val_studies)]
        test = merged_df[merged_df['study'].isin(test_studies)]
        
    elif split_method == 'clustering':
        # Prepare expression matrix
        genes = [col for col in expr_df.columns if col != 'sample_id']
        X = merged_df[genes].values
        
        # Standardize features
        print("Scaling..",flush=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster samples
        print("Clustering..",flush=True)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        merged_df['cluster'] = clusters
        
        # Calculate cluster distances from centroid
        centroids = kmeans.cluster_centers_
        overall_centroid = np.mean(centroids, axis=0)
        print("Distances...",flush=True)
        cluster_distances = np.linalg.norm(centroids - overall_centroid, axis=1)
        
        # Sort clusters by distance (descending)
        sorted_clusters = np.argsort(cluster_distances)[::-1]
        cluster_sizes = merged_df['cluster'].value_counts().sort_index().values
        
        # Assign clusters to sets based on distance
        test_target = test_ratio * len(merged_df)
        val_target = val_ratio * len(merged_df)
        
        test_clusters, val_clusters, train_clusters = [], [], []
        test_count, val_count = 0, 0
        print("In for..",flush=True)
        for cluster in sorted_clusters:
            size = cluster_sizes[cluster]
            
            if test_count < test_target:
                test_clusters.append(cluster)
                test_count += size
            elif val_count < val_target:
                val_clusters.append(cluster)
                val_count += size
            else:
                train_clusters.append(cluster)
        
        # Create splits
        train = merged_df[merged_df['cluster'].isin(train_clusters)]
        val = merged_df[merged_df['cluster'].isin(val_clusters)]
        test = merged_df[merged_df['cluster'].isin(test_clusters)]

        
    elif split_method == 'clustering_dist':
        # Extract gene expression data
        genes = [col for col in expr_df.columns if col != 'sample_id']
        X = merged_df[genes].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute distance matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        dist_matrix = pairwise_distances(X_scaled, metric=distance_metric)
        #dist_matrix=np.load("distance_matrix.npy")
        #np.save(f"distance_matrix.npy", dist_matrix)
        # Perform hierarchical clustering
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='average',        # or 'complete' / 'single'
            metric='precomputed'     # you're passing a distance matrix
        )
        clusters = agg.fit_predict(dist_matrix)
        merged_df['cluster'] = clusters
        
        # Compute cluster centroids
        centroids = []
        for i in range(n_clusters):
            cluster_data = X_scaled[clusters == i]
            if len(cluster_data) > 0:
                centroids.append(cluster_data.mean(axis=0))
            else:
                centroids.append(np.zeros(X_scaled.shape[1]))
        centroids = np.array(centroids)
        
        # Compute overall centroid
        overall_centroid = X_scaled.mean(axis=0)
        
        # Calculate cluster distances
        cluster_distances = np.linalg.norm(centroids - overall_centroid, axis=1)
        
        # Sort clusters by distance
        sorted_clusters = np.argsort(cluster_distances)[::-1]
        cluster_sizes = [np.sum(clusters == i) for i in range(n_clusters)]
        
        # Calculate targets
        total_samples = len(merged_df)
        test_target = test_ratio * total_samples
        val_target = val_ratio * total_samples
        
    # … after computing `cluster_sizes`, `sorted_clusters`, total_samples, test_target, val_target …

        test_clusters, val_clusters, train_clusters = [], [], []
        test_count, val_count = 0, 0

        for cluster in sorted_clusters:
            size = cluster_sizes[cluster]
            
            # Only add to test if it won't overshoot
            if test_count + size <= test_target:
                test_clusters.append(cluster)
                test_count += size
            
            # Otherwise try validation
            elif val_count + size <= val_target:
                val_clusters.append(cluster)
                val_count += size
            
            # Otherwise go to training
            else:
                train_clusters.append(cluster)

        # If, for some reason, val or train is still empty (e.g. very skewed cluster sizes),
        # force at least one cluster into each split:
        if not val_clusters and len(sorted_clusters) > 1:
            # move smallest from train to val
            smallest = min(train_clusters, key=lambda c: cluster_sizes[c])
            train_clusters.remove(smallest)
            val_clusters.append(smallest)

        if not train_clusters and len(sorted_clusters) > 0:
            # move smallest from val (or test) to train
            source = val_clusters or test_clusters
            smallest = min(source, key=lambda c: cluster_sizes[c])
            source.remove(smallest)
            train_clusters.append(smallest)

        # Now subset your DataFrame:
        train = merged_df[merged_df['cluster'].isin(train_clusters)]
        val   = merged_df[merged_df['cluster'].isin(val_clusters)]
        test  = merged_df[merged_df['cluster'].isin(test_clusters)]



        # Now generate visualization if requested
        if plot_clusters:
            print("plotting_clusters", flush=True)
            try:
                # Create 2D projection
                if projection_method == 'umap':
                    reducer = umap.UMAP(random_state=random_state)
                    proj = reducer.fit_transform(X_scaled)
                else:  # Default to PCA
                    pca = PCA(n_components=2, random_state=random_state)
                    proj = pca.fit_transform(X_scaled)
                
                # Create figure with 2 subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Color by cluster
                scatter1 = ax1.scatter(proj[:, 0], proj[:, 1], c=clusters, 
                                    cmap='tab20', alpha=0.6, s=10)
                ax1.set_title(f'Cluster Assignment ({n_clusters} clusters)')
                ax1.set_xlabel(f'{projection_method.upper()} 1')
                ax1.set_ylabel(f'{projection_method.upper()} 2')
                fig.colorbar(scatter1, ax=ax1, label='Cluster ID')
                
                # Plot 2: Color by dataset split
                split_colors = {'train': 'green', 'val': 'orange', 'test': 'red'}
                for split, color in split_colors.items():
                    idx = merged_df[merged_df['cluster'].isin(eval(f"{split}_clusters"))].index
                    ax2.scatter(proj[idx, 0], proj[idx, 1], 
                            color=color, label=split, alpha=0.6, s=10)
                
                ax2.set_title('Dataset Splits')
                ax2.set_xlabel(f'{projection_method.upper()} 1')
                ax2.set_ylabel(f'{projection_method.upper()} 2')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(f"{output_prefix}_cluster_plot.png", dpi=300)
                plt.close()
                
                print(f"Cluster visualization saved to {output_prefix}_cluster_plot.png")
                
            except Exception as e:
                warnings.warn(f"Cluster visualization failed: {str(e)}")
                

            else:
                raise ValueError("Invalid split_method. Choose 'random', 'by_study', 'clustering' or clustering_dist")
            
    # Prepare results
    print("Saving results",flush=True)
    results = {
        'train': train['sample_id'].tolist(),
        'val': val['sample_id'].tolist(),
        'test': test['sample_id'].tolist()
    }
    
    # Save sample IDs to text files
    for split_name in ['train', 'val', 'test']:
        with open(f"{split_method}_{split_name}_samples.txt", "w") as f:
            f.write("\n".join(results[split_name]))
    
    return results
    


# Generate splits with cluster visualization
splits = create_dataset_splits(
    gene_expression_file="/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/data/filtered/filt_log_genes.tsv",
    study_relation_file="/home/projects2/kvs_students/2025/fl_mVAE/VAE_multiomics/data/filtered/study_sample_id.csv",
    split_method='clustering_dist',
    n_clusters=3,
    distance_metric='euclidean',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    plot_clusters=True,          # Enable visualization
    projection_method='umap'     # Use UMAP for projection
)