import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def analyze_gene_isoform_correlations(feature_corr_file, gene_isoform_file, output_dir="gene_analysis", corr_metric="pearson"):
    """
    Analyze per-feature correlations in the context of gene-isoform relationships.
    
    Args:
        feature_corr_file: Path to the per-feature correlation CSV file
        gene_isoform_file: Path to the gene-isoform relationship CSV file
        output_dir: Directory to save analysis results
        corr_metric: Which correlation column to use ("pearson" or "cosine")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print("Loading feature correlations...",flush=True)
    feature_corr = pd.read_csv(feature_corr_file)
    
    print("Loading gene-isoform relationships...",flush=True)
    gene_isoform = pd.read_csv(gene_isoform_file)
     
    if corr_metric not in feature_corr.columns:
        raise ValueError(f"{corr_metric} not found in feature_corr file. Available: {feature_corr.columns.tolist()}")
    
    feature_corr = feature_corr.rename(columns={corr_metric: "correlation"})
    
    print("Merging with gene information...",flush=True)
    merged = pd.merge(feature_corr, gene_isoform, on="transcript_id", how="left")
    
    # Count isoforms per gene
    print("Counting isoforms per gene...",flush=True)
    isoforms_per_gene = merged.groupby("gene_id")["transcript_id"].count().reset_index()
    df_to_save=isoforms_per_gene.copy()
    df_to_save=df_to_save.sort_values(by="transcript_id",ascending=False)
    df_to_save.to_csv("isoforms_per_gene.csv")
    isoforms_per_gene.columns = ["gene_id", "isoform_count"]
    
    # Add isoform count to merged df
    merged = pd.merge(merged, isoforms_per_gene, on="gene_id", how="left")
    
    # Filter for multi-isoform genes
    multi_isoform_genes = merged[merged["isoform_count"] > 1]
    print(f"Found {len(multi_isoform_genes)} isoforms from {multi_isoform_genes['gene_id'].nunique()} multi-isoform genes")
    #multi_isoform_genes = merged.copy()   # 

    # Calculate statistics by isoform count
    stats_by_count = multi_isoform_genes.groupby("isoform_count")["correlation"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    # Overall stats
    overall_stats = multi_isoform_genes["correlation"].agg([
        "count", "mean", "median", "std", "min", "max"
    ])
    overall_stats["isoform_count"] = "all_multi_isoform"
    
    all_stats = pd.concat([stats_by_count, overall_stats.to_frame().T], ignore_index=True)
    
    print("Saving results...",flush=True)
    merged.to_csv(f"{output_dir}/all_features_with_gene_info.csv", index=False)
    all_stats.to_csv(f"{output_dir}/correlation_stats_by_isoform_count.csv", index=False)
    
    # Visualizations
    print("Creating visualizations...")
    
    # 1. Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=multi_isoform_genes, x="correlation", bins=50)
    plt.title(f"Distribution of {corr_metric.capitalize()} Correlations for Multi-Isoform Genes")
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_distribution.png")
    plt.close()
    
    # 2. Boxplot
    plt.figure(figsize=(12, 6))
    max_count_for_viz = min(20, multi_isoform_genes["isoform_count"].max())
    filtered_data = multi_isoform_genes[multi_isoform_genes["isoform_count"] <= max_count_for_viz]
    sns.boxplot(data=filtered_data, x="isoform_count", y="correlation")
    plt.title(f"{corr_metric.capitalize()} Correlation by Number of Isoforms per Gene")
    plt.xlabel("Number of Isoforms")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_by_isoform_count.png")
    plt.close()
    
    # 3. Scatter plot avg correlation vs isoform count
    plt.figure(figsize=(10, 6))
    plt.scatter(stats_by_count["isoform_count"], stats_by_count["mean"], alpha=0.7)
    plt.title(f"Average {corr_metric.capitalize()} Correlation vs Number of Isoforms per Gene")
    plt.xlabel("Number of Isoforms")
    plt.ylabel("Average Correlation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/avg_correlation_vs_isoform_count.png")
    plt.close()
    
    # 4. Top and bottom genes
    gene_stats = multi_isoform_genes.groupby(["gene_id", "gene_name", "isoform_count"])["correlation"].agg([
        "mean", "count"
    ]).reset_index()
    gene_stats.columns = ["gene_id", "gene_name", "isoform_count", "avg_correlation", "isoform_count_verified"]
    
    top_genes = gene_stats.nlargest(10, "avg_correlation")
    top_genes.to_csv(f"{output_dir}/top_10_genes_by_correlation.csv", index=False)
    
    bottom_genes = gene_stats.nsmallest(10, "avg_correlation")
    bottom_genes.to_csv(f"{output_dir}/bottom_10_genes_by_correlation.csv", index=False)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total multi-isoform genes: {gene_stats['gene_id'].nunique()}")
    print(f"Average correlation across all multi-isoform genes: {overall_stats['mean']:.4f}")
    print(f"Median correlation across all multi-isoform genes: {overall_stats['median']:.4f}")
    
    for count in [1,2, 3,4, 5,6,7,8,9, 10,11,12,13,14,15,16,17,18,19,20,30,50,70,90,100]:
        if count in stats_by_count["isoform_count"].values:
            stats = stats_by_count[stats_by_count["isoform_count"] == count].iloc[0]
            print(f"Genes with {count} isoforms: {stats['count']} genes, avg correlation: {stats['mean']:.4f}")


if __name__ == "__main__":
    
    feature_corr_file = "path/to/correlation_file"
    gene_isoform_file = "path/to/gene_transcript_relation.csv"
    
    # pearson or cosine
    analyze_gene_isoform_correlations(feature_corr_file, gene_isoform_file, corr_metric="pearson")
