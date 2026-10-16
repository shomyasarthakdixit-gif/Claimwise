import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: DBSCAN Demonstration")
    print("Concepts: eps, minPts, outliers, non-convex clusters.")
    sample_size = 2000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    dbscan = DBSCAN(eps=5.0, min_samples=10)
    labels = dbscan.fit_predict(X_processed)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise/outliers points: {n_outliers}")
    
    os.makedirs(Config.CLUSTERING_DIR, exist_ok=True)
    pd.DataFrame({"id": df['id'], "cluster": labels}).to_csv(os.path.join(Config.CLUSTERING_DIR, 'dbscan_results.csv'), index=False)
    
    # PCA to 2D for plotting clusters
    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X_processed)
    
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('DBSCAN Clusters (Projected to 2D via PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.text(0.05, 0.95, "Note: PCA is for visualization only. Distances do not imply business relationships.", transform=plt.gca().transAxes, fontsize=8)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'dbscan_clusters.png'))
    plt.close()

if __name__ == "__main__":
    main()
