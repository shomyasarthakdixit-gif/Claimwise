import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: PCA Demonstration")
    print("Concepts: Principal Component Analysis, standardization considerations.")
    sample_size = 5000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    pca = PCA(n_components=10, random_state=random_seed)
    X_pca = pca.fit_transform(X_processed)
    
    print(f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_}")
    
    os.makedirs(Config.CLUSTERING_DIR, exist_ok=True)
    pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(10)]).to_csv(os.path.join(Config.CLUSTERING_DIR, 'pca_results.csv'), index=False)
    
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
    plt.title('PCA Projection (First 2 Components)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.text(0.05, 0.95, "Note: PCA is for visualization only. Distances do not imply business relationships.", transform=plt.gca().transAxes, fontsize=8)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'pca_clusters.png'))
    plt.close()

if __name__ == "__main__":
    main()
