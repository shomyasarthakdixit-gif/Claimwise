import os
import matplotlib.pyplot as plt
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: UMAP Demonstration")
    try:
        import umap
        sample_size = 1000
        random_seed = 42
        df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
        
        X = df.drop(columns=['id', 'loss'])
        preprocessor = create_claimwise_preprocessor(X)
        X_processed = preprocessor.fit_transform(X)
        
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_seed)
        X_umap = reducer.fit_transform(X_processed)
        
        os.makedirs(Config.PLOT_DIR, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.5, c='teal')
        plt.title('UMAP Visualization')
        plt.text(0.05, 0.95, "Note: UMAP is for visualization only. Visual distances do not imply business relationships.", transform=plt.gca().transAxes, fontsize=8)
        plt.savefig(os.path.join(Config.PLOT_DIR, 'umap_visualization.png'))
        plt.close()
        print("UMAP visualization generated.")
    except ImportError:
        print("UMAP is not installed. Gracefully skipping UMAP demonstration.")

if __name__ == "__main__":
    main()
