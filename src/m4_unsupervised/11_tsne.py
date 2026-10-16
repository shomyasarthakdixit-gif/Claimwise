import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: t-SNE Demonstration")
    sample_size = 1000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_seed)
    X_tsne = tsne.fit_transform(X_processed)
    
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5, c='purple')
    plt.title('t-SNE Visualization')
    plt.text(0.05, 0.95, "Note: t-SNE is for visualization only. Visual distances do not represent actual distances or business relationships.", transform=plt.gca().transAxes, fontsize=8)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'tsne_visualization.png'))
    plt.close()
    print("t-SNE visualization generated.")

if __name__ == "__main__":
    main()
