import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Dendrogram Demonstration")
    sample_size = 100
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    Z = linkage(X_processed, 'ward')
    
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.savefig(os.path.join(Config.PLOT_DIR, 'dendrogram.png'))
    plt.close()
    print("Dendrogram generated.")

if __name__ == "__main__":
    main()
