import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Silhouette Score Demonstration")
    sample_size = 2000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    sil_scores = []
    k_range = range(2, 6)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=3)
        labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        sil_scores.append(score)
    
    os.makedirs(Config.CLUSTERING_DIR, exist_ok=True)
    pd.DataFrame({'k': list(k_range), 'silhouette_score': sil_scores}).to_csv(os.path.join(Config.CLUSTERING_DIR, 'silhouette_results.csv'), index=False)
    
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sil_scores, marker='s', color='orange')
    plt.title('Silhouette Scores For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'silhouette_scores.png'))
    plt.close()
    print("Silhouette results and curve generated.")

if __name__ == "__main__":
    main()
