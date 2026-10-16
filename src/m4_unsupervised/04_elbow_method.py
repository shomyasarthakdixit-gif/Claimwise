import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Elbow Method Demonstration")
    sample_size = 5000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=random_seed, n_init=3)
        mbk.fit(X_processed)
        inertias.append(mbk.inertia_)
    
    # Save CSV
    os.makedirs(Config.CLUSTERING_DIR, exist_ok=True)
    pd.DataFrame({'k': list(k_range), 'inertia': inertias}).to_csv(os.path.join(Config.CLUSTERING_DIR, 'elbow_results.csv'), index=False)
    
    # Save Plot
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'elbow_curve.png'))
    plt.close()
    print("Elbow results and curve generated.")

if __name__ == "__main__":
    main()
