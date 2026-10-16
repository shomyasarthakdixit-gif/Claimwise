import os
from sklearn.cluster import KMeans
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: KMeans++ Demonstration")
    sample_size = 5000
    random_seed = 42
    print(f"Loading reproducible sample of {sample_size} records using random_seed={random_seed}...")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=random_seed)
    kmeans.fit(X_processed)
    print(f"KMeans++ completed. Inertia: {kmeans.inertia_:.4f}")

if __name__ == "__main__":
    main()
