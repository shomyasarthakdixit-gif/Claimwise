import os
from sklearn.cluster import AgglomerativeClustering
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Hierarchical/Agglomerative Clustering Demonstration")
    sample_size = 1000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = agg_cluster.fit_predict(X_processed)
    print("Agglomerative clustering completed. Concepts: bottom-up merging.")

if __name__ == "__main__":
    main()
