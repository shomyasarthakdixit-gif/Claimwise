import os
from sklearn.cluster import AgglomerativeClustering
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Linkages Demonstration")
    sample_size = 1000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    linkages = ['ward', 'complete', 'average', 'single']
    for link in linkages:
        agg = AgglomerativeClustering(n_clusters=3, linkage=link)
        agg.fit(X_processed)
        print(f"Linkage: {link} - generated {len(set(agg.labels_))} clusters.")

if __name__ == "__main__":
    main()
