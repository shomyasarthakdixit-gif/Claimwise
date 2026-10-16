import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor, get_train_test_splits
from config import Config

def main():
    print("M4: ClaimWise Clustering as Features Baseline")
    print("Concept: Cluster labels as features / relationship between clustering and supervised learning.")
    
    # We will sample 10000 records, train a clustering model, and then append the clusters as a feature for regression.
    sample_size = 10000
    random_seed = 42
    print(f"Loading reproducible sample of {sample_size} records using random_seed={random_seed}...")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    # 1. Clustering
    X_cluster = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X_cluster)
    X_processed = preprocessor.fit_transform(X_cluster)
    
    mbk = MiniBatchKMeans(n_clusters=5, random_state=random_seed, n_init=3)
    clusters = mbk.fit_predict(X_processed)
    df['cluster_label'] = clusters
    
    os.makedirs(Config.CLUSTERING_DIR, exist_ok=True)
    pd.DataFrame({"id": df['id'], "cluster_label": clusters}).to_csv(os.path.join(Config.CLUSTERING_DIR, 'cluster_results.csv'), index=False)
    print("Clusters saved to cluster_results.csv.")
    
    # 2. Supervised Learning using clusters as features
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    # We need a new preprocessor that includes 'cluster_label' as a categorical feature
    prep_regression = create_claimwise_preprocessor(X_train)
    pipeline = Pipeline(steps=[
        ('preprocessor', prep_regression),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_seed, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Random Forest (with clusters) Test R2: {test_score:.4f}")
    
    # NOTE on comparing to baseline:
    # "When comparing R2 values, clearly state the sample size and random seed for each result. Do not directly rank results obtained from different sample sizes/test splits."
    print("\n--- Metric Report ---")
    print("ClaimWise M3 Random Forest Baseline — 10,000-row reproducible sample (random_seed=42): 0.4868")
    print(f"M4 Random Forest (w/ Clusters) — 10,000-row reproducible sample (random_seed={random_seed}): {test_score:.4f}")
    print("Note: The clustering step did not claim real-world insurance categories; it mathematically segmented the feature space.")

if __name__ == "__main__":
    main()
