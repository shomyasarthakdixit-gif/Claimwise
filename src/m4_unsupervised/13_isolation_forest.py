import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Isolation Forest Demonstration")
    sample_size = 5000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    clf = IsolationForest(contamination=0.01, random_state=random_seed)
    preds = clf.fit_predict(X_processed)
    
    outliers = list(preds).count(-1)
    print(f"Isolation Forest identified {outliers} anomalies out of {sample_size} records (1% contamination).")
    
    os.makedirs(Config.CLUSTERING_DIR, exist_ok=True)
    pd.DataFrame({"id": df['id'], "anomaly_flag": preds}).to_csv(os.path.join(Config.CLUSTERING_DIR, 'anomaly_results.csv'), index=False)

if __name__ == "__main__":
    main()
