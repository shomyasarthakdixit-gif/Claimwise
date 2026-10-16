import os
from sklearn.svm import OneClassSVM
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: One-Class SVM Demonstration")
    sample_size = 1000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    preds = clf.fit_predict(X_processed)
    outliers = list(preds).count(-1)
    print(f"One-Class SVM identified {outliers} anomalies out of {sample_size} records.")

if __name__ == "__main__":
    main()
