import os
from sklearn.linear_model import ElasticNet
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M2: Elastic Net Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(1000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train_processed, y_train)
    print(f"Elastic Net R2 Score: {model.score(X_test_processed, y_test):.4f}")

if __name__ == "__main__":
    main()
