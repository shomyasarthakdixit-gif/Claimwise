import os
from sklearn.ensemble import RandomForestRegressor
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M3: Random Forest Demonstration")
    print("Concepts: Bagging + Random Feature Selection (max_features).\n")
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(2000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # max_features='sqrt' introduces randomness at each split
    rf = RandomForestRegressor(n_estimators=50, max_features='sqrt', random_state=42)
    rf.fit(X_train_processed, y_train)
    
    print(f"Random Forest Test R2: {rf.score(X_test_processed, y_test):.4f}")

if __name__ == "__main__":
    main()
