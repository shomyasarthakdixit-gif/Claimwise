import os
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M3: LightGBM Demonstration")
    print("Concepts: Leaf-wise growth, histogram-based splitting.\n")
    
    try:
        from lightgbm import LGBMRegressor
        df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(2000, random_state=42)
        X_train, X_test, y_train, y_test = get_train_test_splits(df)
        
        preprocessor = create_claimwise_preprocessor(df)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        lgb = LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42, verbose=-1)
        lgb.fit(X_train_processed, y_train)
        print(f"LightGBM Test R2: {lgb.score(X_test_processed, y_test):.4f}")
    except ImportError:
        print("LightGBM is not installed. Gracefully skipping LightGBM demonstration.")

if __name__ == "__main__":
    main()
