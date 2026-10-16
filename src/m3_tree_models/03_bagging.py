import os
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M3: Bagging Demonstration")
    print("Concepts: Bootstrap aggregating, Out-Of-Bag (OOB) evaluation.\n")
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(2000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    bagging = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=50,
        oob_score=True,
        bootstrap=True,
        random_state=42
    )
    
    bagging.fit(X_train_processed, y_train)
    print(f"OOB Score (no validation set needed): {bagging.oob_score_:.4f}")
    print(f"Test Set R2: {bagging.score(X_test_processed, y_test):.4f}")

if __name__ == "__main__":
    main()
