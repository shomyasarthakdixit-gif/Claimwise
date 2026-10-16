import os
import numpy as np
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M2: Normal Equation Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(1000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Adding bias term
    X_b = np.c_[np.ones((X_train_processed.shape[0], 1)), X_train_processed]
    
    # Using Moore-Penrose pseudo-inverse (SVD) for numerically safe computation
    # Normal equation: theta = (X^T * X)^-1 * X^T * y
    print("Computing normal equation using numerically safe pseudo-inverse...")
    theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    
    print(f"Normal equation solved. Theta shape: {theta_best.shape}")

if __name__ == "__main__":
    main()
