import os
from sklearn.linear_model import SGDRegressor
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M2: Gradient Descent Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(1000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    import numpy as np
    
    # Add bias term
    X_b = np.c_[np.ones((X_train_processed.shape[0], 1)), X_train_processed]
    y = y_train.values.reshape(-1, 1)
    
    # Gradient Descent Mathematics
    eta = 0.01  # learning rate
    n_iterations = 1000
    m = len(X_b)
    
    theta = np.random.randn(X_b.shape[1], 1)
    
    print("Starting educational Gradient Descent loop...")
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        
    print(f"Gradient Descent completed. Theta shape: {theta.shape}")
    
    # Compare with Test Data
    X_test_b = np.c_[np.ones((X_test_processed.shape[0], 1)), X_test_processed]
    y_test_pred = X_test_b.dot(theta)
    
    # Calculate R2 manually
    ss_total = np.sum((y_test.values.reshape(-1, 1) - np.mean(y_test)) ** 2)
    ss_res = np.sum((y_test.values.reshape(-1, 1) - y_test_pred) ** 2)
    r2_score = 1 - (ss_res / ss_total)
    
    print(f"Manual Gradient Descent R2 Score: {r2_score:.4f}")

if __name__ == "__main__":
    main()
