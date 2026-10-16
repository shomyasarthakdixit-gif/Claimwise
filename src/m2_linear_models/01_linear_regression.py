import os
from sklearn.linear_model import LinearRegression
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from utils.model_utils import save_model
from config import Config

def main():
    print("M2: Linear Regression Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(1000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    score = model.score(X_test_processed, y_test)
    print(f"Linear Regression R2 Score on sample: {score:.4f}")
    save_model(model, "m2_01_linear_regression.pkl")

if __name__ == "__main__":
    main()
