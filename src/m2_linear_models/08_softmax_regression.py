import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M2: Softmax Regression Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(2000, random_state=42)
    
    # Derived educational classification target - NOT a fraud label (Multi-class)
    # Bin loss into 3 categories: low, medium, high educational loss levels
    df['loss_tier_educational'] = pd.qcut(df['loss'], q=3, labels=[0, 1, 2])
    print("Created Derived educational classification target (3 tiers) - NOT a fraud label")
    
    X_train, X_test, y_train, y_test = get_train_test_splits(df, target_col='loss_tier_educational')
    X_train = X_train.drop(columns=['loss'])
    X_test = X_test.drop(columns=['loss'])
    
    preprocessor = create_claimwise_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train_processed, y_train)
    print(f"Softmax Regression Accuracy: {model.score(X_test_processed, y_test):.4f}")

if __name__ == "__main__":
    main()
