import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M3: Decision Tree Demonstration")
    print("Concepts: greedy construction, Gini, entropy, regression/MSE, stopping criteria, bias-variance.\n")
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(1000, random_state=42)
    threshold = df['loss'].quantile(0.75)
    df['high_loss_educational'] = (df['loss'] >= threshold).astype(int)
    
    X_train, X_test, y_train_cls, y_test_cls = get_train_test_splits(df, target_col='high_loss_educational')
    X_train = X_train.drop(columns=['loss'])
    X_test = X_test.drop(columns=['loss'])
    
    preprocessor = create_claimwise_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("1. Classification: Gini vs Entropy (Information Gain)")
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf_gini.fit(X_train_processed, y_train_cls)
    print(f"Gini split accuracy: {clf_gini.score(X_test_processed, y_test_cls):.4f}")
    
    clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    clf_entropy.fit(X_train_processed, y_train_cls)
    print(f"Entropy split accuracy: {clf_entropy.score(X_test_processed, y_test_cls):.4f}")
    
    print("\n2. Regression: MSE Splitting")
    _, _, y_train_reg, y_test_reg = get_train_test_splits(df, target_col='loss')
    reg_mse = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
    reg_mse.fit(X_train_processed, y_train_reg)
    print(f"MSE Regressor R2: {reg_mse.score(X_test_processed, y_test_reg):.4f}")
    
    print("\n3. Bias-Variance & Stopping Criteria")
    unrestricted_tree = DecisionTreeRegressor(random_state=42)
    unrestricted_tree.fit(X_train_processed, y_train_reg)
    print(f"Unrestricted Tree Train R2 (Overfit/High Variance): {unrestricted_tree.score(X_train_processed, y_train_reg):.4f}")
    print(f"Unrestricted Tree Test R2: {unrestricted_tree.score(X_test_processed, y_test_reg):.4f}")

if __name__ == "__main__":
    main()
