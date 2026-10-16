import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M3: Feature Importance Demonstration")
    print("Concepts: Impurity/Gini Importance, Permutation Importance, SHAP.\n")
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(1000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # We use a small subset of features to make output readable
    X_train_small = X_train_processed[:, :5]
    X_test_small = X_test_processed[:, :5]
    
    rf = RandomForestRegressor(n_estimators=20, random_state=42)
    rf.fit(X_train_small, y_train)
    
    print("1. Impurity-based Feature Importance:")
    for i, imp in enumerate(rf.feature_importances_):
        print(f"Feature {i}: {imp:.4f}")
        
    print("\n2. Permutation Importance:")
    perm_imp = permutation_importance(rf, X_test_small, y_test, n_repeats=5, random_state=42)
    for i, imp in enumerate(perm_imp.importances_mean):
        print(f"Feature {i}: {imp:.4f}")
        
    print("\n3. SHAP (Shapley Additive exPlanations):")
    try:
        import shap
        # SHAP as a conceptual preview
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test_small[:5]) # just 5 samples
        print(f"SHAP values successfully computed. Shape: {shap_values.shape}")
    except ImportError:
        print("SHAP is not installed. Gracefully skipping SHAP demonstration.")

if __name__ == "__main__":
    main()
