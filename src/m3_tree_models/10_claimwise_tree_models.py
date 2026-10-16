import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from utils.model_utils import save_model
from config import Config

def main():
    print("M3: ClaimWise Production-Style Tree Baseline")
    
    sample_size = 10000
    random_seed = 42
    print(f"Loading reproducible sample of {sample_size} records using random_seed={random_seed}...")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    preprocessor = create_claimwise_preprocessor(df)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42))
    ])
    
    print("Training RandomForestRegressor on actual loss target...")
    pipeline.fit(X_train, y_train)
    
    test_score = pipeline.score(X_test, y_test)
    print(f"Random Forest Test R2: {test_score:.4f}")
    
    # Save Model
    save_model(pipeline, "random_forest.pkl")
    
    # Evaluate and Compare
    eval_dir = Config.EVALUATION_DIR
    os.makedirs(eval_dir, exist_ok=True)
    comp_file = os.path.join(eval_dir, "model_comparison.csv")
    
    # Update model comparison CSV
    new_result = pd.DataFrame({"model": ["Random Forest Baseline"], "r2": [test_score]})
    if os.path.exists(comp_file):
        comp_df = pd.read_csv(comp_file)
        # Avoid duplicate appends if script is run multiple times
        comp_df = comp_df[comp_df['model'] != "Random Forest Baseline"]
        comp_df = pd.concat([comp_df, new_result], ignore_index=True)
    else:
        comp_df = new_result
    
    comp_df.to_csv(comp_file, index=False)
    print(f"Updated model comparison at {comp_file}")
    
    # Generate Plots
    plot_dir = Config.PLOT_DIR
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Model Comparison Plot
    plt.figure(figsize=(8, 5))
    plt.bar(comp_df['model'], comp_df['r2'], color=['blue', 'green'])
    plt.ylabel('R2 Score')
    plt.title('M2 Ridge vs M3 Random Forest Baseline')
    plt.savefig(os.path.join(plot_dir, "model_comparison.png"))
    plt.close()
    
    # 2. Actual vs Predicted Plot
    y_pred = pipeline.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Loss')
    plt.ylabel('Predicted Loss')
    plt.title('Actual vs Predicted Loss (Random Forest)')
    plt.savefig(os.path.join(plot_dir, "actual_vs_predicted.png"))
    plt.close()
    
    # 3. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Loss')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Random Forest)')
    plt.savefig(os.path.join(plot_dir, "residual_plot.png"))
    plt.close()
    
    # 4. Feature Importance
    importances = pipeline.named_steps['model'].feature_importances_
    # For a real pipeline, extracting original feature names is complex due to OneHotEncoding.
    # We'll plot the top 20 raw indices for the dashboard concept.
    top_indices = importances.argsort()[-20:][::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), importances[top_indices])
    plt.xticks(range(20), [f"F_{i}" for i in top_indices], rotation=45)
    plt.title('Top 20 Feature Importances')
    plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
    plt.close()
    
    pd.DataFrame({
        "feature_index": top_indices, 
        "importance": importances[top_indices]
    }).to_csv(os.path.join(Config.MODELS_DIR, "feature_importances.csv"), index=False)
    
    print("Artifacts generated in Output/plot/ and Output/models/.")

if __name__ == "__main__":
    main()
