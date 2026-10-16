import os
import json
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from utils.model_utils import save_model
from config import Config

def main():
    print("=== FINAL MODEL-SELECTION EXPERIMENT ===")
    sample_size = 15000
    random_seed = 42
    print(f"Loading reproducible sample of {sample_size} records using random_seed={random_seed}...")
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    preprocessor = create_claimwise_preprocessor(df)
    
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_seed)
    }
    
    results = []
    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None
    
    print("\nEvaluating models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        import numpy as np
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Metrics for {name}:")
        print(f"  Test R2: {r2:.4f}")
        print(f"  RMSE:    {rmse:.4f}")
        print(f"  MAE:     {mae:.4f}")
        
        results.append({
            "model": name,
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\n=== SELECTION ===")
    print(f"Selected Model: {best_model_name}")
    print("Reason: Achieved the highest test R2 among the models in the controlled final comparison.")
    
    print(f"\nSaving {best_model_name} as 'claimwise_final_model.pkl'...")
    save_model(best_pipeline, "claimwise_final_model.pkl")
    
    # Save the results
    os.makedirs(Config.EVALUATION_DIR, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(Config.EVALUATION_DIR, "final_model_selection.csv"), index=False)
    
    with open(os.path.join(Config.EVALUATION_DIR, "final_model_selection.json"), "w") as f:
        json.dump({
            "sample_size": sample_size,
            "random_seed": random_seed,
            "selected_model": best_model_name,
            "reason": "Achieved the highest test R2 among the models in the controlled final comparison.",
            "results": results
        }, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()
