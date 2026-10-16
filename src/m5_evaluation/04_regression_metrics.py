"""
M5.4: Regression Metrics & Outlier Sensitivity
----------------------------------------------
Syllabus: Regression metrics: RMSE, MAE, MAPE, R²;
the choice between RMSE and MAE depending on outlier sensitivity.

Key Concepts:
- RMSE (Root Mean Squared Error):
  Penalizes large errors quadratically (L2 loss). Highly sensitive to outliers.
  Best when large errors are disproportionately catastrophic to business operations.
- MAE (Mean Absolute Error):
  Penalizes errors linearly (L1 loss). Robust to extreme outliers.
  Corresponds to the conditional median, making it ideal for insurance claim distributions
  characterized by long right-hand tails.
- MAPE (Mean Absolute Percentage Error):
  Expresses error as a percentage of actual loss. Scale-independent,
  but susceptible to explosion when ground-truth values approach zero.
- R² (Coefficient of Determination):
  Proportion of variance explained by the model relative to the constant mean predictor.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor, get_train_test_splits
from config import Config

def calculate_mape(y_true, y_pred):
    epsilon = 1.0  # Prevent division by zero
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0

def evaluate_regression_models(models, X_train, y_train, X_test, y_test, preprocessor):
    results = []
    predictions = {}
    
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        predictions[name] = y_pred
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
            'R2': r2
        })
        
    return pd.DataFrame(results), predictions

def demonstrate_outlier_sensitivity(y_test, y_pred):
    """
    Simulates introducing 1% catastrophic extreme claim outliers
    to quantify the surge in RMSE vs stability in MAE.
    """
    clean_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    clean_mae = mean_absolute_error(y_test, y_pred)
    
    # Inject 1% extreme outliers (claims that are 10x normal severity)
    y_test_contaminated = np.array(y_test, dtype=float, copy=True)
    y_test_contaminated.setflags(write=1)
    n_outliers = max(1, int(0.01 * len(y_test_contaminated)))
    outlier_idx = np.random.choice(len(y_test_contaminated), n_outliers, replace=False)
    y_test_contaminated[outlier_idx] += 30000.0  # $30k outlier shock
    
    dirty_rmse = np.sqrt(mean_squared_error(y_test_contaminated, y_pred))
    dirty_mae = mean_absolute_error(y_test_contaminated, y_pred)
    
    rmse_increase_pct = ((dirty_rmse - clean_rmse) / clean_rmse) * 100
    mae_increase_pct = ((dirty_mae - clean_mae) / clean_mae) * 100
    
    return {
        'clean_rmse': clean_rmse, 'dirty_rmse': dirty_rmse, 'rmse_surge': rmse_increase_pct,
        'clean_mae': clean_mae, 'dirty_mae': dirty_mae, 'mae_surge': mae_increase_pct
    }

def plot_regression_diagnostics(results_df, outlier_stats, y_test, y_pred, save_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))
    
    # 1. Model Comparison across Metrics (Normalized for Comparison)
    models = results_df['Model'].values
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, results_df['RMSE'], width, label='RMSE ($)', color='#ef4444')
    ax1.bar(x, results_df['MAE'], width, label='MAE ($)', color='#3b82f6')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight='bold')
    ax1.set_ylabel('Loss Metric ($)', fontweight='bold')
    ax1.set_title('RMSE vs MAE by Algorithm\n(Notice RMSE is significantly higher due to tail errors)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. R² Comparison
    ax2.bar(models, results_df['R2'], color=['#64748b', '#10b981', '#059669'], width=0.5)
    ax2.set_ylabel('R² Score', fontweight='bold')
    ax2.set_title('Model Goodness of Fit (R²)', fontweight='bold')
    ax2.set_ylim(0, max(results_df['R2']) * 1.25)
    for i, v in enumerate(results_df['R2']):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Outlier Sensitivity Experiment
    cats = ['Clean Test Set', 'Contaminated (+1% Catastrophic Outliers)']
    rmse_vals = [outlier_stats['clean_rmse'], outlier_stats['dirty_rmse']]
    mae_vals = [outlier_stats['clean_mae'], outlier_stats['dirty_mae']]
    
    x2 = np.arange(2)
    ax3.bar(x2 - 0.15, rmse_vals, 0.3, label=f"RMSE (+{outlier_stats['rmse_surge']:.1f}% surge!)", color='#dc2626')
    ax3.bar(x2 + 0.15, mae_vals, 0.3, label=f"MAE (+{outlier_stats['mae_surge']:.1f}% robust)", color='#2563eb')
    ax3.set_xticks(x2)
    ax3.set_xticklabels(cats, fontweight='bold')
    ax3.set_ylabel('Error Magnitude ($)', fontweight='bold')
    ax3.set_title('Outlier Sensitivity Stress Test\n(Why MAE is robust while RMSE explodes on outliers)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Error Distribution (Residuals vs Theoretical Density)
    residuals = y_test - y_pred
    ax4.hist(residuals, bins=50, range=(-6000, 6000), density=True, alpha=0.6, color='#6366f1', label='Residual Error Distribution')
    ax4.axvline(0, color='r', linestyle='--', label='Zero Bias')
    ax4.axvline(np.mean(residuals), color='black', linestyle=':', label=f'Mean Residual (${np.mean(residuals):.1f})')
    ax4.set_xlabel('Residual ($ = Actual - Predicted)', fontweight='bold')
    ax4.set_ylabel('Density', fontweight='bold')
    ax4.set_title('Residual Error Distribution & Heavy Tail Analysis', fontweight='bold')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.4] Regression metrics distribution plot saved to: {save_path}")

def main():
    print("=== M5.4: Regression Metrics & Outlier Sensitivity ===")
    sample_size = 10000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    X_train, X_test, y_train, y_test = get_train_test_splits(df, random_state=random_seed)
    preprocessor = create_claimwise_preprocessor(df)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_seed, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_seed)
    }
    
    print("\nEvaluating regression models across all syllabus metrics...")
    results_df, predictions = evaluate_regression_models(models, X_train, y_train, X_test, y_test, preprocessor)
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    
    # Outlier Sensitivity Experiment
    gb_pred = predictions['Gradient Boosting']
    outlier_stats = demonstrate_outlier_sensitivity(y_test, gb_pred)
    print("\n--- Outlier Sensitivity Test (1% Extreme Outlier Injection) ---")
    print(f"  RMSE surge: +{outlier_stats['rmse_surge']:.2f}% (from ${outlier_stats['clean_rmse']:,.1f} to ${outlier_stats['dirty_rmse']:,.1f})")
    print(f"  MAE surge:  +{outlier_stats['mae_surge']:.2f}% (from ${outlier_stats['clean_mae']:,.1f} to ${outlier_stats['dirty_mae']:,.1f})")
    print("  Conclusion: RMSE is quadratically penalized by extreme insurance claims; MAE provides robust linear protection.")
    
    plot_path = os.path.join(Config.PLOT_DIR, "regression_metrics_distribution.png")
    plot_regression_diagnostics(results_df, outlier_stats, y_test, gb_pred, plot_path)
    
    # Save CSV summary
    os.makedirs(Config.EVALUATION_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(Config.EVALUATION_DIR, "m5_regression_summary.csv"), index=False)
    print("M5.4 completed successfully.")

if __name__ == "__main__":
    main()
