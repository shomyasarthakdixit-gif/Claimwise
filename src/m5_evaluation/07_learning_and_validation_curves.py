"""
M5.7: Learning Curves & Validation Curves
-----------------------------------------
Syllabus: Learning curves and validation curves:
diagnosing under-fitting, over-fitting, data scarcity from a single plot.

Key Concepts:
- Learning Curves:
  Evaluates Training Score vs Cross-Validation Score as training dataset size grows.
  * Underfitting (High Bias): Both train and val scores converge quickly to an unsatisfactory low level.
    The gap is small. Adding more data will NOT help. The model lacks representational capacity.
  * Overfitting (High Variance): Train score is high, val score is significantly lower, large gap.
    The model memorized training artifacts. Adding more data or regularizing will improve generalization.
  * Data Scarcity: The validation curve is still sloping upward steeply at the maximum sample size.
    Collecting more training data will directly yield performance gains.
- Validation Curves:
  Evaluates Training Score vs Validation Score as a key hyperparameter varies (e.g. Ridge alpha).
  Demonstrates the Bias-Variance tradeoff sweet spot.
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
from sklearn.model_selection import learning_curve, validation_curve, KFold
from sklearn.pipeline import Pipeline
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def plot_curves(train_sizes, train_scores_lc, val_scores_lc, param_range, train_scores_vc, val_scores_vc, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Learning Curve
    train_mean_lc = np.mean(train_scores_lc, axis=1)
    train_std_lc = np.std(train_scores_lc, axis=1)
    val_mean_lc = np.mean(val_scores_lc, axis=1)
    val_std_lc = np.std(val_scores_lc, axis=1)
    
    ax1.plot(train_sizes, train_mean_lc, 'o-', color='#dc2626', lw=2, label='Training Score')
    ax1.fill_between(train_sizes, train_mean_lc - train_std_lc, train_mean_lc + train_std_lc, alpha=0.15, color='#dc2626')
    
    ax1.plot(train_sizes, val_mean_lc, 's-', color='#16a34a', lw=2, label='Cross-Validation Score')
    ax1.fill_between(train_sizes, val_mean_lc - val_std_lc, val_mean_lc + val_std_lc, alpha=0.15, color='#16a34a')
    
    ax1.set_xlabel('Number of Training Examples', fontweight='bold')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('Learning Curve: Diagnosing Data Scarcity & Generalization', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate diagnosis
    gap = train_mean_lc[-1] - val_mean_lc[-1]
    ax1.annotate(f"Final Generalization Gap = {gap:.4f}\n(Moderate Variance, Stable Convergence)",
                 xy=(train_sizes[-1], val_mean_lc[-1]), xytext=(train_sizes[-2], val_mean_lc[-1] - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#94a3b8'))
    
    # 2. Validation Curve
    train_mean_vc = np.mean(train_scores_vc, axis=1)
    train_std_vc = np.std(train_scores_vc, axis=1)
    val_mean_vc = np.mean(val_scores_vc, axis=1)
    val_std_vc = np.std(val_scores_vc, axis=1)
    
    ax2.semilogx(param_range, train_mean_vc, 'o-', color='#dc2626', lw=2, label='Training Score')
    ax2.fill_between(param_range, train_mean_vc - train_std_vc, train_mean_vc + train_std_vc, alpha=0.15, color='#dc2626')
    
    ax2.semilogx(param_range, val_mean_vc, 's-', color='#2563eb', lw=2, label='Validation Score')
    ax2.fill_between(param_range, val_mean_vc - val_std_vc, val_mean_vc + val_std_vc, alpha=0.15, color='#2563eb')
    
    best_param_idx = np.argmax(val_mean_vc)
    best_param = param_range[best_param_idx]
    ax2.axvline(best_param, color='purple', linestyle='--', label=f'Optimal Regularization (α={best_param})')
    
    ax2.set_xlabel('Ridge Regularization Strength (α)', fontweight='bold')
    ax2.set_ylabel('R² Score', fontweight='bold')
    ax2.set_title('Validation Curve: Diagnosing Under-fitting vs Over-fitting', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.7] Learning and validation curves saved to: {save_path}")

def main():
    print("=== M5.7: Learning Curves & Validation Curves ===")
    sample_size = 6000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    X = df.drop(columns=['id', 'loss'])
    y = df['loss']
    preprocessor = create_claimwise_preprocessor(df)
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=100.0))
    ])
    
    cv = KFold(n_splits=4, shuffle=True, random_state=random_seed)
    
    # 1. Compute Learning Curve
    print("Computing Learning Curve across training set fractions (10% to 100%)...")
    train_sizes, train_scores_lc, val_scores_lc = learning_curve(
        estimator=pipe,
        X=X,
        y=y,
        train_sizes=np.linspace(0.15, 1.0, 6),
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=random_seed
    )
    print(f"  Final Train R2: {np.mean(train_scores_lc[-1]):.4f}, Final Val R2: {np.mean(val_scores_lc[-1]):.4f}")
    
    # 2. Compute Validation Curve across Ridge alpha
    print("\nComputing Validation Curve across hyperparameter alpha...")
    param_range = np.logspace(-1, 4, 6)  # 0.1 to 10000
    train_scores_vc, val_scores_vc = validation_curve(
        estimator=pipe,
        X=X,
        y=y,
        param_name='model__alpha',
        param_range=param_range,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    best_idx = np.argmax(np.mean(val_scores_vc, axis=1))
    print(f"  Best Alpha from Validation Curve: {param_range[best_idx]} (Val R2: {np.mean(val_scores_vc[best_idx]):.4f})")
    
    plot_path = os.path.join(Config.PLOT_DIR, "learning_and_validation_curves.png")
    plot_curves(train_sizes, train_scores_lc, val_scores_lc, param_range, train_scores_vc, val_scores_vc, plot_path)
    print("M5.7 completed successfully.")

if __name__ == "__main__":
    main()
