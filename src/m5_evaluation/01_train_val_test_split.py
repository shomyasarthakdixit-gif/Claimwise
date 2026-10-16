"""
M5.1: Train / Validation / Test Split Discipline
------------------------------------------------
Syllabus: Train/validation/test split discipline: the three-way split,
why a test set must be held out from every modelling decision.

Key Concepts:
- Three-Way Split: Partition data into Train (e.g. 60%), Validation (20%), and Test (20%).
- Why Test Set Must Be Held Out:
  If hyperparameters, feature selection, or model architectures are selected using test set performance,
  information "leaks" from the test set into the modeling decisions. The test score becomes optimistically biased
  (data snooping / adaptive overfitting). The true generalization error on new real-world data will be worse.
  The validation set is used for tuning decisions; the test set is touched exactly ONCE at the end.
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def three_way_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2, random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    # Shuffle indices
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    return train_df, val_df, test_df

def plot_split_discipline(train_len, val_len, test_len, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Plot 1: Split proportion donut
    sizes = [train_len, val_len, test_len]
    labels = [f'Train Set\n({train_len:,} rows\n60%)', 
              f'Validation Set\n({val_len:,} rows\n20%)', 
              f'Test Set [HOLD-OUT]\n({test_len:,} rows\n20%)']
    colors = ['#2563eb', '#f59e0b', '#10b981']
    explode = (0, 0, 0.08)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140, pctdistance=0.75,
        textprops=dict(color="black", fontsize=10, weight='bold')
    )
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Three-Way Split Discipline\n(Strict Partitioning)', fontsize=12, fontweight='bold')
    
    # Plot 2: Workflow discipline diagram
    ax2.axis('off')
    discipline_text = (
        "WHY TEST SET MUST BE HELD OUT FROM EVERY DECISION:\n"
        "────────────────────────────────────────────────────────\n"
        "1. Training Set (60%):\n"
        "   • Fits model parameters (weights, coefficients, tree splits).\n\n"
        "2. Validation Set (20%):\n"
        "   • Guides hyperparameter tuning (alpha, tree depth, n_estimators).\n"
        "   • Feature selection & model architecture choice.\n"
        "   • Prevents overfitting to the training partition.\n\n"
        "3. Test Set (20% - Strictly Locked):\n"
        "   • NEVER seen during feature engineering, imputation, or tuning.\n"
        "   • Evaluated exactly ONCE at the end of the project.\n"
        "   • Provides an UNBIASED estimate of true production risk.\n"
        "   • Violating this creates Data Leakage & Optimistic Bias."
    )
    ax2.text(0.05, 0.95, discipline_text, transform=ax2.transAxes,
             fontsize=10.5, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1'))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.1] Split discipline plot saved to: {save_path}")

def main():
    print("=== M5.1: Train/Validation/Test Split Discipline ===")
    sample_size = 10000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    train_df, val_df, test_df = three_way_split(df, 0.6, 0.2, 0.2, random_state=random_seed)
    
    print(f"Data partitioning: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    target_col = 'loss'
    drop_cols = ['id', target_col]
    X_train, y_train = train_df.drop(columns=drop_cols), train_df[target_col]
    X_val, y_val = val_df.drop(columns=drop_cols), val_df[target_col]
    X_test, y_test = test_df.drop(columns=drop_cols), test_df[target_col]
    
    preprocessor = create_claimwise_preprocessor(train_df)
    
    # Demonstrate tuning on Validation set across alphas
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_val_r2 = -float("inf")
    best_alpha = None
    
    for a in alphas:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=a))
        ])
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        print(f"  Alpha={a:6.2f} -> Validation R2: {val_r2:.4f}")
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = a
            
    print(f"\nOptimal Hyperparameter chosen via VALIDATION SET: Alpha={best_alpha} (Val R2={best_val_r2:.4f})")
    
    # Final Unbiased Evaluation on Test Set
    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=best_alpha))
    ])
    final_pipe.fit(X_train, y_train)
    test_pred = final_pipe.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"FINAL UNBIASED TEST EVALUATION (Alpha={best_alpha}):")
    print(f"  Test R2:   {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print("  Note: Test set was completely held out until this exact point.")
    
    plot_path = os.path.join(Config.PLOT_DIR, "train_val_test_split.png")
    plot_split_discipline(len(train_df), len(val_df), len(test_df), plot_path)
    print("M5.1 completed successfully.")

if __name__ == "__main__":
    main()
