"""
M5.2: Cross-Validation & Nested CV
-----------------------------------
Syllabus: k-fold and stratified k-fold cross-validation; nested CV when both
hyperparameter selection and unbiased evaluation are needed.

Key Concepts:
- k-Fold CV: Splits data into k subsets, rotating validation across folds to reduce split variance.
- Stratified k-Fold CV: Preserves class or severity tier proportions across folds, critical when
  distribution tails (e.g. catastrophic high-cost claims) must appear proportionally in all splits.
- Nested CV:
  * Inner loop (e.g. 3 folds): Optimizes hyperparameters (Grid/Random Search).
  * Outer loop (e.g. 5 folds): Evaluates the best model on held-out test folds.
  * Essential because tuning hyperparameters and assessing generalization on the same CV loop
    yields an optimistically biased performance estimate.
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def run_cross_validation_experiments(X, y, preprocessor, random_state=42):
    print("Running 5-Fold Standard Cross-Validation...")
    ridge = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    kfold_scores = cross_val_score(ridge, X, y, cv=kfold, scoring='r2', n_jobs=-1)
    print(f"  Standard 5-Fold R2: Mean = {kfold_scores.mean():.4f} (std = {kfold_scores.std():.4f})")
    print(f"  Fold scores: {[round(s, 4) for s in kfold_scores]}")
    
    # 2. Stratified k-Fold on Severity Tiers (binned continuous target)
    print("\nRunning 5-Fold Stratified Cross-Validation on Severity Tiers...")
    bins = np.percentile(y, [0, 33.3, 66.6, 100])
    severity_tiers = np.digitize(y, bins[1:-1])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    skf_scores = []
    for train_idx, val_idx in skf.split(X, severity_tiers):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        ridge.fit(X_tr, y_tr)
        skf_scores.append(r2_score(y_va, ridge.predict(X_va)))
    skf_scores = np.array(skf_scores)
    print(f"  Stratified 5-Fold R2: Mean = {skf_scores.mean():.4f} (std = {skf_scores.std():.4f})")
    print(f"  Fold scores: {[round(s, 4) for s in skf_scores]}")
    
    # 3. Nested Cross-Validation (Outer 5-fold, Inner 3-fold)
    print("\nRunning Nested Cross-Validation (Outer=5 folds, Inner=3 folds)...")
    param_grid = {'model__alpha': [0.1, 1.0, 10.0, 100.0]}
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1
    )
    
    nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='r2', n_jobs=-1)
    print(f"  Nested CV (5x3) R2: Mean = {nested_scores.mean():.4f} (std = {nested_scores.std():.4f})")
    print(f"  Outer loop scores: {[round(s, 4) for s in nested_scores]}")
    
    # 4. Standard Non-nested CV with hyperparameter tuning on same folds (to show optimistic bias)
    grid_search.fit(X, y)
    non_nested_best_score = grid_search.best_score_
    print(f"  Non-nested Best CV Score: {non_nested_best_score:.4f} (Optimistic estimation)")
    
    return {
        'kfold': kfold_scores,
        'stratified': skf_scores,
        'nested': nested_scores,
        'non_nested_best': non_nested_best_score
    }

def plot_cv_comparison(cv_results, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Boxplot of Fold Scores
    data = [cv_results['kfold'], cv_results['stratified'], cv_results['nested']]
    labels = ['Standard 5-Fold\n(Regression)', 'Stratified 5-Fold\n(Severity Tiers)', 'Nested CV (5x3)\n(Unbiased Tuning)']
    
    bplot = ax1.boxplot(data, tick_labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='#dbeafe', color='#1d4ed8'),
                        medianprops=dict(color='#dc2626', linewidth=2),
                        whiskerprops=dict(color='#1d4ed8', linewidth=1.5),
                        capprops=dict(color='#1d4ed8', linewidth=1.5))
    
    colors = ['#93c5fd', '#86efac', '#fde047']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    ax1.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Cross-Validation Performance Distributions', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add means as diamond markers
    means = [np.mean(d) for d in data]
    ax1.plot([1, 2, 3], means, 'rD', markersize=7, label='Fold Mean')
    ax1.legend(loc='lower right')
    
    # Explanatory Text / Architecture Diagram
    ax2.axis('off')
    concept_text = (
        "CROSS-VALIDATION & NESTED CV ARCHITECTURE\n"
        "──────────────────────────────────────────────────────────\n"
        "• Standard k-Fold CV:\n"
        f"  Mean R²: {np.mean(cv_results['kfold']):.4f} ± {np.std(cv_results['kfold']):.4f}\n"
        "  Rotates holdout folds to reduce partition variance.\n\n"
        "• Stratified k-Fold CV:\n"
        f"  Mean R²: {np.mean(cv_results['stratified']):.4f} ± {np.std(cv_results['stratified']):.4f}\n"
        "  Guarantees rare extreme claim severities appear evenly.\n\n"
        "• Nested Cross-Validation (Inner 3-Fold + Outer 5-Fold):\n"
        f"  True Unbiased R²: {np.mean(cv_results['nested']):.4f} ± {np.std(cv_results['nested']):.4f}\n"
        f"  Naive Non-Nested CV: {cv_results['non_nested_best']:.4f} (Optimistic)\n\n"
        "WHY NESTED CV IS CRITICAL:\n"
        "When hyperparameters are optimized on the same CV used to\n"
        "report accuracy, the model selection process overfits to the CV\n"
        "folds. Nested CV decouples model optimization (inner loop) from\n"
        "performance estimation (outer loop), yielding strictly unbiased bounds."
    )
    ax2.text(0.05, 0.95, concept_text, transform=ax2.transAxes,
             fontsize=10.5, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1'))
             
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.2] Nested CV plot saved to: {save_path}")

def main():
    print("=== M5.2: Cross-Validation & Nested CV ===")
    sample_size = 6000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    X = df.drop(columns=['id', 'loss'])
    y = df['loss']
    
    preprocessor = create_claimwise_preprocessor(df)
    results = run_cross_validation_experiments(X, y, preprocessor, random_state=random_seed)
    
    plot_path = os.path.join(Config.PLOT_DIR, "nested_cv_results.png")
    plot_cv_comparison(results, plot_path)
    print("M5.2 completed successfully.")

if __name__ == "__main__":
    main()
