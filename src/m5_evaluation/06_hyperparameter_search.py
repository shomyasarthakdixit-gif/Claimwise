"""
M5.6: Hyperparameter Search Strategies
--------------------------------------
Syllabus: Hyperparameter search: grid search, random search (Bergstra-Bengio),
Bayesian optimisation with Gaussian processes at preview level.

Key Concepts:
- Grid Search:
  Exhaustive evaluation across a predefined Cartesian grid.
  Suffers from the curse of dimensionality O(k^d) and tests redundant values
  on uninformative hyperparameters.
- Random Search (Bergstra & Bengio, 2012):
  Demonstrates that in real-world ML problems, only a small subset of hyperparameters
  dominate model performance (low effective dimensionality). Random search explores
  unique parameter values continuously across each dimension, finding better optima
  in a fraction of the computational trials compared to Grid Search.
- Bayesian Optimization with Gaussian Processes:
  Surrogate-based optimization that constructs a posterior Gaussian Process regression
  model over the hyperparameter landscape. Uses an acquisition function (e.g. Expected
  Improvement / Upper Confidence Bound) to sequentially select evaluation points,
  intelligently balancing exploration of high-uncertainty zones and exploitation of peak scores.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, loguniform
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor, get_train_test_splits
from config import Config

def run_bayesian_optimization_gp(X_train, y_train, preprocessor, n_iter=15, random_state=42):
    """
    Gaussian Process Bayesian Optimization (Preview Level)
    Optimizing Ridge alpha in log-space [-2, 3] using GP surrogate and Expected Improvement.
    """
    print("Running Bayesian Optimization with Gaussian Process surrogate...")
    np.random.seed(random_state)
    
    # Objective function: 3-fold CV R2 score for given log(alpha)
    from sklearn.model_selection import cross_val_score
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    
    def objective(log_alpha):
        alpha = 10.0 ** log_alpha
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=alpha))
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        return scores.mean()
        
    # Initial random samples
    X_sample = np.random.uniform(-2, 3, size=(4, 1))
    y_sample = np.array([objective(x[0]) for x in X_sample])
    
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True, random_state=random_state)
    
    history_scores = list(y_sample)
    history_alphas = list(X_sample.flatten())
    
    for i in range(len(X_sample), n_iter):
        gp.fit(X_sample, y_sample)
        
        # Grid of candidate points
        X_candidates = np.linspace(-2, 3, 200).reshape(-1, 1)
        mu, sigma = gp.predict(X_candidates, return_std=True)
        
        # Upper Confidence Bound (UCB) acquisition function: mu + kappa * sigma
        kappa = 1.96
        ucb = mu + kappa * sigma
        next_x = X_candidates[np.argmax(ucb)]
        
        score = objective(next_x[0])
        X_sample = np.vstack([X_sample, next_x])
        y_sample = np.append(y_sample, score)
        
        history_scores.append(score)
        history_alphas.append(next_x[0])
        
    best_idx = np.argmax(y_sample)
    best_log_alpha = X_sample[best_idx][0]
    best_score = y_sample[best_idx]
    
    return {
        'best_alpha': 10.0 ** best_log_alpha,
        'best_score': best_score,
        'history_scores': np.maximum.accumulate(history_scores),
        'all_scores': history_scores,
        'all_alphas': history_alphas
    }

def plot_search_comparisons(grid_hist, rand_hist, bayes_hist, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Cumulative Optimization Trajectory (Best Score vs Iterations)
    iters = np.arange(1, len(bayes_hist['history_scores']) + 1)
    
    ax1.plot(iters, bayes_hist['history_scores'], marker='o', linestyle='-', color='#10b981', lw=2.5, label='Bayesian Opt (GP Surrogate)')
    ax1.plot(iters, rand_hist[:len(iters)], marker='s', linestyle='--', color='#3b82f6', lw=2, label='Random Search (Bergstra-Bengio)')
    ax1.plot(iters, grid_hist[:len(iters)], marker='^', linestyle=':', color='#ef4444', lw=2, label='Grid Search (Discretized)')
    
    ax1.set_xlabel('Search Iterations / Evaluations', fontweight='bold')
    ax1.set_ylabel('Best CV R² Score', fontweight='bold')
    ax1.set_title('Hyperparameter Optimization Convergence Trajectories', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Conceptual Paradigm Comparison
    ax2.axis('off')
    comparison_text = (
        "HYPERPARAMETER SEARCH STRATEGIES COMPARISON\n"
        "──────────────────────────────────────────────────────────\n"
        "1. Grid Search:\n"
        "   • Evaluates rigid combinatorial grid.\n"
        "   • Wastes compute testing insensitive hyperparameter coordinates.\n"
        "   • Complexity: Exponential O(k^d).\n\n"
        "2. Random Search (Bergstra-Bengio 2012 Theorem):\n"
        "   • Samples continuous hyperparameter distributions.\n"
        "   • For problems with low effective dimension, explores far\n"
        "     more unique hyperparameter values than Grid Search\n"
        "     within the identical compute budget.\n\n"
        "3. Bayesian Optimization (Gaussian Process Preview):\n"
        "   • Builds surrogate probabilistic model of objective function.\n"
        "   • Uses acquisition function (Upper Confidence Bound / EI)\n"
        "     to balance EXPLORATION (uncertainty) and EXPLOITATION (peak).\n"
        f"   • Achieved optimal alpha={bayes_hist['best_alpha']:.3f} (R²={bayes_hist['best_score']:.4f})\n"
        "     with minimal function evaluations."
    )
    ax2.text(0.05, 0.95, comparison_text, transform=ax2.transAxes,
             fontsize=10.5, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1'))
             
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.6] Hyperparameter search comparison saved to: {save_path}")

def main():
    print("=== M5.6: Hyperparameter Search Strategies ===")
    sample_size = 5000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    X_train, X_test, y_train, y_test = get_train_test_splits(df, random_state=random_seed)
    preprocessor = create_claimwise_preprocessor(df)
    
    # 1. Grid Search
    print("\nExecuting Grid Search...")
    ridge_pipe = Pipeline([('preprocessor', preprocessor), ('model', Ridge())])
    param_grid = {'model__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 7500.0, 10000.0, 15000.0]}
    
    grid = GridSearchCV(ridge_pipe, param_grid, cv=3, scoring='r2', n_jobs=-1, return_train_score=False)
    grid.fit(X_train, y_train)
    grid_scores = grid.cv_results_['mean_test_score']
    grid_cum_best = np.maximum.accumulate(grid_scores)
    print(f"  Grid Search Best Score: {grid.best_score_:.4f} (best alpha={grid.best_params_['model__alpha']})")
    
    # 2. Random Search (Bergstra & Bengio)
    print("\nExecuting Random Search (Bergstra-Bengio)...")
    param_dist = {'model__alpha': loguniform(0.01, 15000.0)}
    rand_search = RandomizedSearchCV(ridge_pipe, param_distributions=param_dist, n_iter=15, cv=3, scoring='r2', random_state=random_seed, n_jobs=-1)
    rand_search.fit(X_train, y_train)
    rand_scores = rand_search.cv_results_['mean_test_score']
    rand_cum_best = np.maximum.accumulate(rand_scores)
    print(f"  Random Search Best Score: {rand_search.best_score_:.4f} (best alpha={rand_search.best_params_['model__alpha']:.2f})")
    
    # 3. Bayesian Optimization
    bayes_res = run_bayesian_optimization_gp(X_train, y_train, preprocessor, n_iter=15, random_state=random_seed)
    print(f"  Bayesian Opt Best Score: {bayes_res['best_score']:.4f} (best alpha={bayes_res['best_alpha']:.2f})")
    
    plot_path = os.path.join(Config.PLOT_DIR, "hyperparameter_search_comparison.png")
    plot_search_comparisons(grid_cum_best, rand_cum_best, bayes_res, plot_path)
    print("M5.6 completed successfully.")

if __name__ == "__main__":
    main()
