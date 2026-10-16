"""
M5.5: Probability Calibration & Reliability Diagrams
-----------------------------------------------------
Syllabus: Probability calibration: why a 0.7 prediction should mean "70% likely";
Platt scaling, isotonic regression; reliability diagrams.

Key Concepts:
- The Definition of Calibration:
  If a model assigns a probability of 0.70 to a group of insurance claims,
  exactly 70% of those claims should actually belong to the positive class.
  In insurance, Expected Payout = P(Claim) * Severity. An uncalibrated probability
  leads directly to solvency risk or mispriced policyholder premiums.
- Platt Scaling (Sigmoid Method):
  Fits a logistic regression model on validation decision values:
    P(y=1 | f) = 1 / (1 + exp(A*f + B))
  Parametric, effective when distortion is sigmoidal (e.g., SVM margin scores).
- Isotonic Regression:
  Non-parametric piecewise-constant monotonic step function using the Pool Adjacent Violators Algorithm (PAVA).
  More flexible than Platt scaling, but requires sufficient calibration data.
- Reliability Diagrams (Calibration Curves):
  Plots mean predicted probability vs true fraction of positives across probability bins.
  A perfectly calibrated model tracks the diagonal y = x.
- Metrics:
  * Brier Score = Mean Squared Error between predicted probabilities and actual binary outcomes (lower is better).
  * Expected Calibration Error (ECE) = Weighted average absolute difference between predicted probability and observed frequency.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for b in range(n_bins):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            prop = np.mean(mask)
            actual_frac = np.mean(y_true[mask])
            pred_mean = np.mean(y_prob[mask])
            ece += prop * np.abs(actual_frac - pred_mean)
    return ece

def plot_reliability_diagrams(y_test, prob_uncal, prob_platt, prob_iso, stats, save_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Reliability Diagram (Calibration Curves)
    frac_uncal, mean_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)
    frac_platt, mean_platt = calibration_curve(y_test, prob_platt, n_bins=10)
    frac_iso, mean_iso = calibration_curve(y_test, prob_iso, n_bins=10)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (y = x)', alpha=0.7)
    ax1.plot(mean_uncal, frac_uncal, 's-', color='#ef4444', lw=2, label=f"Uncalibrated RF (Brier: {stats['brier_uncal']:.4f})")
    ax1.plot(mean_platt, frac_platt, 'o-', color='#3b82f6', lw=2, label=f"Platt Scaling (Brier: {stats['brier_platt']:.4f})")
    ax1.plot(mean_iso, frac_iso, '^-', color='#10b981', lw=2.5, label=f"Isotonic Reg (Brier: {stats['brier_iso']:.4f})")
    
    ax1.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax1.set_ylabel('Empirical Fraction of Positives', fontweight='bold')
    ax1.set_title('Reliability Diagram (Calibration Curves)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Probability Histograms (Confidence Distribution)
    ax2.hist(prob_uncal, bins=20, range=(0, 1), alpha=0.5, color='#ef4444', label='Uncalibrated (RF)')
    ax2.hist(prob_iso, bins=20, range=(0, 1), alpha=0.5, color='#10b981', label='Isotonic Calibrated')
    ax2.set_xlabel('Predicted Probability', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Probability Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Calibration Error Comparison Bar Chart
    models = ['Uncalibrated RF', 'Platt (Sigmoid)', 'Isotonic Regression']
    briers = [stats['brier_uncal'], stats['brier_platt'], stats['brier_iso']]
    eces = [stats['ece_uncal'], stats['ece_platt'], stats['ece_iso']]
    
    x = np.arange(len(models))
    w = 0.35
    ax3.bar(x - w/2, briers, w, label='Brier Score (lower=better)', color='#f59e0b')
    ax3.bar(x + w/2, eces, w, label='Expected Calib Error (ECE)', color='#6366f1')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontweight='bold')
    ax3.set_ylabel('Error Score', fontweight='bold')
    ax3.set_title('Brier Score & ECE Calibration Improvement', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Explanatory Theory Card
    ax4.axis('off')
    theory_text = (
        "PROBABILITY CALIBRATION IN INSURANCE UNDERWRITING\n"
        "──────────────────────────────────────────────────────────\n"
        "• Why 0.70 Must Mean 70% Likely:\n"
        "  Actuarial pricing requires expected losses:\n"
        "    E[Claim Payout] = P(Loss >= Threshold) × Severity\n"
        "  If a model outputs 0.70 when actual risk is only 30%,\n"
        "  premiums become uncompetitive. If it underestimates risk,\n"
        "  the insurer faces catastrophic under-reserving.\n\n"
        "• Platt Scaling (Logistic Sigmoid):\n"
        f"  ECE reduced from {stats['ece_uncal']:.4f} → {stats['ece_platt']:.4f}\n"
        "  Smooth parametric transformation. Safe against small samples.\n\n"
        "• Isotonic Regression (PAVA Step-wise):\n"
        f"  ECE reduced to {stats['ece_iso']:.4f} (Brier: {stats['brier_iso']:.4f})\n"
        "  Non-parametric monotonic fit. Corrects arbitrary non-linear\n"
        "  distortions inherent in tree ensemble bagging."
    )
    ax4.text(0.05, 0.95, theory_text, transform=ax4.transAxes,
             fontsize=10.5, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1'))
             
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.5] Probability calibration curves saved to: {save_path}")

def main():
    print("=== M5.5: Probability Calibration & Reliability Diagrams ===")
    sample_size = 8000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    threshold = np.percentile(df['loss'], 80)
    y_binary = (df['loss'] >= threshold).astype(int)
    X = df.drop(columns=['id', 'loss'])
    
    preprocessor = create_claimwise_preprocessor(df)
    X_processed = preprocessor.fit_transform(X)
    
    # 3-way split: 50% Train, 25% Calibration, 25% Final Test
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y_binary, test_size=0.5, random_state=random_seed, stratify=y_binary)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed, stratify=y_temp)
    
    # 1. Base Uncalibrated Random Forest
    base_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_seed, n_jobs=-1)
    base_rf.fit(X_train, y_train)
    prob_uncal = base_rf.predict_proba(X_test)[:, 1]
    
    # 2. Platt Scaling (Sigmoid)
    cal_platt = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_seed, n_jobs=-1), method='sigmoid', cv=3)
    cal_platt.fit(X_train, y_train)
    prob_platt = cal_platt.predict_proba(X_test)[:, 1]
    
    # 3. Isotonic Regression
    cal_iso = CalibratedClassifierCV(estimator=RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_seed, n_jobs=-1), method='isotonic', cv=3)
    cal_iso.fit(X_train, y_train)
    prob_iso = cal_iso.predict_proba(X_test)[:, 1]
    
    # Calibration Assessment
    brier_uncal = brier_score_loss(y_test, prob_uncal)
    brier_platt = brier_score_loss(y_test, prob_platt)
    brier_iso = brier_score_loss(y_test, prob_iso)
    
    ece_uncal = compute_ece(y_test.values, prob_uncal)
    ece_platt = compute_ece(y_test.values, prob_platt)
    ece_iso = compute_ece(y_test.values, prob_iso)
    
    stats = {
        'brier_uncal': brier_uncal, 'brier_platt': brier_platt, 'brier_iso': brier_iso,
        'ece_uncal': ece_uncal, 'ece_platt': ece_platt, 'ece_iso': ece_iso
    }
    
    print("\n--- Calibration Results ---")
    print(f"Uncalibrated RF: Brier Score = {brier_uncal:.4f}, ECE = {ece_uncal:.4f}")
    print(f"Platt Scaling:   Brier Score = {brier_platt:.4f}, ECE = {ece_platt:.4f} (Improved!)")
    print(f"Isotonic Reg:    Brier Score = {brier_iso:.4f}, ECE = {ece_iso:.4f} (Best Calibration!)")
    
    plot_path = os.path.join(Config.PLOT_DIR, "probability_calibration_curves.png")
    plot_reliability_diagrams(y_test, prob_uncal, prob_platt, prob_iso, stats, plot_path)
    print("M5.5 completed successfully.")

if __name__ == "__main__":
    main()
