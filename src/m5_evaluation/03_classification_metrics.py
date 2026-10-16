"""
M5.3: Classification Metrics & Class Imbalance Impact
------------------------------------------------------
Syllabus: Classification metrics: accuracy (and when it lies),
precision/recall/F1, the confusion matrix, ROC-AUC, PR-AUC; class-imbalance impact.

Key Concepts:
- Accuracy and "When It Lies":
  On an imbalanced insurance dataset (e.g., 90% normal claims, 10% severe/fraud claims),
  a naive majority-class dummy model achieves 90% accuracy while having ZERO recall,
  completely failing to detect any high-severity event.
- Precision vs Recall vs F1-Score:
  * Precision: Of all predicted severe claims, how many were actually severe? (Avoid false alarms).
  * Recall: Of all actual severe claims, how many did we catch? (Avoid costly undetected claims).
  * F1-Score: Harmonic mean of precision and recall.
- Confusion Matrix: True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).
- ROC-AUC vs PR-AUC Under Imbalance:
  ROC-AUC evaluates True Positive Rate vs False Positive Rate. Under severe class imbalance,
  a large pool of True Negatives suppresses the False Positive Rate, making ROC-AUC deceptively optimistic.
  Precision-Recall AUC (PR-AUC) replaces FPR with Precision, focusing directly on minority class detection.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def create_imbalanced_classification_target(df, percentile=85):
    """
    Creates an imbalanced binary indicator:
    1 = High-Severity Catastrophic Claim (top 15%), 0 = Standard Claim (85%).
    """
    threshold = np.percentile(df['loss'], percentile)
    y_binary = (df['loss'] >= threshold).astype(int)
    print(f"Target derived at {percentile}th percentile (${threshold:,.2f}):")
    print(f"  Class 0 (Standard):      {(y_binary == 0).sum():,} rows ({100*(y_binary == 0).mean():.1f}%)")
    print(f"  Class 1 (High Severity): {(y_binary == 1).sum():,} rows ({100*(y_binary == 1).mean():.1f}%)")
    return y_binary, threshold

def plot_confusion_matrices(cm_dummy, cm_model, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Dummy Majority Classifier
    sns.heatmap(cm_dummy, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['Standard (0)', 'Severe (1)'],
                yticklabels=['Standard (0)', 'Severe (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title('Naive Majority Classifier\n(High Accuracy, 0% Recall Fallacy)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Actual Label', fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    
    # Trained Random Forest Classifier
    sns.heatmap(cm_model, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2,
                xticklabels=['Standard (0)', 'Severe (1)'],
                yticklabels=['Standard (0)', 'Severe (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    ax2.set_title('Random Forest Classifier\n(Balanced Diagnostic Detection)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Actual Label', fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.3] Confusion matrix plot saved to: {save_path}")

def plot_roc_and_pr_curves(y_test, y_probs_rf, y_probs_log, baseline_rate, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. ROC-AUC Curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
    auc_rf = roc_auc_score(y_test, y_probs_rf)
    
    fpr_log, tpr_log, _ = roc_curve(y_test, y_probs_log)
    auc_log = roc_auc_score(y_test, y_probs_log)
    
    ax1.plot(fpr_rf, tpr_rf, label=f'Random Forest (ROC-AUC = {auc_rf:.3f})', color='#16a34a', lw=2.5)
    ax1.plot(fpr_log, tpr_log, label=f'Logistic Reg (ROC-AUC = {auc_log:.3f})', color='#2563eb', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.500)', alpha=0.6)
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate / Recall (TPR)', fontsize=11, fontweight='bold')
    ax1.set_title('ROC Curves (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. PR-AUC Curves
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_probs_rf)
    pr_auc_rf = average_precision_score(y_test, y_probs_rf)
    
    prec_log, rec_log, _ = precision_recall_curve(y_test, y_probs_log)
    pr_auc_log = average_precision_score(y_test, y_probs_log)
    
    ax2.plot(rec_rf, prec_rf, label=f'Random Forest (PR-AUC = {pr_auc_rf:.3f})', color='#16a34a', lw=2.5)
    ax2.plot(rec_log, prec_log, label=f'Logistic Reg (PR-AUC = {pr_auc_log:.3f})', color='#2563eb', lw=2)
    ax2.axhline(baseline_rate, color='r', linestyle='--', label=f'No-Skill Baseline ({baseline_rate:.2f})')
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curves\n(Superior Metric for Severe Class Imbalance)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[M5.3] ROC and PR curves saved to: {save_path}")

def main():
    print("=== M5.3: Classification Metrics & Class Imbalance ===")
    sample_size = 8000
    random_seed = 42
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    y_binary, threshold = create_imbalanced_classification_target(df, percentile=85)
    X = df.drop(columns=['id', 'loss'])
    
    preprocessor = create_claimwise_preprocessor(df)
    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_binary, test_size=0.25, random_state=random_seed, stratify=y_binary
    )
    
    # 1. Dummy Classifier (Predict Majority Class) - Demonstrates when accuracy lies!
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_prec = precision_score(y_test, dummy_pred, zero_division=0)
    dummy_rec = recall_score(y_test, dummy_pred, zero_division=0)
    dummy_f1 = f1_score(y_test, dummy_pred, zero_division=0)
    
    print("\n--- Model 1: Naive Majority Dummy Classifier ---")
    print(f"  Accuracy:  {dummy_acc:.4f}  <-- Accuracy lies! Looks impressive at 85%!")
    print(f"  Precision: {dummy_prec:.4f}")
    print(f"  Recall:    {dummy_rec:.4f}  <-- Caught 0% of severe claims!")
    print(f"  F1-Score:  {dummy_f1:.4f}")
    
    # 2. Logistic Regression
    log_reg = LogisticRegression(class_weight='balanced', max_iter=500, random_state=random_seed)
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
    log_probs = log_reg.predict_proba(X_test)[:, 1]
    
    # 3. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=random_seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_roc = roc_auc_score(y_test, rf_probs)
    rf_pr_auc = average_precision_score(y_test, rf_probs)
    
    print("\n--- Model 2: Balanced Random Forest Classifier ---")
    print(f"  Accuracy:  {rf_acc:.4f}")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_rec:.4f}  <-- Successfully identifies catastrophic claims")
    print(f"  F1-Score:  {rf_f1:.4f}")
    print(f"  ROC-AUC:   {rf_roc:.4f}")
    print(f"  PR-AUC:    {rf_pr_auc:.4f}")
    
    # Confusion Matrices
    cm_dummy = confusion_matrix(y_test, dummy_pred)
    cm_rf = confusion_matrix(y_test, rf_pred)
    
    plot_cm_path = os.path.join(Config.PLOT_DIR, "confusion_matrix.png")
    plot_confusion_matrices(cm_dummy, cm_rf, plot_cm_path)
    
    plot_curves_path = os.path.join(Config.PLOT_DIR, "classification_roc_pr_curves.png")
    baseline_rate = y_test.mean()
    plot_roc_and_pr_curves(y_test, rf_probs, log_probs, baseline_rate, plot_curves_path)
    
    print("M5.3 completed successfully.")

if __name__ == "__main__":
    main()
