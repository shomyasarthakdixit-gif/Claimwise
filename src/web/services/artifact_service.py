import os
import json
import pandas as pd
from config import Config

def get_dashboard_summary():
    '''Reads existing artifacts to populate dashboard overview safely.'''
    summary = {
        'training_records': 188318, # From instructions/known clean dataset
        'test_records': 125546,
        'features': 130, # Categorical and continuous
        
        'm1_status': 'Complete',
        'm2_status': 'Complete',
        'm3_status': 'Complete',
        'm4_status': 'Complete',
        'm5_status': 'Complete',
        
        'models': [],
        'reports': [],
        'clusters': []
    }
    
    # Safely scan models directory
    if os.path.exists(Config.MODELS_DIR):
        summary['models'] = [f for f in os.listdir(Config.MODELS_DIR) if f.endswith('.pkl')]
        
    # Safely scan reports
    if os.path.exists(Config.REPORT_DIR):
        summary['reports'] = [f for f in os.listdir(Config.REPORT_DIR) if f.endswith('.txt') or f.endswith('.json')]
        
    # Safely scan clustering
    if os.path.exists(Config.CLUSTERING_DIR):
        summary['clusters'] = [f for f in os.listdir(Config.CLUSTERING_DIR) if f.endswith('.csv')]
        
    return summary

ALLOWED_PLOTS = {
    # M1 EDA & Lifecycle
    "loss_distribution.png": "loss_distribution.png",
    "feature_distribution.png": "feature_distribution.png",
    "kde_density.png": "kde_density.png",
    "pairplot.png": "pairplot.png",
    # M2 Linear Models
    "actual_vs_predicted.png": "actual_vs_predicted.png",
    "residual_plot.png": "residual_plot.png",
    # M3 Tree Models
    "feature_importance.png": "feature_importance.png",
    "model_comparison.png": "model_comparison.png",
    # M4 Clustering & Unsupervised
    "pca_clusters.png": "pca_clusters.png",
    "elbow_curve.png": "elbow_curve.png",
    "silhouette_scores.png": "silhouette_scores.png",
    "dbscan_clusters.png": "dbscan_clusters.png",
    "dendrogram.png": "dendrogram.png",
    "tsne_visualization.png": "tsne_visualization.png",
    # M5 Model Evaluation, Selection & Calibration
    "train_val_test_split.png": "train_val_test_split.png",
    "nested_cv_results.png": "nested_cv_results.png",
    "confusion_matrix.png": "confusion_matrix.png",
    "classification_roc_pr_curves.png": "classification_roc_pr_curves.png",
    "regression_metrics_distribution.png": "regression_metrics_distribution.png",
    "probability_calibration_curves.png": "probability_calibration_curves.png",
    "hyperparameter_search_comparison.png": "hyperparameter_search_comparison.png",
    "learning_and_validation_curves.png": "learning_and_validation_curves.png",
    "statistical_significance_bootstrap.png": "statistical_significance_bootstrap.png",
    "m5_model_evaluation_dashboard.png": "m5_model_evaluation_dashboard.png"
}

ALLOWED_REPORTS = {
    "data_quality_report.json": "data_quality_report.json",
    "m5_evaluation_metrics.json": "m5_evaluation_metrics.json",
    "final_model_selection.json": "final_model_selection.json"
}

ALLOWED_EVALUATION = {
    "model_comparison.csv": "model_comparison.csv",
    "final_model_selection.csv": "final_model_selection.csv",
    "m5_regression_summary.csv": "m5_regression_summary.csv"
}

def get_plot_path(filename):
    if filename not in ALLOWED_PLOTS:
        # Also check if it's a valid png file located directly in PLOT_DIR without path traversal
        if ('..' not in filename) and (not filename.startswith('/')) and (not filename.startswith('\\')) and filename.endswith('.png'):
            direct_path = os.path.join(Config.PLOT_DIR, filename)
            if os.path.exists(direct_path):
                return direct_path
        return None
    path = os.path.join(Config.PLOT_DIR, ALLOWED_PLOTS[filename])
    if os.path.exists(path):
        return path
    return None

def get_json_report(filename):
    if filename not in ALLOWED_REPORTS:
        return None
    
    # Check REPORT_DIR first
    path = os.path.join(Config.REPORT_DIR, ALLOWED_REPORTS[filename])
    if not os.path.exists(path):
        # Also check EVALUATION_DIR for evaluation JSON reports
        path = os.path.join(Config.EVALUATION_DIR, ALLOWED_REPORTS[filename])
        
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def get_csv_data(directory, filename):
    # Depending on the directory, check allowlist
    path = None
    if directory == 'evaluation' and filename in ALLOWED_EVALUATION:
        path = os.path.join(Config.OUTPUT_DIR, 'evaluation', ALLOWED_EVALUATION[filename])
    elif directory == 'models' and filename in ['feature_importances.csv', 'linear_regression_coefficients.csv']:
        path = os.path.join(Config.MODELS_DIR, filename)
    elif directory == 'clustering' and filename in ['anomaly_results.csv', 'cluster_results.csv', 'dbscan_results.csv', 'elbow_results.csv', 'pca_results.csv', 'silhouette_results.csv']:
        path = os.path.join(Config.CLUSTERING_DIR, filename)
        
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path).to_dict(orient='records')
        except Exception:
            return None
    return None
