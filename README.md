# ClaimWise: Intelligent Insurance Claims Analytics & Prediction Platform

## 1. Project Overview
ClaimWise is a comprehensive machine learning platform designed to predict claim loss severity using the Allstate Claims Severity dataset. It demonstrates a complete end-to-end ML lifecycle wrapped in a professional, secure Flask dashboard.

**Important Note on Fraud:** The Allstate Claims Severity dataset does NOT contain a labelled fraud target. ClaimWise predicts claim loss severity, NOT fraud probability. Any classification targets used during the educational modules are derived for demonstration purposes only.

## 2. Scope & Implementation Status
ClaimWise implements the following modules (M1–M4):
- **M1 - ML Lifecycle**: End-to-end data processing, feature engineering, and robust ML pipelines.
- **M2 - Linear Models**: Educational demonstrations and baseline Ridge Regression.
- **M3 - Tree Models**: Random Forest, Feature Importance, and Tree Ensembles.
- **M4 - Unsupervised Learning**: Clustering (KMeans, DBSCAN, Hierarchical) and Dimensionality Reduction (PCA).

**Future Scope (Not Implemented):**
- **M5 - Evaluation & Methodology** (Cross-validation, advanced calibration)
- **M6 - ML Engineering** (MLflow, DVC, automated retraining)

## 3. Architecture & Folder Structure
```text
ClaimWise/
├── app.py                  # Application entry point
├── config.py               # Central configuration
├── data/                   # Raw and processed datasets
├── src/
│   ├── m1_lifecycle/       # M1 implementation scripts
│   ├── m2_linear_models/   # M2 implementation scripts
│   ├── m3_tree_models/     # M3 implementation scripts
│   ├── m4_unsupervised/    # M4 implementation scripts
│   └── web/                # Flask web application logic
├── templates/              # HTML views (Jinja2)
├── static/                 # CSS/JS and assets
├── tests/                  # Pytest verification suite
└── Output/                 # Generated ML artifacts (models, plots, reports)
```

## 4. Installation & Running the Application
1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Access the application:** Open `http://localhost:5000` in a browser.

## 5. Web Application Features
The web layer provides secure, read-only analytics views over the M1-M4 artifacts without regenerating models unnecessarily:

- **Authentication**: Secure login and registration required for all dashboard views.
- **Dashboard Overview**: Tracks dataset status, available models, and completed module metrics.
- **Dataset Explorer**: Server-side paginated and filterable interface for training and test records.
- **Prediction Interface**:
  - **Single Prediction**: Predict loss severity using the 130-feature schema. Preprocessing is securely handled by the scikit-learn pipeline.
  - **Batch Prediction**: Upload a CSV (up to 50,000 rows) for bulk prediction. The `loss` column is safely ignored if present.
- **ML Modules (M1–M4) & Reports**: Deep dives into the architecture, educational techniques, visual plots, and structural reports (e.g., Data Quality).

## 6. Testing
ClaimWise employs a comprehensive Pytest suite covering web routing, prediction logic, authentication, error handling, and security (path traversal prevention).
```bash
python -m pytest tests/web/
```

## 7. Known Limitations
- The dashboard is a presentation layer. It will not retrain models. If you update raw data, you must manually execute the M1-M4 scripts to regenerate the `Output/` artifacts.
- Advanced boosting models (XGBoost/LightGBM) and UMAP may be marked as "Unavailable/Skipped" if the respective optional dependencies are not installed in the environment.
