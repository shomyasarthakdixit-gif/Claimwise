import os
import joblib
import pandas as pd
from flask import current_app
from config import Config

# Supported models explicitly registered to prevent arbitrary pickle loading
SUPPORTED_MODELS = {
    'gradient_boosting_final': 'claimwise_final_model.pkl',
    'random_forest': 'random_forest.pkl',
    'linear_regression': 'linear_regression.pkl'
}

# In-process cache for models to avoid loading on every request
_model_cache = {}

def load_model(model_id):
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_id} is not supported.")
        
    if model_id in _model_cache:
        return _model_cache[model_id]
        
    model_path = os.path.join(Config.OUTPUT_DIR, 'models', SUPPORTED_MODELS[model_id])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found for {model_id}.")
        
    # Pickles are trusted local application artifacts
    model = joblib.load(model_path)
    _model_cache[model_id] = model
    return model

def get_model_features(model_id):
    """
    Extracts expected features from the sklearn Pipeline.
    RandomForest and LinearRegression pipelines expect exactly 130 features.
    """
    model = load_model(model_id)
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    # Fallback to the known 130 features for Allstate
    cats = [f'cat{i}' for i in range(1, 117)]
    conts = [f'cont{i}' for i in range(1, 15)]
    return cats + conts

def validate_input(data, feature_names):
    """
    Validates single prediction input dictionary against required features.
    Returns a pandas DataFrame ready for the pipeline.
    """
    # Check for target leakage
    if 'loss' in data:
        raise ValueError("Target leakage detected: 'loss' cannot be used as an input feature.")
        
    missing = [f for f in feature_names if f not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing[:5]}... ({len(missing)} total missing)")
        
    # Convert data into appropriate types
    processed_data = {}
    for f in feature_names:
        val = data[f]
        if f.startswith('cont'):
            try:
                processed_data[f] = [float(val)]
            except ValueError:
                raise ValueError(f"Feature {f} must be numeric. Received: {val}")
        else:
            processed_data[f] = [str(val)]
            
    return pd.DataFrame(processed_data)

def predict_single(model_id, input_data):
    """
    Executes a single prediction safely using the requested model.
    """
    model = load_model(model_id)
    features = get_model_features(model_id)
    
    # Validates and creates a 1-row DataFrame
    df = validate_input(input_data, features)
    
    # The existing pipeline handles missing values (imputation), scaling, and one-hot encoding internally.
    try:
        prediction = model.predict(df)[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
        
    return prediction

def predict_batch(model_id, df):
    """
    Executes batch prediction on a pandas DataFrame securely.
    """
    if 'loss' in df.columns:
        df = df.drop(columns=['loss'])
        
    model = load_model(model_id)
    features = get_model_features(model_id)
    
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Batch data is missing required features: {missing[:5]}... ({len(missing)} total missing)")
        
    # Ensure correct order and subset
    X = df[features]
    
    try:
        predictions = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Batch prediction failed: {str(e)}")
        
    return predictions
