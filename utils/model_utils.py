import os
import joblib
from config import Config

def save_model(model, filename):
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    filepath = os.path.join(Config.MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename):
    filepath = os.path.join(Config.MODELS_DIR, filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None
