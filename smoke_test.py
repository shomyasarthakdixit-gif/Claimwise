import os
import pandas as pd
from utils.data_loader import load_processed_data
from src.web.services.prediction_service import predict_single, predict_batch
from config import Config

def main():
    print("=== SMOKE TEST ===")
    
    # 1. Load one valid feature row
    print("Loading test data...")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "test.csv"))
    single_row = df.iloc[0].to_dict()
    if 'loss' in single_row:
        del single_row['loss']
        
    model_id = 'gradient_boosting_final'
    
    # 2. Run single prediction
    print(f"Running single prediction for model {model_id}...")
    pred = predict_single(model_id, single_row)
    print(f"Single prediction result: {pred} (type: {type(pred)})")
    assert isinstance(pred, (float, int)), "Prediction is not numeric"
    
    # 3. Run batch prediction on a small CSV
    small_df = df.head(10).copy()
    if 'loss' in small_df.columns:
        small_df = small_df.drop(columns=['loss'])
        
    print(f"Running batch prediction for model {model_id} (10 rows)...")
    batch_preds = predict_batch(model_id, small_df)
    print(f"Batch prediction returned {len(batch_preds)} results.")
    print("First 3 predictions:", batch_preds[:3])
    assert len(batch_preds) == 10, "Batch prediction count mismatch"
    
    print("Smoke test passed.")

if __name__ == "__main__":
    main()
