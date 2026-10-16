import os
import json
from utils.data_loader import ingest_and_clean_csv
from config import Config

def main():
    print("Starting M1 Data Ingestion...")
    
    # 1. Ingest Train Data
    train_path = os.path.join(Config.RAW_DATA_DIR, 'train.csv')
    print(f"Ingesting {train_path}...")
    train_df, train_stats = ingest_and_clean_csv(
        filepath=train_path, 
        expected_cols=132, 
        quarantine_dir=os.path.join(Config.INTERIM_DATA_DIR, 'quarantine')
    )
    
    # Save processed train data
    processed_train_path = os.path.join(Config.PROCESSED_DATA_DIR, 'train.csv')
    train_df.to_csv(processed_train_path, index=False)
    print(f"Saved processed train data to {processed_train_path}")

    # 2. Ingest Test Data
    test_path = os.path.join(Config.RAW_DATA_DIR, 'test.csv')
    print(f"Ingesting {test_path}...")
    test_df, test_stats = ingest_and_clean_csv(
        filepath=test_path, 
        expected_cols=131, 
        quarantine_dir=os.path.join(Config.INTERIM_DATA_DIR, 'quarantine')
    )
    
    processed_test_path = os.path.join(Config.PROCESSED_DATA_DIR, 'test.csv')
    test_df.to_csv(processed_test_path, index=False)
    print(f"Saved processed test data to {processed_test_path}")

    # 3. Ingest Sample Submission Data
    sample_sub_path = os.path.join(Config.RAW_DATA_DIR, 'sample_submission.csv')
    print(f"Ingesting {sample_sub_path}...")
    sample_sub_df, sample_sub_stats = ingest_and_clean_csv(
        filepath=sample_sub_path, 
        expected_cols=2, 
        quarantine_dir=os.path.join(Config.INTERIM_DATA_DIR, 'quarantine')
    )
    
    processed_sample_sub_path = os.path.join(Config.PROCESSED_DATA_DIR, 'sample_submission.csv')
    sample_sub_df.to_csv(processed_sample_sub_path, index=False)
    print(f"Saved processed sample submission data to {processed_sample_sub_path}")

    # Combine stats and save to interim for the validation script to pick up
    all_stats = {
        "train": train_stats,
        "test": test_stats,
        "sample_submission": sample_sub_stats
    }
    
    stats_path = os.path.join(Config.INTERIM_DATA_DIR, 'ingestion_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
        
    print(f"Ingestion complete. Stats saved to {stats_path}")

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.INTERIM_DATA_DIR, 'quarantine'), exist_ok=True)
    main()
