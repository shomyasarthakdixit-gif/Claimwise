import os
import json
import pandas as pd
from config import Config
from utils.data_loader import load_processed_data

def generate_validation_report(df, filename, expected_cols, stats):
    """
    Validates the dataframe and merges ingestion stats into a final report dictionary.
    """
    actual_cols = len(df.columns)
    validation_status = "PASS" if actual_cols == expected_cols else "FAIL"
    
    # Calculate missing values
    missing_counts = df.isnull().sum()
    missing_summary = missing_counts[missing_counts > 0].to_dict()
    total_missing = int(missing_counts.sum())
    
    # Calculate duplicates
    duplicate_count = int(df.duplicated().sum())
    if 'id' in df.columns:
        duplicate_ids = int(df['id'].duplicated().sum())
    else:
        duplicate_ids = 0
        
    report = {
        "source_file": stats.get("source_file", filename),
        "expected_columns": expected_cols,
        "detected_columns": actual_cols,
        "original_record_count": stats.get("original_record_count", len(df)),
        "original_valid_records": stats.get("original_valid_records", len(df)),
        "reconstructed_duplicate_records": stats.get("reconstructed_duplicate_records", 0),
        "quarantined_records": stats.get("quarantined_records", 0),
        "quarantine_reasons": stats.get("quarantine_reasons", []),
        "final_usable_records": stats.get("final_usable_records", len(df)),
        "missing_value_summary": {
            "total_missing_values": total_missing,
            "columns_with_missing": missing_summary
        },
        "duplicate_summary": {
            "duplicate_rows": duplicate_count,
            "duplicate_ids": duplicate_ids
        },
        "validation_status": validation_status
    }
    
    return report

def write_text_report(report_dict, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write("                 DATA QUALITY REPORT                        \n")
        f.write("============================================================\n\n")
        
        for file_key, r in report_dict.items():
            f.write(f"--- {r['source_file'].upper()} ---\n")
            f.write(f"Validation Status:                  {r['validation_status']}\n")
            f.write(f"Expected Columns:                   {r['expected_columns']}\n")
            f.write(f"Detected Columns:                   {r['detected_columns']}\n")
            f.write(f"Original Records (incl. corrupted): {r['original_record_count']}\n")
            f.write(f"Original Valid Records:             {r['original_valid_records']}\n")
            f.write(f"Reconstructed Duplicate Records:    {r['reconstructed_duplicate_records']}\n")
            f.write(f"Quarantined Records:                {r['quarantined_records']}\n")
            f.write(f"Final Modeling Records:             {r['final_usable_records']}\n")
            
            f.write("Missing Value Summary:\n")
            f.write(f"  Total Missing:        {r['missing_value_summary']['total_missing_values']}\n")
            if r['missing_value_summary']['total_missing_values'] > 0:
                for col, count in r['missing_value_summary']['columns_with_missing'].items():
                    f.write(f"  - {col}: {count}\n")
                    
            f.write("Duplicate Summary:\n")
            f.write(f"  Duplicate Rows:       {r['duplicate_summary']['duplicate_rows']}\n")
            f.write(f"  Duplicate IDs:        {r['duplicate_summary']['duplicate_ids']}\n")
            
            if r['quarantined_records'] > 0:
                f.write("Quarantine Reasons:\n")
                # Summarize reasons
                reasons = {}
                for reason in r['quarantine_reasons']:
                    reasons[reason] = reasons.get(reason, 0) + 1
                for reason, count in reasons.items():
                    f.write(f"  - {count} record(s): {reason}\n")
            
            f.write("\n")

def main():
    print("Starting M1 Data Validation...")
    os.makedirs(Config.REPORT_DIR, exist_ok=True)
    
    # Load ingestion stats
    stats_path = os.path.join(Config.INTERIM_DATA_DIR, 'ingestion_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            ingestion_stats = json.load(f)
    else:
        print("Warning: ingestion_stats.json not found. Did you run 01_data_ingestion.py?")
        ingestion_stats = {"train": {}, "test": {}, "sample_submission": {}}
        
    full_report = {}
    
    # 1. Validate Train
    print("Validating Train Data...")
    train_df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, 'train.csv'))
    full_report['train'] = generate_validation_report(
        train_df, 'train.csv', 132, ingestion_stats.get('train', {})
    )
    
    # 2. Validate Test
    print("Validating Test Data...")
    test_df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, 'test.csv'))
    full_report['test'] = generate_validation_report(
        test_df, 'test.csv', 131, ingestion_stats.get('test', {})
    )
    
    # 3. Validate Sample Submission
    print("Validating Sample Submission Data...")
    sample_sub_df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, 'sample_submission.csv'))
    full_report['sample_submission'] = generate_validation_report(
        sample_sub_df, 'sample_submission.csv', 2, ingestion_stats.get('sample_submission', {})
    )
    
    # Save JSON Report
    json_report_path = os.path.join(Config.REPORT_DIR, 'data_quality_report.json')
    with open(json_report_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    print(f"Saved JSON report to {json_report_path}")
        
    # Save TXT Report
    txt_report_path = os.path.join(Config.REPORT_DIR, 'data_quality_report.txt')
    write_text_report(full_report, txt_report_path)
    print(f"Saved TXT report to {txt_report_path}")
    
    print("Data Validation Complete.")

if __name__ == '__main__':
    main()
