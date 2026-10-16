import os
import pandas as pd
import json

def ingest_and_clean_csv(filepath, expected_cols, quarantine_dir):
    """
    Reads a raw CSV file, attempts safe reconstruction of malformed lines,
    quarantines unrecoverable lines, and returns a clean DataFrame along with stats.
    """
    filename = os.path.basename(filepath)
    quarantine_filepath = os.path.join(quarantine_dir, f"quarantined_{filename}")
    
    valid_records = []
    repaired_records = []
    quarantined_records = []
    
    # We will read line by line to handle quoting issues manually
    with open(filepath, 'r', encoding='utf-8') as f:
        header_line = next(f).strip('\r\n')
        header = [h.strip() for h in header_line.split(',')]
        
        if len(header) != expected_cols:
            raise ValueError(f"{filename} header has {len(header)} columns, expected {expected_cols}.")
            
        for i, line in enumerate(f, start=2):
            raw_line = line.strip('\r\n')
            fields = raw_line.split(',')
            
            if len(fields) == expected_cols:
                # Valid record
                valid_records.append([f.strip() for f in fields])
            elif len(fields) == expected_cols + 1 and fields[-1].strip() in ('"', '""'):
                # Safe Reconstruction: A trailing comma and quote was added erroneously
                # Check if the first field is an ID (numeric)
                repaired_fields = [f.strip() for f in fields[:-1]]
                if repaired_fields[0].isdigit():
                    repaired_records.append({
                        "line": i,
                        "data": repaired_fields,
                        "reason": f"Dropped extraneous trailing quote/comma. Expected {expected_cols}, got {len(fields)}."
                    })
                    valid_records.append(repaired_fields)
                else:
                    quarantined_records.append({
                        "line": i,
                        "raw_data": raw_line,
                        "reason": f"Column count mismatch ({len(fields)}) and ID missing. Cannot safely reconstruct."
                    })
            else:
                # Ambiguous or Unrecoverable
                if raw_line.strip() in ('"', '""', ''):
                    reason = "Empty or isolated quote mark line."
                else:
                    reason = f"Column count mismatch. Expected {expected_cols}, got {len(fields)}."
                    
                quarantined_records.append({
                    "line": i,
                    "raw_data": raw_line,
                    "reason": reason
                })

    # Save quarantined records
    if quarantined_records:
        os.makedirs(quarantine_dir, exist_ok=True)
        with open(quarantine_filepath, 'w', encoding='utf-8') as qf:
            for q in quarantined_records:
                qf.write(f"Line {q['line']} | Reason: {q['reason']} | Raw: {q['raw_data']}\n")

    # Create DataFrame
    df = pd.DataFrame(valid_records, columns=header)
    
    # Cast ID to integer
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
    # Identify exact duplicates and separate them
    duplicate_mask = df.duplicated(keep='first')
    duplicates_df = df[duplicate_mask].copy()
    
    if not duplicates_df.empty:
        df = df[~duplicate_mask].copy()
        # Save duplicates to audit
        audit_dir = os.path.join(os.path.dirname(quarantine_dir), 'audit')
        os.makedirs(audit_dir, exist_ok=True)
        duplicates_filepath = os.path.join(audit_dir, f"reconstructed_duplicates_{filename}")
        duplicates_df.to_csv(duplicates_filepath, index=False)
        
    stats = {
        "source_file": filename,
        "expected_columns": expected_cols,
        "original_record_count": i - 1, # Excluding header
        "original_valid_records": len(df),
        "reconstructed_duplicate_records": len(duplicates_df),
        "quarantined_records": len(quarantined_records),
        "final_usable_records": len(df),
        "quarantine_reasons": [q['reason'] for q in quarantined_records],
    }
    
    return df, stats

def load_processed_data(filepath):
    """
    Utility function for subsequent M2-M4 modules to load clean data.
    """
    return pd.read_csv(filepath)
