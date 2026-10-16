import os
import math
import pandas as pd
from config import Config
from flask import abort

# Simple in-process cache to avoid reloading 188k rows repeatedly.
# Cache Documentation:
# - First-load behavior: Reads the entire dataset into memory (approx 130MB for 188k rows).
# - Memory usage: Small datasets take very little, largest takes ~130MB.
# - Use case: This is an in-process development cache to provide fast server-side pagination without re-reading the 188k-row CSV on every request.
# - Limitations: In a production environment with multiple workers, a centralized cache (like Redis) or database offloading would be needed. The app should not necessarily cache every dataset simultaneously if memory is constrained.
_dataset_cache = {}

ALLOWED_DATASETS = {
    'train': 'train.csv',
    'test': 'test.csv',
    'sample_submission': 'sample_submission.csv'
}

def _get_dataframe(dataset_name):
    if dataset_name not in ALLOWED_DATASETS:
        abort(400, "Invalid dataset requested.")
        
    if dataset_name in _dataset_cache:
        return _dataset_cache[dataset_name]
        
    file_path = os.path.join(Config.PROCESSED_DATA_DIR, ALLOWED_DATASETS[dataset_name])
    if not os.path.exists(file_path):
        abort(404, "Dataset file not found.")
        
    df = pd.read_csv(file_path)
    _dataset_cache[dataset_name] = df
    return df

def get_dataset_info(dataset_name):
    df = _get_dataframe(dataset_name)
    
    # Calculate basic missing value count globally
    missing_count = int(df.isnull().sum().sum())
    
    return {
        'name': dataset_name.capitalize(),
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': missing_count,
    }

def get_dataset_schema(dataset_name):
    df = _get_dataframe(dataset_name)
    schema = []
    for col in df.columns:
        schema.append({
            'name': col,
            'type': str(df[col].dtype),
            'missing': int(df[col].isnull().sum())
        })
    return schema

def get_dataset_statistics(dataset_name):
    df = _get_dataframe(dataset_name)
    # Calculate stats only for a small subset or on demand to avoid expensive computation.
    # For now, we will return basic desc for first few numeric columns.
    numeric_cols = df.select_dtypes(include=['number']).columns[:5] 
    stats = df[numeric_cols].describe().round(2).to_dict()
    return stats

def get_dataset_page(dataset_name, page, page_size, search=None, sort_col=None, sort_order='asc', filter_column=None, filter_operator=None, filter_value=None):
    df = _get_dataframe(dataset_name)
    
    # Apply column filtering
    if filter_column and filter_column in df.columns and filter_value:
        try:
            is_numeric = pd.api.types.is_numeric_dtype(df[filter_column])
            if filter_operator == 'equals':
                if is_numeric:
                    df = df[df[filter_column] == float(filter_value)]
                else:
                    df = df[df[filter_column].astype(str) == str(filter_value)]
            elif filter_operator == 'contains' and not is_numeric:
                df = df[df[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)]
            elif filter_operator == 'gt' and is_numeric:
                df = df[df[filter_column] > float(filter_value)]
            elif filter_operator == 'gte' and is_numeric:
                df = df[df[filter_column] >= float(filter_value)]
            elif filter_operator == 'lt' and is_numeric:
                df = df[df[filter_column] < float(filter_value)]
            elif filter_operator == 'lte' and is_numeric:
                df = df[df[filter_column] <= float(filter_value)]
        except ValueError:
            # If parsing filter_value fails (e.g. passing 'abc' to a float), silently ignore the filter or return empty.
            # Returning empty DataFrame safely:
            df = df.iloc[0:0]

    # Apply global search if requested
    if search:
        search_str = str(search).lower()
        # To avoid massive overhead, only search first 20 cols or specific subset.
        search_cols = df.columns[:20]
        mask = df[search_cols].astype(str).apply(lambda x: x.str.lower().str.contains(search_str)).any(axis=1)
        df = df[mask]
        
    # Apply sorting
    if sort_col and sort_col in df.columns:
        ascending = sort_order == 'asc'
        df = df.sort_values(by=sort_col, ascending=ascending)
        
    total_rows = len(df)
    total_pages = math.ceil(total_rows / page_size) if total_rows > 0 else 1
    
    # Safe bounds
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    page_data = df.iloc[start_idx:end_idx]
    
    return {
        'data': page_data.to_dict(orient='records'),
        'columns': list(page_data.columns),
        'total_rows': total_rows,
        'total_pages': total_pages,
        'current_page': page
    }

def get_dataset_filepath(dataset_name):
    if dataset_name not in ALLOWED_DATASETS:
        return None
    return os.path.join(Config.PROCESSED_DATA_DIR, ALLOWED_DATASETS[dataset_name])
