import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from utils.data_loader import load_processed_data
from config import Config

def main():
    print("M2: Missing Value Handling Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(100, random_state=42)
    
    # Artificially inject some missing values for demonstration
    demo_df = df[['cont1', 'cont2']].copy()
    demo_df.iloc[0:10, demo_df.columns.get_loc('cont1')] = np.nan
    
    print(f"Missing before imputation:\n{demo_df.isnull().sum()}")
    
    # Missing Indicator
    indicator = MissingIndicator()
    mask = indicator.fit_transform(demo_df)
    print(f"Missing indicator shape: {mask.shape}")
    
    # Mean Imputation
    imputer = SimpleImputer(strategy='mean')
    imputed = imputer.fit_transform(demo_df)
    
    print(f"Missing after mean imputation: {pd.DataFrame(imputed).isnull().sum().sum()}")

if __name__ == "__main__":
    main()
