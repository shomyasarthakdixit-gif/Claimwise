import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def create_claimwise_preprocessor(df):
    """
    Creates a preprocessor for the ClaimWise dataset (M2-M4).
    """
    cat_cols = [c for c in df.columns if c.startswith('cat')]
    cont_cols = [c for c in df.columns if c.startswith('cont')]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cont_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
        
    return preprocessor

def get_train_test_splits(df, target_col='loss', test_size=0.2, random_state=42):
    X = df.drop(columns=['id', target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
