import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils.data_loader import load_processed_data
from config import Config

def main():
    print("M2: Categorical Encoding Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(100, random_state=42)
    
    cat_cols = ['cat1', 'cat2', 'cat3']
    data = df[cat_cols].fillna('missing')
    
    ohe = OneHotEncoder(sparse_output=False)
    ohe_encoded = ohe.fit_transform(data)
    
    ord_enc = OrdinalEncoder()
    ord_encoded = ord_enc.fit_transform(data)
    
    print(f"Original categories for cat1: {df['cat1'].unique()}")
    print(f"OneHotEncoded shape: {ohe_encoded.shape}")
    print(f"OrdinalEncoded shape: {ord_encoded.shape}")
    print("Target/Embedding encoding conceptual demonstration omitted.")

if __name__ == "__main__":
    main()
