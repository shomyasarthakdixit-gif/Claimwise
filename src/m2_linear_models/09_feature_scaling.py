import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.data_loader import load_processed_data
from config import Config

def main():
    print("M2: Feature Scaling Demonstration")
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(100, random_state=42)
    
    cont_cols = [c for c in df.columns if c.startswith('cont')]
    data = df[cont_cols].fillna(df[cont_cols].median())
    
    scaler_std = StandardScaler()
    std_scaled = scaler_std.fit_transform(data)
    
    scaler_minmax = MinMaxScaler()
    minmax_scaled = scaler_minmax.fit_transform(data)
    
    print(f"Original cont1 mean: {data['cont1'].mean():.4f}")
    print(f"StandardScaled cont1 mean: {std_scaled[:, 0].mean():.4f}")
    print(f"MinMaxScaled cont1 min: {minmax_scaled[:, 0].min():.4f}, max: {minmax_scaled[:, 0].max():.4f}")

if __name__ == "__main__":
    main()
