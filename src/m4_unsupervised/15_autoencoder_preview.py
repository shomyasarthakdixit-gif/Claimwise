import os
from sklearn.neural_network import MLPRegressor
from utils.data_loader import load_processed_data
from utils.preprocessing import create_claimwise_preprocessor
from config import Config

def main():
    print("M4: Autoencoder Conceptual Preview")
    print("Concept: Training a lightweight neural network (MLP) to reconstruct its own inputs.")
    sample_size = 1000
    random_seed = 42
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(sample_size, random_state=random_seed)
    
    X = df.drop(columns=['id', 'loss'])
    preprocessor = create_claimwise_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    # We use MLPRegressor where X is both the input and the target
    # This acts as an autoencoder conceptually. 
    # Hidden layer sizes define the bottleneck.
    print(f"Original Feature Dimensions: {X_processed.shape[1]}")
    autoencoder = MLPRegressor(hidden_layer_sizes=(32, 16, 32), max_iter=200, random_state=random_seed)
    autoencoder.fit(X_processed, X_processed)
    
    score = autoencoder.score(X_processed, X_processed)
    print(f"Autoencoder self-reconstruction R2: {score:.4f} (Lightweight scikit-learn preview without TF/PyTorch)")

if __name__ == "__main__":
    main()
