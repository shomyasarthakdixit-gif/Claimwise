import os
from sklearn.tree import DecisionTreeRegressor
from utils.data_loader import load_processed_data
from utils.preprocessing import get_train_test_splits, create_claimwise_preprocessor
from config import Config

def main():
    print("M3: Tree Pruning Demonstration")
    print("Concepts: Pre-pruning (max_depth, min_samples) & Post-pruning (CCP).\n")
    
    df = load_processed_data(os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")).sample(2000, random_state=42)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    preprocessor = create_claimwise_preprocessor(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("1. Unpruned Tree")
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train_processed, y_train)
    print(f"Nodes: {tree.tree_.node_count}, Test R2: {tree.score(X_test_processed, y_test):.4f}")
    
    print("\n2. Pre-pruning (max_depth=5, min_samples_leaf=10)")
    tree_pre = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    tree_pre.fit(X_train_processed, y_train)
    print(f"Nodes: {tree_pre.tree_.node_count}, Test R2: {tree_pre.score(X_test_processed, y_test):.4f}")
    
    print("\n3. Post-pruning (Cost-Complexity Pruning)")
    path = tree.cost_complexity_pruning_path(X_train_processed, y_train)
    ccp_alphas = path.ccp_alphas
    # Just grab a sensible alpha from the middle
    alpha = ccp_alphas[len(ccp_alphas) // 2]
    tree_post = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
    tree_post.fit(X_train_processed, y_train)
    print(f"Using ccp_alpha={alpha:.4f}")
    print(f"Nodes: {tree_post.tree_.node_count}, Test R2: {tree_post.score(X_test_processed, y_test):.4f}")

if __name__ == "__main__":
    main()
