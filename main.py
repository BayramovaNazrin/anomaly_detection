from src.data_loader import load_and_explore_data
from src.models.classical import train_random_forest, train_svm
from src.models.graph_models import train_node2vec_rf, train_graphsage

def main():
    # --- Paths ---
    features_path = "/content/drive/MyDrive/anomaly/elliptic_txs_features.csv"
    edges_path = "/content/drive/MyDrive/anomaly/elliptic_txs_edgelist.csv"
    classes_path = "/content/drive/MyDrive/anomaly/elliptic_txs_classes.csv"

    # --- Load Data ---
    features, edges, classes, merged_df = load_and_explore_data(
        features_path=features_path,
        edges_path=edges_path,
        classes_path=classes_path
    )

    # --- Classical models ---
    train_random_forest(merged_df)
    train_svm(merged_df)

    # --- Graph models ---
    train_node2vec_rf(features_path, edges_path, classes_path)
    train_graphsage(features_path, edges_path, classes_path)

if __name__ == "__main__":
    main()
