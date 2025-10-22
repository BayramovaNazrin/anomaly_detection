from src.data_loader import load_and_explore_data
from src.models.classical import train_random_forest, train_svm

# === Load Data ===
features, edges, classes, merged_df = load_and_explore_data(
    features_path = "/content/drive/MyDrive/anomaly/elliptic_txs_features.csv",
    edges_path = "/content/drive/MyDrive/anomaly/elliptic_txs_edgelist.csv",
    classes_path = "/content/drive/MyDrive/anomaly/elliptic_txs_classes.csv"
)

# === Run Classical Models ===
train_random_forest(merged_df)
train_svm(merged_df)
