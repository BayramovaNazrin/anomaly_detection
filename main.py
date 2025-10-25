import os
import argparse
import pandas as pd

from src.data_loader import load_and_explore_data
from src.models.classical import train_random_forest, train_svm
from src.models.graph import train_node2vec_rf, train_graphsage
from src.visualization.eda import run_eda
from src.visualization.evaluation import plot_model_comparison, plot_bar_comparison
from src.models.feature_importance import (
    permutation_importance, plot_feature_importances,
    create_top_feature_graph, retrain_with_top_features
)
from src.visualization.feature_importance_plots import (
    plot_rf_feature_importance,
    plot_graphsage_importance,
    plot_feature_comparison
)
from src.utils.helpers import setup_directories, set_seed, get_data_path
from src.utils.helpers import safe_report, evaluate_with_threshold


def main():
    # --- Command-line argument for data directory ---
    parser = argparse.ArgumentParser(description="Anomaly Detection for Illicit Bitcoin Transactions")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional path to the folder containing the Elliptic dataset."
    )
    args = parser.parse_args()

    # --- Setup ---
    setup_directories()
    set_seed(42)

    # --- Resolve data paths ---
    if args.data_dir:
        features_path = os.path.join(args.data_dir, "txs_features.csv")
        edges_path    = os.path.join(args.data_dir, "txs_edgelist.csv")
        classes_path  = os.path.join(args.data_dir, "txs_classes.csv")
    else:
        features_path = get_data_path("txs_features.csv")
        edges_path    = get_data_path("txs_edgelist.csv")
        classes_path  = get_data_path("txs_classes.csv")

    print(f"📂 Using data from:\n{os.path.dirname(features_path)}")

    # --- Load Data ---
   
    features, edges, classes, merged_df = load_and_explore_data(
    features_path,
    edges_path,
    classes_path
    )

    # --- EDA & Classical Models ---
    run_eda(merged_df)
    train_random_forest(merged_df)
    train_svm(merged_df)

    # --- Graph Models ---
    train_node2vec_rf(features_path, edges_path, classes_path)
    train_graphsage(features_path, edges_path, classes_path)

    # === Model Comparison (example values) ===
    results = [
        {"Model": "RandomForest", "Accuracy": 0.988, "Precision (Illicit)": 0.95, "Recall (Illicit)": 0.92, "F1 (Illicit)": 0.94},
        {"Model": "SVM",          "Accuracy": 0.983, "Precision (Illicit)": 0.93, "Recall (Illicit)": 0.89, "F1 (Illicit)": 0.91},
        {"Model": "Node2Vec+RF",  "Accuracy": 0.979, "Precision (Illicit)": 0.91, "Recall (Illicit)": 0.88, "F1 (Illicit)": 0.89},
        {"Model": "GraphSAGE",    "Accuracy": 0.981, "Precision (Illicit)": 0.92, "Recall (Illicit)": 0.90, "F1 (Illicit)": 0.91},
    ]
    df_results = pd.DataFrame(results)

    plot_model_comparison(df_results)
    plot_bar_comparison(df_results)

    # === Feature Importance Visualization (optional, only if data exists) ===
    try:
        rf_imp_df = pd.read_csv("artifacts/data/rf_feature_importance.csv")
        gs_imp_df = pd.read_csv("artifacts/data/graphsage_feature_importance.csv")

        plot_rf_feature_importance(rf_imp_df)
        plot_graphsage_importance(gs_imp_df)
        plot_feature_comparison(rf_imp_df, gs_imp_df)
    except FileNotFoundError:
        print("⚠️ Skipping feature importance plots (files not found).")

    print("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
