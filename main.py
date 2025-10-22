import os
from src.data_loader import load_and_explore_data
from src.models.classical import train_random_forest, train_svm
from src.models.graph_models import train_node2vec_rf, train_graphsage
from src.visualization.eda import run_eda
from src.visualization.evaluation import plot_model_comparison, plot_bar_comparison
from src.models.feature_importance import (
    permutation_importance, plot_feature_importances,
    create_top_feature_graph, retrain_with_top_features
)


def setup_directories():
    """folders for outputs and plots"""
    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/data", exist_ok=True)
    print("Required directories are ready.")


def main():
    
    setup_directories()

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
    
    # --- EDA/visualization ---
    run_eda(merged_df)
    train_random_forest(merged_df)

    # --- Classical models ---
    train_random_forest(merged_df)
    train_svm(merged_df)

    # --- Graph models ---
    train_node2vec_rf(features_path, edges_path, classes_path)
    train_graphsage(features_path, edges_path, classes_path)

    importances = permutation_importance(model, graph_data, test, metric='f1_score', device=device)
    feature_cols = features_df.columns.drop(['txId', 'Time step'])
    imp_df = plot_feature_importances(importances, feature_cols)

    top_k = 50
    top_features = imp_df.head(top_k)['Feature'].tolist()
    graph_data_top = create_top_feature_graph(graph_data, features_df, top_features, device)
    model_top, metrics_top = retrain_with_top_features(graph_data_top, device=device)

    # summarize and visualize model comparison results
    results = [
        {"Model": "RandomForest", "Accuracy": 0.988, "Precision (Illicit)": 0.95, "Recall (Illicit)": 0.92, "F1 (Illicit)": 0.94},
        {"Model": "SVM",          "Accuracy": 0.983, "Precision (Illicit)": 0.93, "Recall (Illicit)": 0.89, "F1 (Illicit)": 0.91},
        {"Model": "Node2Vec+RF",  "Accuracy": 0.979, "Precision (Illicit)": 0.91, "Recall (Illicit)": 0.88, "F1 (Illicit)": 0.89},
        {"Model": "GraphSAGE",    "Accuracy": 0.981, "Precision (Illicit)": 0.92, "Recall (Illicit)": 0.90, "F1 (Illicit)": 0.91},
    ]
    df_results = pd.DataFrame(results)

    plot_model_comparison(df_results)
    plot_bar_comparison(df_results)


if __name__ == "__main__":
    main()
