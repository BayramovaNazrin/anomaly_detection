import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


# ==========================================================
# 1️⃣ Permutation Feature Importance for GraphSAGE
# ==========================================================
def permutation_importance(model, data, test_fn, metric='f1_score', device='cpu'):
    """
    Compute permutation feature importance for a trained GraphSAGE model.
    model: trained GraphSAGE model
    data: torch_geometric.data.Data
    test_fn: function(model, data) -> dict of metrics
    metric: which metric key to evaluate (f1_score, accuracy, etc.)
    """
    model.eval()
    data = data.to(device)

    baseline_metrics = test_fn(model, data)
    baseline_score = baseline_metrics[metric]
    print(f"Baseline {metric}: {baseline_score:.4f}")

    x_orig = data.x.clone()
    importances = []

    for i in range(x_orig.shape[1]):
        x_perm = x_orig.clone()
        idx = torch.randperm(x_perm.shape[0])
        x_perm[:, i] = x_perm[idx, i]
        data_perm = Data(
            x=x_perm, edge_index=data.edge_index, y=data.y,
            train_mask=data.train_mask, test_mask=data.test_mask
        ).to(device)
        perm_metric = test_fn(model, data_perm)[metric]
        importances.append(baseline_score - perm_metric)

    return np.array(importances)


# ==========================================================
# 2️⃣ Plotting Utilities
# ==========================================================
def plot_feature_importances(importances, feature_names, save_dir="artifacts/plots", top_k=30):
    """
    Plot top and bottom feature importances.
    """
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    imp_df.head(top_k).iloc[::-1].plot(kind='barh', x='Feature', y='Importance', color='teal', legend=False)
    plt.title(f"Top {top_k} Important Features (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/graphsage_top_features.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    imp_df.tail(top_k).iloc[::-1].plot(kind='barh', x='Feature', y='Importance', color='coral', legend=False)
    plt.title(f"Least {top_k} Important Features (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/graphsage_least_features.png", dpi=300)
    plt.close()

    print(f"✅ Saved feature importance plots in {save_dir}")
    return imp_df


# ==========================================================
# 3️⃣ Recreate Graph with Selected Top Features
# ==========================================================
def create_top_feature_graph(graph_data, feature_df, top_features, device):
    """
    Create a new graph using only the top features.
    """
    feature_indices = [i for i, col in enumerate(feature_df.columns) if col in top_features]
    x_top = graph_data.x[:, feature_indices].clone()

    graph_data_top = Data(
        x=x_top,
        edge_index=graph_data.edge_index,
        y=graph_data.y,
        train_mask=graph_data.train_mask,
        test_mask=graph_data.test_mask
    ).to(device)

    print(f"Graph with top {len(top_features)} features created:")
    print(f"  - Nodes: {graph_data_top.num_nodes}")
    print(f"  - Node feature dimension: {graph_data_top.num_node_features}")
    return graph_data_top


# ==========================================================
# 4️⃣ Retraining GraphSAGE on Top Features
# ==========================================================
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def retrain_with_top_features(graph_data_top, hidden_dim=128, lr=0.01, weight_decay=5e-4, epochs=200, device='cpu'):
    """
    Retrain GraphSAGE on graph with top features only.
    """
    model = GraphSAGEModel(
        in_channels=graph_data_top.num_node_features,
        hidden_channels=hidden_dim,
        out_channels=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    from src.models.graph_models import test, train  # reuse existing functions

    print(f"\nTraining GraphSAGE on top {graph_data_top.num_node_features} features...")
    for epoch in range(1, epochs + 1):
        loss = train(model, graph_data_top, optimizer, criterion)
        if epoch % 10 == 0:
            metrics = test(model, graph_data_top)
            print(f"Epoch {epoch:03d} | Loss={loss:.4f} | F1={metrics['f1_score']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")

    final_metrics = test(model, graph_data_top)
    print("\n✅ Final evaluation after retraining on top features:")
    for metric, value in final_metrics.items():
        print(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")

    return model, final_metrics
