import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1️⃣  NODE2VEC + RANDOM FOREST
# ============================================================

def train_node2vec_rf(features_path, edges_path, classes_path):
    """
    Train Node2Vec embeddings and a RandomForest classifier.
    Returns performance metrics.
    """
    # --- Load data ---
    features = pd.read_csv(features_path, header=None)
    edges = pd.read_csv(edges_path)
    classes = pd.read_csv(classes_path)
    classes = classes[classes['class'] != 'unknown'].copy()
    classes['class'] = classes['class'].astype(int)

    # Align nodes
    valid_nodes = classes['txId'].values
    features = features[features[0].isin(valid_nodes)]
    node_ids = features[0].values
    node_id_map = {id_: i for i, id_ in enumerate(node_ids)}

    # Build edge index
    edges = edges[edges['txId1'].isin(node_id_map) & edges['txId2'].isin(node_id_map)]
    edge_index = torch.tensor(
        [[node_id_map[src], node_id_map[dst]] for src, dst in zip(edges['txId1'], edges['txId2'])],
        dtype=torch.long
    ).t().contiguous()

    x = torch.tensor(features.iloc[:, 1:].values, dtype=torch.float)
    x = torch.tensor(
    features.drop(columns=["txId"], errors="ignore")
            .select_dtypes(include=[float, int])
            .fillna(0)
            .values,
    dtype=torch.float
)

    y = torch.tensor(classes.set_index('txId').loc[node_ids, 'class'].values, dtype=torch.long)

    # --- Split ---
    node_indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(node_indices, test_size=0.4, random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

    y_train, y_val, y_test = y[train_idx].numpy(), y[val_idx].numpy(), y[test_idx].numpy()

    # --- Train Node2Vec ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node2vec = Node2Vec(
        edge_index=edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1, q=1, sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=2)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    for epoch in range(1, 6):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Node2Vec Epoch {epoch}, Loss: {total_loss:.4f}")

    # --- Combine embeddings + features ---
    embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    original_features = features.iloc[:, 1:].values
    X_combined = np.concatenate([embeddings, original_features], axis=1)

    X_train = X_combined[train_idx]
    X_val = X_combined[val_idx]
    X_test = X_combined[test_idx]

    # --- Random Forest ---
    clf = RandomForestClassifier(
        n_estimators=500, random_state=42,
        class_weight={1: 5, 2: 1}, max_depth=15
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Node2Vec + RandomForest ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    cm = pd.crosstab(y_test, y_pred)
    print("\nConfusion matrix:\n", cm)

    return clf, X_combined, y


# ============================================================
# 2️⃣  GRAPH SAGE MODEL
# ============================================================

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


def build_graph_data(features_path, edges_path, classes_path):
    features_df = pd.read_csv(features_path)
    edges_df = pd.read_csv(edges_path)
    classes_df = pd.read_csv(classes_path)
    classes_df['class'] = (
        classes_df['class']
        .astype(str)
        .str.strip()
        .replace({'unknown': 3, '1': 1, '2': 2})
        .astype(int)
    )
    combined = pd.merge(features_df, classes_df, on='txId', how='inner')
    df = combined[combined['class'].isin([1, 2])].copy()

    df['class'] = df['class'].map({1: 1, 2: 0})

    
    print("Unique class values in combined:", combined['class'].unique())
    print("Rows before filtering:", len(combined))
    print("Rows after filtering [1,2]:", len(df))

    
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(df.drop(['txId', 'Time step', 'class'], axis=1)), dtype=torch.float)
    tx_map = {tx: i for i, tx in enumerate(df['txId'])}

    valid_edges = edges_df[
        edges_df['txId1'].isin(tx_map) & edges_df['txId2'].isin(tx_map)
    ].copy()
    valid_edges['txId1'] = valid_edges['txId1'].map(tx_map)
    valid_edges['txId2'] = valid_edges['txId2'].map(tx_map)
    edge_index = torch.tensor(valid_edges[['txId1', 'txId2']].values.T, dtype=torch.long)

    y = torch.tensor(df['class'].values, dtype=torch.long)
    time_steps = torch.tensor(df['Time step'].values)
    train_mask = time_steps <= 34
    test_mask = time_steps > 34

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data


def train_graphsage(features_path, edges_path, classes_path, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build_graph_data(features_path, edges_path, classes_path).to(device)
    model = GraphSAGEModel(data.num_node_features, 128, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train_epoch():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = probs > 0.5
        test_labels = data.y[data.test_mask].cpu().numpy()
        test_preds = preds[data.test_mask].cpu().numpy()
        test_probs = probs[data.test_mask].cpu().numpy()
        return {
            "accuracy": accuracy_score(test_labels, test_preds),
            "f1": f1_score(test_labels, test_preds),
            "precision": precision_score(test_labels, test_preds),
            "recall": recall_score(test_labels, test_preds),
            "roc_auc": roc_auc_score(test_labels, test_probs),
            "pr_auc": average_precision_score(test_labels, test_probs),
        }

    print("Training GraphSAGE...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch()
        if epoch % 20 == 0:
            metrics = evaluate()
            print(f"Epoch {epoch:03d}, Loss={loss:.4f}, F1={metrics['f1']:.4f}")

    print("\nFinal GraphSAGE Metrics:")
    final = evaluate()
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")
    return model, final
