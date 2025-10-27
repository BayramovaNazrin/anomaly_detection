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
# 1️⃣ NODE2VEC + RANDOM FOREST
# ============================================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import pandas as pd
import numpy as np



def train_node2vec_rf(features, edges, classes):
    """
    Train Node2Vec embeddings + RandomForest classifier using pre-loaded dataframes.
    Returns a dictionary of performance metrics.
    """

    # --- Ensure correct types ---
    features['txId'] = features['txId'].astype(str)
    edges['txId1'] = edges['txId1'].astype(str)
    edges['txId2'] = edges['txId2'].astype(str)
    classes['txId'] = classes['txId'].astype(str)

    # --- Keep only valid classes 1 (illicit) and 2 (licit) ---
    classes = classes[classes['class'].astype(str).isin(['1', '2'])].copy()
    classes['class'] = classes['class'].astype(int)

    # --- Determine valid nodes present in features, edges, and classes ---
    nodes_in_features = set(features['txId'])
    nodes_in_edges = set(edges['txId1']).union(set(edges['txId2']))
    nodes_in_classes = set(classes['txId'])
    valid_node_ids = list(nodes_in_features & nodes_in_edges & nodes_in_classes)

    if len(valid_node_ids) == 0:
        raise ValueError("No overlapping txId found between features, edges, and classes.")

    # --- Filter dataframes to only valid nodes ---
    features = features[features['txId'].isin(valid_node_ids)].copy()
    classes = classes[classes['txId'].isin(valid_node_ids)].copy()
    edges = edges[edges['txId1'].isin(valid_node_ids) & edges['txId2'].isin(valid_node_ids)].copy()

    # --- Map txId to integer indices for Node2Vec ---
    node_ids = sorted(valid_node_ids)
    node_id_map = {tx: i for i, tx in enumerate(node_ids)}
    edge_index = torch.tensor(
        [[node_id_map[src], node_id_map[dst]] for src, dst in zip(edges['txId1'], edges['txId2'])],
        dtype=torch.long
    ).t().contiguous()

    # --- Prepare feature matrix ---
    x_features = features.drop(columns=['txId', 'Time step'], errors='ignore') \
                         .apply(pd.to_numeric, errors='coerce') \
                         .fillna(0)
    features_aligned = x_features.set_index(features['txId']).loc[node_ids].values

    # --- Align labels ---
    y_aligned = classes.set_index('txId').loc[node_ids]['class'].values

    # --- Node2Vec embeddings ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node2vec = Node2Vec(
        edge_index=edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1, q=1,
        sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=2)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    print("Training Node2Vec embeddings...")
    for epoch in range(1, 6):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    embeddings_aligned = embeddings  # Node2Vec embeddings already follow node_ids order

    # --- Combine embeddings and features ---
    X_combined = np.concatenate([embeddings_aligned, features_aligned], axis=1)

    # --- Train/test split ---
    node_indices = np.arange(len(y_aligned))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        node_indices, y_aligned, test_size=0.4, random_state=42, stratify=y_aligned
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    X_train = X_combined[train_idx]
    X_val = X_combined[val_idx]
    X_test = X_combined[test_idx]

    # --- Train RandomForest ---
    print("Training RandomForest on Node2Vec features...")
    clf = RandomForestClassifier(
        n_estimators=500, random_state=42,
        class_weight={1: 5, 2: 1}, max_depth=15
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- Probabilities for ROC/PR metrics ---
    illicit_class_index = list(clf.classes_).index(1)
    y_prob = clf.predict_proba(X_test)[:, illicit_class_index]

    print("\n=== Node2Vec + RandomForest ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Confusion matrix
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print("\nConfusion matrix:\n", cm)

    # --- Return metrics ---
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 (Illicit)": f1_score(y_test, y_pred, pos_label=1),
        "Precision (Illicit)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Illicit)": recall_score(y_test, y_pred, pos_label=1),
        "ROC-AUC": roc_auc_score((y_test == 1), y_prob),
        "PR-AUC": average_precision_score((y_test == 1), y_prob)
    }

    return metrics


# ============================================================
# 2️⃣ GRAPH SAGE MODEL
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


def build_graph_data(features_df, edges_df, classes_df):
    """ Builds torch_geometric Data object from pre-loaded dataframes."""
    
    # --- Clean and merge ---
    # Note: Using copies to avoid SettingWithCopyWarning
    features_df = features_df.copy()
    classes_df = classes_df.copy()

    classes_df['class'] = (
        classes_df['class']
        .astype(str)
        .str.strip()
        .replace({'unknown': 3, '1': 1, '2': 2}) # Use numeric labels
    )
    classes_df['class'] = pd.to_numeric(classes_df['class'], errors='coerce')
    
    # Ensure txId is string for merging
    features_df['txId'] = features_df['txId'].astype(str)
    classes_df['txId'] = classes_df['txId'].astype(str)

    combined = pd.merge(features_df, classes_df, on='txId', how='inner')
    
    # Filter for known classes (1 and 2)
    df = combined[combined['class'].isin([1, 2])].copy()

    # --- Remap labels for GraphSAGE: 1 (illicit) -> 1, 2 (licit) -> 0 ---
    df['class'] = df['class'].map({1: 1, 2: 0})
    
    print(f"GraphSAGE using {len(df)} labeled nodes (Illicit: 1, Licit: 0).")

    # --- Prepare features (X) ---
    # Get feature columns (all columns except txId, Time step, and class)
    feature_cols = [c for c in df.columns if c not in ['txId', 'Time step', 'class']]
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(df[feature_cols]), dtype=torch.float)

    # --- Create node mapping ---
    tx_map = {tx: i for i, tx in enumerate(df['txId'])}
    
    # Ensure edges txId types match the map keys
    edges_df['txId1'] = edges_df['txId1'].astype(str)
    edges_df['txId2'] = edges_df['txId2'].astype(str)

    valid_edges = edges_df[
        edges_df['txId1'].isin(tx_map) & edges_df['txId2'].isin(tx_map)
    ].copy()
    
    # Map edges to new integer indices
    valid_edges['txId1'] = valid_edges['txId1'].map(tx_map)
    valid_edges['txId2'] = valid_edges['txId2'].map(tx_map)
    edge_index = torch.tensor(valid_edges[['txId1', 'txId2']].values.T, dtype=torch.long)

    # --- Prepare labels (y) and masks ---
    y = torch.tensor(df['class'].values, dtype=torch.long)
    time_steps = torch.tensor(df['Time step'].values)
    
    # Using original paper's split: T <= 34 for train, T > 34 for test
    train_mask = time_steps <= 34
    test_mask = time_steps > 34

    print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    
    # Store feature names for importance plotting later
    data.feature_names = feature_cols 
    
    return data

# --- Top-level functions for training and evaluation ---
# (Moved from inside train_graphsage to be importable)

def train_graphsage_epoch(model, data, optimizer, criterion):
    """Performs a single training epoch for GraphSAGE."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_graphsage_test(model, data):
    """Evaluates the GraphSAGE model on the test set."""
    model.eval()
    out = model(data.x, data.edge_index)
    
    # Get probabilities for class 1 (illicit)
    probs = F.softmax(out, dim=1)[:, 1] 
    preds = probs > 0.5 # Using 0.5 threshold
    
    test_labels = data.y[data.test_mask].cpu().numpy()
    test_preds = preds[data.test_mask].cpu().numpy()
    test_probs = probs[data.test_mask].cpu().numpy()
    
    if len(test_labels) == 0 or len(test_preds) == 0:
        print("Warning: No test samples found. Returning zero metrics.")
        return {
            "Accuracy": 0, "F1 (Illicit)": 0, "Precision (Illicit)": 0,
            "Recall (Illicit)": 0, "ROC-AUC": 0, "PR-AUC": 0
        }

    return {
        "Accuracy": accuracy_score(test_labels, test_preds),
        "F1 (Illicit)": f1_score(test_labels, test_preds),
        "Precision (Illicit)": precision_score(test_labels, test_preds, zero_division=0),
        "Recall (Illicit)": recall_score(test_labels, test_preds, zero_division=0),
        "ROC-AUC": roc_auc_score(test_labels, test_probs),
        "PR-AUC": average_precision_score(test_labels, test_probs),
    }


def train_graphsage(features, edges, classes, epochs=200):
    """
    Trains and evaluates a GraphSAGE model using pre-loaded dataframes.
    
    Returns:
    - model: The trained torch model.
    - data: The torch_geometric Data object.
    - metrics: A dictionary of final performance metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Build graph data ---
    # We pass the original dataframes to the builder
    data = build_graph_data(features, edges, classes).to(device)
    
    if data.num_node_features == 0:
        raise ValueError("Graph data has 0 node features. Check feature columns.")
        
    model = GraphSAGEModel(data.num_node_features, 128, 2).to(device) # 2 classes: 0 (licit), 1 (illicit)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Adjust loss for class imbalance if needed
    # Class 0 (licit) is majority, Class 1 (illicit) is minority
    class_counts = data.y[data.train_mask].bincount()
    if len(class_counts) == 2:
        weight = class_counts[0].item() / class_counts[1].item()
        print(f"Using class weight for Illicit (1): {weight:.2f}")
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight]).to(device))
    else:
        print("Warning: Could not calculate class weights. Using standard CrossEntropyLoss.")
        criterion = torch.nn.CrossEntropyLoss()


    print("Training GraphSAGE...")
    for epoch in range(1, epochs + 1):
        loss = train_graphsage_epoch(model, data, optimizer, criterion)
        if epoch % 20 == 0:
            metrics = evaluate_graphsage_test(model, data)
            print(f"Epoch {epoch:03d}, Loss={loss:.4f}, F1={metrics['F1 (Illicit)']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}")

    print("\nFinal GraphSAGE Metrics:")
    final_metrics = evaluate_graphsage_test(model, data)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    return model, data, final_metrics
