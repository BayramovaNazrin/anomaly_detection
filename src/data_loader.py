# src/data_loader.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_explore_data(features_path, edges_path, classes_path):


    # --- Load CSV files ---
    features = pd.read_csv(features_path)
    edges = pd.read_csv(edges_path)
    classes = pd.read_csv(classes_path, header=None)

    # --- Set proper column names for classes ---
    classes.columns = ['txId', 'class']

    # --- Normalize and clean 'class' column ---
    classes['class'] = classes['class'].astype(str).str.strip().replace({'unknown': 3})
    features['txId'] = features['txId'].astype(str)
    classes['txId'] = classes['txId'].astype(str)

    # --- Merge ---
    merged_df = features.merge(classes, on='txId', how='left')
    merged_df['class'] = pd.to_numeric(merged_df['class'], errors='coerce')

    # === Dataset Summary ===
    print("\n=== DATASET SUMMARY ===")

    num_nodes = features.shape[0]
    num_edges = edges.shape[0]
    num_transactions = merged_df.shape[0]

    licit = merged_df['class'].eq(2).sum()
    illicit = merged_df['class'].eq(1).sum()
    unlabeled = merged_df['class'].eq(3).sum()
    nan_count = merged_df['class'].isna().sum()

    licit_pct = licit / num_nodes * 100
    illicit_pct = illicit / num_nodes * 100
    unlabeled_pct = unlabeled / num_nodes * 100

    print(f"Nodes: {num_nodes:,}")
    print(f"Edges: {num_edges:,}")
    print(f"Transactions: {num_transactions:,}")
    print(f"Licit (2): {licit:,} ({licit_pct:.2f}%)")
    print(f"Illicit (1): {illicit:,} ({illicit_pct:.2f}%)")
    print(f"Unlabeled (3): {unlabeled:,} ({unlabeled_pct:.2f}%)")
    print(f"Missing labels: {nan_count:,}\n")

    # === Missing Values ===
    print("=== Missing Values ===")
    missing = merged_df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("None found ✅")
    else:
        print(missing)

    # === Descriptive Stats ===
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(features.describe().T, '\n', '-' * 60)
    print(edges.describe().T, '\n', '-' * 60)
    print(classes.describe().T, '\n', '-' * 60)

    # === Data Types ===
    print("\n=== DATA TYPES ===")
    print('Features table:\n', features.dtypes)
    print('\nEdges table:\n', edges.dtypes)
    print('\nClasses table:\n', classes.dtypes)

    # === Missing Value Check ===
    print("\n=== MISSING VALUE CHECK ===")
    print("Features missing:\n", features.isna().sum().sum())
    print("Edges missing:\n", edges.isna().sum().sum())
    print("Classes missing:\n", classes.isna().sum().sum())

    # === Visualize Feature Distributions ===
    print("\nPlotting feature distributions (first 8 features)...")
    features.iloc[:, 2:10].hist(figsize=(10, 6))
    plt.tight_layout()
    plt.show()

    # === Correlation Matrix ===
    print("\nPlotting correlation heatmap...")
    correlation_matrix = features.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()

    print("\n✅ Data loading and summary complete.")
    return features, edges, classes, merged_df
