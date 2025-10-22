import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_rf_feature_importance(imp_df, save_dir="artifacts/plots", top_k=30, least_k=20):
    """
    Plot feature importances for RandomForest:
    - Top k most important features
    - Least k important features
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Top K ---
    plt.figure(figsize=(10, 8))
    imp_df.head(top_k).iloc[::-1].plot(
        kind='barh',
        x='Feature',
        y='Importance',
        color='#1f77b4',
        legend=False
    )
    plt.title(f"Top {top_k} Important Features (RandomForest)")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rf_top_features.png", dpi=300)
    plt.close()

    # --- Least K ---
    plt.figure(figsize=(10, 8))
    imp_df.tail(least_k).iloc[::-1].plot(
        kind='barh',
        x='Feature',
        y='Importance',
        color='r',
        legend=False
    )
    plt.title(f"Least {least_k} Important Features (RandomForest)")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rf_least_features.png", dpi=300)
    plt.close()

    print(f"✅ RandomForest importance plots saved in {save_dir}")


def plot_graphsage_importance(imp_df, save_dir="artifacts/plots", top_k=30, least_k=30):
    """
    Plot top and least important features for GraphSAGE (permutation importance results).
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Top K ---
    plt.figure(figsize=(10, 8))
    imp_df.head(top_k).iloc[::-1].plot(
        kind='barh',
        x='Feature',
        y='Importance',
        color='teal',
        legend=False
    )
    plt.title(f"Top {top_k} Important Features (GraphSAGE Permutation Importance)")
    plt.xlabel("Importance (ΔF1)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/graphsage_top_features.png", dpi=300)
    plt.close()

    # --- Least K ---
    plt.figure(figsize=(10, 8))
    imp_df.tail(least_k).iloc[::-1].plot(
        kind='barh',
        x='Feature',
        y='Importance',
        color='coral',
        legend=False
    )
    plt.title(f"Least {least_k} Important Features (GraphSAGE Permutation Importance)")
    plt.xlabel("Importance (ΔF1)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/graphsage_least_features.png", dpi=300)
    plt.close()

    print(f"✅ GraphSAGE importance plots saved in {save_dir}")


def plot_feature_comparison(rf_imp_df, gs_imp_df, save_dir="artifacts/plots", top_k=20):
    """
    Compare top overlapping features between RandomForest and GraphSAGE importances.
    """
    os.makedirs(save_dir, exist_ok=True)

    rf_top = set(rf_imp_df.head(top_k)['Feature'])
    gs_top = set(gs_imp_df.head(top_k)['Feature'])

    overlap = rf_top.intersection(gs_top)
    unique_rf = rf_top - gs_top
    unique_gs = gs_top - rf_top

    overlap_df = pd.DataFrame({
        'Feature': list(overlap),
        'Type': 'Overlap'
    })
    rf_unique_df = pd.DataFrame({
        'Feature': list(unique_rf),
        'Type': 'RandomForest-only'
    })
    gs_unique_df = pd.DataFrame({
        'Feature': list(unique_gs),
        'Type': 'GraphSAGE-only'
    })

    combined = pd.concat([overlap_df, rf_unique_df, gs_unique_df], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.countplot(data=combined, y='Type', order=['Overlap', 'RandomForest-only', 'GraphSAGE-only'], palette='Set2')
    plt.title(f"Overlap of Top {top_k} Important Features")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_overlap_comparison.png", dpi=300)
    plt.close()

    print(f"✅ Feature overlap comparison plot saved in {save_dir}")
