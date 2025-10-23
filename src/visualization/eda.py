import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")

def run_eda(merged_df, save_dir="artifacts/plots"):
    """
    Perform exploratory data analysis (EDA) and save basic visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Basic statistics ---
    print("=== Dataset Overview ===")
    print(f"Total transactions: {len(merged_df):,}")
    print("Class distribution:")
    print(merged_df['class'].value_counts())

    # --- Missing values ---
    missing = merged_df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\nColumns with missing values:\n", missing)
    else:
        print("\nNo missing values found ✅")

    # --- Class distribution plot ---
    class_counts = merged_df['class'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        palette="Set2",
        legend=False
    )
    plt.title("Class Distribution (1=Illicit, 2=Licit, 3=Unknown)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_distribution.png", dpi=300)
    plt.close()

    # --- Feature histograms ---
    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.tolist()
    print(f"\n📊 Found {len(numeric_cols)} numeric columns out of {merged_df.shape[1]} total.")

    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns found to plot histograms.")
    else:
        print(f"🧩 Plotting first 10 numeric columns: {numeric_cols[:10]}")
        merged_df[numeric_cols[:10]].hist(figsize=(12, 8))
        plt.suptitle("Distribution of First 10 Numeric Features", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_histograms.png", dpi=300)
        plt.show()
        plt.close()

    # --- Correlation heatmap ---
    feature_cols = [c for c in merged_df.columns if c.startswith("feature_")]
    if len(feature_cols) > 0:
        subset = feature_cols[:min(30, len(feature_cols))]
        corr = merged_df[subset].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Feature Correlation (First 30 Features)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=300)
        plt.close()
    else:
        print("⚠️ No 'feature_' columns found for correlation heatmap.")

    # --- Time-step analysis ---
    if "time_step" in merged_df.columns or "Time step" in merged_df.columns:
        t_col = "time_step" if "time_step" in merged_df.columns else "Time step"
        counts = merged_df.groupby([t_col, "class"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
        plt.title("Transaction Classes over Time Steps")
        plt.xlabel("Time Step")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/time_distribution.png", dpi=300)
        plt.close()

    print(f"\n✅ EDA visualizations saved to: {save_dir}")
