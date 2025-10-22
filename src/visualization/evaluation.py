import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ============================================================
# 1️⃣  Radar chart for model comparison
# ============================================================

def plot_model_comparison(results_df, save_path="artifacts/plots/model_comparison_radar.png"):
    """
    Creates a radar/spider chart comparing multiple models.
    Expected columns: Model, Accuracy, Precision (Illicit), Recall (Illicit), F1 (Illicit)
    """
    if results_df.empty:
        print("⚠️ No results provided to plot.")
        return

    labels = ["Accuracy", "Precision (Illicit)", "Recall (Illicit)", "F1 (Illicit)"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [
            row["Accuracy"],
            row["Precision (Illicit)"],
            row["Recall (Illicit)"],
            row["F1 (Illicit)"]
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, label=row["Model"], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Model Performance Comparison", size=14, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Radar chart saved: {save_path}")


# ============================================================
# 2️⃣  Bar chart for metric comparison
# ============================================================

def plot_bar_comparison(results_df, save_path="artifacts/plots/model_comparison_bars.png"):
    """
    Creates a grouped bar chart comparing Accuracy, F1, Precision, Recall.
    """
    metrics = ["Accuracy", "F1 (Illicit)", "Precision (Illicit)", "Recall (Illicit)"]
    melted = results_df.melt(id_vars="Model", value_vars=metrics,
                             var_name="Metric", value_name="Score")

    plt.figure(figsize=(9, 6))
    for i, metric in enumerate(metrics):
        subset = melted[melted["Metric"] == metric]
        plt.bar(subset["Model"], subset["Score"], alpha=0.9 / (i + 1),
                label=metric)

    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Bar chart saved: {save_path}")
