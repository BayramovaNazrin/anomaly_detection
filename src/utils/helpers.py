import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.multiclass import unique_labels


# ==========================================================
#  Directory setup
# ==========================================================
def setup_directories():
    """
    Ensure all necessary output folders exist.
    """
    for folder in ["artifacts", "artifacts/plots", "artifacts/models", "artifacts/data"]:
        os.makedirs(folder, exist_ok=True)
    print("✅ Required directories are ready.")


# ==========================================================
#  Safe report generator
# ==========================================================
def safe_report(y_true, y_pred):
    """
    Generate a classification report safely, handling missing labels.
    Returns a string and key metrics as a dict.
    """
    labels = unique_labels(y_true, y_pred)
    names = ["Licit" if l == 0 else "Illicit" for l in labels]
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=names,
        digits=4,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }

    print("\n=== Classification Report ===")
    print(report)
    return metrics


# ==========================================================
# Threshold-based evaluator (optional for GNNs)
# ==========================================================
def evaluate_with_threshold(y_true, y_prob, threshold=0.5):
    """
    Evaluate probabilistic predictions (e.g., sigmoid outputs from GNNs).
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob)
    }

    print(f"\n=== Evaluation (threshold={threshold}) ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")
    return metrics


# ==========================================================
#  Reproducibility helper
# ==========================================================
def set_seed(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    import random
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"✅ Random seed set to {seed}")

# ==========================================================
#  Data path resolver
# ==========================================================
def get_data_path(filename):
    """
    Return the absolute path to a file inside the project's 'data' folder,
    regardless of where the script is executed from.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    return os.path.join(base_dir, filename)
