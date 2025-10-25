import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score
)

sns.set_theme(style='whitegrid')





def prepare_data(merged_df):
    """
    Filters known (labeled) transactions and prepares X, y.
    """
    df_known = merged_df[merged_df['class'] != 3].copy()
    df_unknown = merged_df[merged_df['class'] == 3].copy()

    # --- Select and clean feature columns ---
    feature_cols = [c for c in merged_df.columns
                    if 'feature' in c.lower() and c not in ['txId', 'time step', 'class']]

    X = df_known[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X.loc[:, X.var() != 0]  # drop constant columns
    y = df_known['class'].astype(int)

    print("Prepared data shape:", X.shape)
    print("Remaining NaNs:", X.isna().sum().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_unknown = None
    if not df_unknown.empty:
        X_unknown = df_unknown[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        X_unknown = X_unknown.loc[:, X_unknown.var() != 0]

    return X_train, X_test, y_train, y_test, X_unknown


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Visualizes confusion matrix for licit vs illicit classification."""
    cm = confusion_matrix(y_test, y_pred, labels=[2, 1])
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=['Pred Licit (2)', 'Pred Illicit (1)'],
        yticklabels=['True Licit (2)', 'True Illicit (1)'],
        cmap='Blues'
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()


def evaluate_model(y_test, y_pred, y_prob=None):
    """Prints evaluation metrics and optional ROC-AUC / PR-AUC."""
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 (weighted):", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    if y_prob is not None:
        # Ensure probability for class 1 (illicit)
        roc_auc = roc_auc_score((y_test == 1), y_prob)
        pr_auc = average_precision_score((y_test == 1), y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC : {pr_auc:.4f}")


def train_random_forest(merged_df):
    """Trains and evaluates a Random Forest classifier."""
    X_train, X_test, y_train, y_test, X_unknown = prepare_data(merged_df)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            random_state=42, class_weight='balanced',
            n_estimators=200, max_depth=10))
    ])
    print("Training shape:", X_train.shape)
    print("Dtypes:\n", X_train.dtypes.value_counts())
    print("Non-numeric columns:\n", X_train.select_dtypes(exclude='number').columns.tolist()[:10])
    print("Any NaN in X_train:", X_train.isna().sum().sum())

    print("=== DEBUG RF TRAIN DATA ===")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_train columns:", list(X_train.columns[:10]))
    print("Sample values:\n", X_train.head(3))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Random Forest Results ===")
    evaluate_model(y_test, y_pred)

    # Probabilities for ROC/PR
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        y_prob_illicit = proba[:, list(pipeline.classes_).index(1)]
        evaluate_model(y_test, y_pred, y_prob_illicit)

    plot_confusion_matrix(y_test, y_pred, "Random Forest")

    # Predict unknowns
    if X_unknown is not None and not X_unknown.empty:
        unknown_pred = pipeline.predict(X_unknown)
        licit_count = np.count_nonzero(unknown_pred == 2)
        illicit_count = np.count_nonzero(unknown_pred == 1)
        plt.bar(["Licit (2)", "Illicit (1)"], [licit_count, illicit_count],
                color=['#4caf50', '#f44336'])
        plt.ylabel("Count")
        plt.title("Predicted classes on previously unknown transactions")
        plt.show()


def train_svm(merged_df):
    """Trains and evaluates an SVM classifier."""
    X_train, X_test, y_train, y_test, X_unknown = prepare_data(merged_df)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42,
                    class_weight='balanced', probability=True))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== SVM Results ===")
    evaluate_model(y_test, y_pred)

    # Probabilities for ROC/PR
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        y_prob_illicit = proba[:, list(pipeline.classes_).index(1)]
        evaluate_model(y_test, y_pred, y_prob_illicit)

    plot_confusion_matrix(y_test, y_pred, "SVM")

    # Predict unknowns
    if X_unknown is not None and not X_unknown.empty:
        unknown_pred = pipeline.predict(X_unknown)
        licit_count = np.count_nonzero(unknown_pred == 2)
        illicit_count = np.count_nonzero(unknown_pred == 1)
        plt.bar(["Licit (2)", "Illicit (1)"], [licit_count, illicit_count],
                color=['#4caf50', '#f44336'])
        plt.ylabel("Count")
        plt.title("Predicted classes on previously unknown transactions")
        plt.show()

