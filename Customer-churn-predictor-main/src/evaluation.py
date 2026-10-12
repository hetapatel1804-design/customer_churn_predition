"""
evaluation.py
--------------
Metric computation and plotting utilities used during model comparison
and for the Streamlit "Model Evaluation" page.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute the standard classification metrics. All real, no invented numbers."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def get_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])


def plot_confusion_matrix(y_true, y_pred, save_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return cm


def plot_roc_curve(y_true, y_proba, save_path: str, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, save_path: str, title: str = "Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
