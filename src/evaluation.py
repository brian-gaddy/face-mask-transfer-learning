"""Evaluation utilities for multiclass face-mask classification."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


CLASS_NAMES = ["incorrect_mask", "with_mask", "without_mask"]


def evaluate_predictions(y_true: Iterable[int], y_pred: Iterable[int], labels: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a confusion-matrix table and classification-report table."""
    labels = labels or CLASS_NAMES
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.index.name = "true_label"

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index(names="class")
    return matrix_df, report_df


def save_confusion_matrix_plot(matrix_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a readable confusion-matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix_df.values, cmap="Blues")
    ax.set_title("Face Mask Detection Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(matrix_df.columns)))
    ax.set_yticks(range(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix_df.index)

    for row in range(matrix_df.shape[0]):
        for col in range(matrix_df.shape[1]):
            ax.text(col, row, str(matrix_df.iloc[row, col]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_class_metrics_plot(report_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a grouped bar chart for precision, recall, and F1-score by class."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_rows = report_df[report_df["class"].isin(CLASS_NAMES)].copy()
    metrics = ["precision", "recall", "f1-score"]

    x = np.arange(len(class_rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, metric in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, class_rows[metric], width, label=metric)

    ax.set_title("Per-Class Precision, Recall, and F1-Score")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_rows["class"], rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
