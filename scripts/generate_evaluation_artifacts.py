"""Generate evaluation artifacts from saved face-mask predictions.

Expected input CSV columns:
- true_label: class name or integer label
- predicted_label: class name or integer label

Example:
    python scripts/generate_evaluation_artifacts.py --predictions data/processed/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import (
    CLASS_NAMES,
    evaluate_predictions,
    save_confusion_matrix_plot,
    save_per_class_metrics_plot,
)


def normalize_labels(series: pd.Series) -> pd.Series:
    """Convert class names to integer IDs when needed."""
    mapping = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if series.dtype == object:
        return series.map(mapping).astype(int)
    return series.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with true_label and predicted_label columns")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for CSV outputs")
    parser.add_argument("--figures-dir", default="figures", help="Directory for figure outputs")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    y_true = normalize_labels(predictions["true_label"])
    y_pred = normalize_labels(predictions["predicted_label"])

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    matrix_df, report_df = evaluate_predictions(y_true, y_pred)
    matrix_df.to_csv(output_dir / "confusion_matrix.csv")
    report_df.to_csv(output_dir / "classification_report.csv", index=False)

    save_confusion_matrix_plot(matrix_df, figures_dir / "confusion_matrix.png")
    save_per_class_metrics_plot(report_df, figures_dir / "per_class_metrics.png")

    print("Saved confusion matrix, classification report, and metric figures.")


if __name__ == "__main__":
    main()
