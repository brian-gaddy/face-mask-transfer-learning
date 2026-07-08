import pandas as pd

from src.evaluation import evaluate_predictions


def test_evaluate_predictions_returns_matrix_and_report():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 1, 2]

    matrix, report = evaluate_predictions(y_true, y_pred)

    assert matrix.shape == (3, 3)
    assert matrix.loc["incorrect_mask", "incorrect_mask"] == 1
    assert set(["class", "precision", "recall", "f1-score", "support"]).issubset(report.columns)
    assert "accuracy" in report["class"].values
