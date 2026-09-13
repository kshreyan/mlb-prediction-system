"""Calibration and honest-performance metrics. Raw accuracy is reported but
explicitly de-emphasized — log loss, Brier score, and calibration error are
what this system is actually selected and judged on, per the project's
honesty standard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": len(y_true),
        "accuracy": float(np.mean((y_prob > 0.5) == y_true)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "mean_predicted_prob": float(np.mean(y_prob)),
        "actual_rate": float(np.mean(y_true)),
    }


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            rows.append({"bin_lo": bins[b], "bin_hi": bins[b + 1], "n": 0, "mean_predicted": np.nan, "mean_actual": np.nan})
            continue
        rows.append({
            "bin_lo": bins[b],
            "bin_hi": bins[b + 1],
            "n": int(mask.sum()),
            "mean_predicted": float(y_prob[mask].mean()),
            "mean_actual": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    tbl = reliability_table(y_true, y_prob, n_bins)
    valid = tbl.dropna()
    n_total = valid["n"].sum()
    if n_total == 0:
        return float("nan")
    ece = np.sum(valid["n"] * np.abs(valid["mean_predicted"] - valid["mean_actual"])) / n_total
    return float(ece)


def total_runs_metrics(actual_total: np.ndarray, pred_mean_total: np.ndarray, total_distribution_samples: list[np.ndarray] | None = None) -> dict:
    actual_total = np.asarray(actual_total, dtype=float)
    pred_mean_total = np.asarray(pred_mean_total, dtype=float)
    mae = float(np.mean(np.abs(actual_total - pred_mean_total)))
    rmse = float(np.sqrt(np.mean((actual_total - pred_mean_total) ** 2)))
    bias = float(np.mean(pred_mean_total - actual_total))
    out = {"mae": mae, "rmse": rmse, "bias": bias, "n": len(actual_total)}

    if total_distribution_samples is not None:
        # Coverage calibration: does the model's own central interval
        # contain the actual total the fraction of the time it claims to?
        coverages = {}
        for level in (0.5, 0.8, 0.9):
            lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
            hits = []
            for actual, samples in zip(actual_total, total_distribution_samples):
                lo, hi = np.quantile(samples, [lo_q, hi_q])
                hits.append(lo <= actual <= hi)
            coverages[f"coverage_{int(level*100)}"] = float(np.mean(hits))
        out.update(coverages)
    return out
