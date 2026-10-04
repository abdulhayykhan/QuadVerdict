"""
metrics.py — Threshold tables, ROC/PR curves, calibration bins, and scalar metrics for QuadVerdict.

Design rules (TRD §4.4, PRD §6.2–6.4):
1. Threshold tables:
   - For each repeat, construct the OOF prediction vector (each sample appears exactly once).
   - Evaluate TP, FP, TN, FN across 101 threshold steps in [0, 1].
   - Average counts across repeats (floats allowed).
   - Enforce invariants: tp + fn == n_pos, fp + tn == n_neg, monotonic tp and fp.
2. Curves:
   - ROC and PR computed on repeat-0 OOF predictions.
   - Downsampled to <= 200 points via uniform cumulative arc-length sampling, preserving endpoints.
3. Calibration:
   - 10 quantile bins on repeat-0 OOF probabilities (mean_pred, frac_pos, n).
   - Brier score per fold plus overall.
4. Edge cases:
   - Zero-division in precision (define as 1.0 when tp+fp=0).
   - Zero-division in MCC (define as 0.0 when denominator is 0).
   - Zero-division in F1 (define as 0.0 when 2*tp+fp+fn=0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import DEFAULT_COST_RATIO, N_THRESHOLD_STEPS


def compute_derived_threshold_metrics(
    tp: float,
    fp: float,
    tn: float,
    fn: float,
    cost_ratio: float = DEFAULT_COST_RATIO,
) -> dict[str, float]:
    """Calculate derived metrics from confusion matrix counts with zero-division safety.

    Parameters
    ----------
    tp, fp, tn, fn : float
    cost_ratio : float, default=DEFAULT_COST_RATIO (5)
        FN cost weight with FP cost = 1.0.

    Returns
    -------
    dict[str, float]
        precision, recall, f1, mcc, balanced_accuracy, cost, cost_per_1k.
    """
    total = tp + fp + tn + fn
    # Precision: 1.0 if no positive predictions (TRD §4.4)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0

    # Recall
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # F1
    denom_f1 = 2 * tp + fp + fn
    if denom_f1 > 0:
        f1 = (2 * tp) / denom_f1
    else:
        f1 = 0.0

    # MCC: zero if denominator is zero
    denom_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_mcc > 0:
        mcc = (tp * tn - fp * fn) / math.sqrt(denom_mcc)
    else:
        mcc = 0.0

    # Balanced accuracy: (recall + specificity) / 2
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0
    bal_acc = (recall + specificity) / 2.0

    # Total and normalized cost: cost = c_fn * FN + c_fp * FP
    cost = cost_ratio * fn + 1.0 * fp
    cost_per_1k = (cost / total * 1000.0) if total > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "balanced_accuracy": float(bal_acc),
        "cost": float(cost),
        "cost_per_1k": float(cost_per_1k),
    }


def compute_scalar_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute comprehensive scalar classification metrics for predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_proba : array-like of shape (n_samples,)
        Predicted probability of the positive class.
    threshold : float, default=0.5

    Returns
    -------
    dict[str, float]
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_proba, dtype=float)

    preds = (y_p >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_t, y_p))
    roc_auc = float(roc_auc_score(y_t, y_p))
    brier = float(brier_score_loss(y_t, y_p))

    # Confusion counts
    tp = float(np.sum((preds == 1) & (y_t == 1)))
    fp = float(np.sum((preds == 1) & (y_t == 0)))
    tn = float(np.sum((preds == 0) & (y_t == 0)))
    fn = float(np.sum((preds == 0) & (y_t == 1)))

    derived = compute_derived_threshold_metrics(tp, fp, tn, fn)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        **derived,
    }


def extract_repeat_predictions(
    fold_results: list[dict[str, Any]],
    repeat_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble single OOF predictions for a specific repeat across all its folds.

    Parameters
    ----------
    fold_results : list of fold dicts
    repeat_idx : int

    Returns
    -------
    y_true, y_proba : np.ndarray
    """
    rep_folds = [f for f in fold_results if f.get("repeat_idx") == repeat_idx]
    if not rep_folds:
        # Fallback if no repeat_idx tag
        rep_folds = fold_results

    all_indices = []
    all_y_true = []
    all_y_proba = []

    for f in rep_folds:
        indices = f["test_indices"]
        all_indices.extend(indices)
        all_y_true.extend(f["y_true"])
        all_y_proba.extend(f["y_proba"])

    # Sort by original dataset index to guarantee aligned rows
    sort_order = np.argsort(all_indices)
    sorted_y_true = np.asarray(all_y_true, dtype=int)[sort_order]
    sorted_y_proba = np.asarray(all_y_proba, dtype=float)[sort_order]

    return sorted_y_true, sorted_y_proba


def build_threshold_table(
    fold_results: list[dict[str, Any]],
    n_splits: int,
    n_repeats: int,
    n_steps: int = N_THRESHOLD_STEPS,
    cost_ratio: float = DEFAULT_COST_RATIO,
) -> list[dict[str, Any]]:
    """Build threshold table averaged over outer CV repeats (TRD §4.4).

    Parameters
    ----------
    fold_results : list of dict
        All fold results for a single model from run_nested_cv().
    n_splits : int
    n_repeats : int
    n_steps : int, default=101
    cost_ratio : float, default=DEFAULT_COST_RATIO (5)

    Returns
    -------
    list of dict
        101 entries for thresholds in np.linspace(0, 1, 101).
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)

    # Accumulate counts per repeat
    repeat_counts: list[list[tuple[float, float, float, float]]] = []

    for r in range(n_repeats):
        y_true, y_proba = extract_repeat_predictions(fold_results, repeat_idx=r)
        counts_for_r = []

        is_pos = (y_true == 1)
        is_neg = (y_true == 0)

        for t in thresholds:
            pred_pos = (y_proba >= t)
            tp = float(np.sum(pred_pos & is_pos))
            fp = float(np.sum(pred_pos & is_neg))
            fn = float(np.sum((~pred_pos) & is_pos))
            tn = float(np.sum((~pred_pos) & is_neg))
            counts_for_r.append((tp, fp, tn, fn))

        repeat_counts.append(counts_for_r)

    # Average counts across repeats
    table: list[dict[str, Any]] = []
    n_rep_float = float(n_repeats)

    for step_idx, t in enumerate(thresholds):
        mean_tp = sum(repeat_counts[r][step_idx][0] for r in range(n_repeats)) / n_rep_float
        mean_fp = sum(repeat_counts[r][step_idx][1] for r in range(n_repeats)) / n_rep_float
        mean_tn = sum(repeat_counts[r][step_idx][2] for r in range(n_repeats)) / n_rep_float
        mean_fn = sum(repeat_counts[r][step_idx][3] for r in range(n_repeats)) / n_rep_float

        derived = compute_derived_threshold_metrics(
            mean_tp, mean_fp, mean_tn, mean_fn, cost_ratio=cost_ratio
        )

        entry = {
            "threshold": float(round(float(t), 4)),
            "tp": float(round(mean_tp, 2)),
            "fp": float(round(mean_fp, 2)),
            "tn": float(round(mean_tn, 2)),
            "fn": float(round(mean_fn, 2)),
            **{k: float(round(v, 5)) for k, v in derived.items()},
        }
        table.append(entry)

    return table


def downsample_curve_arc_length(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = 200,
) -> np.ndarray:
    """Downsample a 2D curve uniformly along its cumulative arc length.

    Preserves endpoints and captures sharp transitions without distortion.

    Parameters
    ----------
    x, y : np.ndarray
    max_points : int, default=200

    Returns
    -------
    indices : np.ndarray
        Indices of the retained curve points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= max_points:
        return np.arange(n)

    # Arc length increments
    dx = np.diff(x)
    dy = np.diff(y)
    seg_lens = np.sqrt(dx**2 + dy**2)
    cum_arc = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_arc = cum_arc[-1]

    if total_arc == 0.0:
        return np.array([0, n - 1])

    target_arcs = np.linspace(0.0, total_arc, max_points)
    sampled_indices = np.searchsorted(cum_arc, target_arcs)
    sampled_indices = np.clip(sampled_indices, 0, n - 1)

    # Always pin exact endpoints
    sampled_indices[0] = 0
    sampled_indices[-1] = n - 1

    return np.unique(sampled_indices)


def build_roc_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    max_points: int = 200,
) -> dict[str, list[dict[str, float]]]:
    """Compute and downsample ROC and PR curves from repeat-0 OOF predictions (TRD §4.4).

    Parameters
    ----------
    y_true, y_proba : np.ndarray
    max_points : int, default=200

    Returns
    -------
    dict
        'roc': list of {'fpr', 'tpr', 'threshold'}
        'pr':  list of {'recall', 'precision', 'threshold'}
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_proba, dtype=float)

    # 1. Full ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_t, y_p)
    roc_idx = downsample_curve_arc_length(fpr, tpr, max_points=max_points)

    roc_points = [
        {
            "fpr": float(round(float(fpr[i]), 5)),
            "tpr": float(round(float(tpr[i]), 5)),
            "threshold": float(round(float(roc_thresholds[i]), 5)),
        }
        for i in roc_idx
    ]

    # 2. Full PR curve
    prec, rec, pr_thresholds = precision_recall_curve(y_t, y_p)
    # pr_thresholds has length len(prec) - 1; pad last threshold with 1.0 for alignment
    pr_thresh_padded = np.append(pr_thresholds, 1.0)
    pr_idx = downsample_curve_arc_length(rec, prec, max_points=max_points)

    pr_points = [
        {
            "recall": float(round(float(rec[i]), 5)),
            "precision": float(round(float(prec[i]), 5)),
            "threshold": float(round(float(pr_thresh_padded[i]), 5)),
        }
        for i in pr_idx
    ]

    return {"roc": roc_points, "pr": pr_points}


def build_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute quantile calibration bins on repeat-0 predictions (TRD §4.4).

    Parameters
    ----------
    y_true, y_proba : np.ndarray
    n_bins : int, default=10

    Returns
    -------
    list of dict
        Quantile bins with mean_pred, frac_pos, n, lower, upper.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_proba, dtype=float)

    # Quantile bin edges with duplicates dropped
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(y_p, quantiles))

    # Fallback to uniform if unique probabilities are fewer than bins
    if len(bin_edges) - 1 < 2:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[dict[str, Any]] = []

    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            mask = (y_p >= lower) & (y_p <= upper)
        else:
            mask = (y_p >= lower) & (y_p < upper)

        n_samples = int(np.sum(mask))
        if n_samples > 0:
            mean_pred = float(np.mean(y_p[mask]))
            frac_pos = float(np.mean(y_t[mask]))
        else:
            mean_pred = float((lower + upper) / 2.0)
            frac_pos = 0.0

        bins.append({
            "bin": int(i),
            "mean_pred": float(round(mean_pred, 5)),
            "frac_pos": float(round(frac_pos, 5)),
            "n": n_samples,
            "lower": float(round(float(lower), 5)),
            "upper": float(round(float(upper), 5)),
        })

    return bins
