"""
explain.py — Permutation feature importance across outer folds for QuadVerdict.

Design rules (TRD §4.6, PRD §6.5):
1. Permutation importance is evaluated on outer test folds with each fold's fitted best estimator.
2. Scoring is strictly 'average_precision' (matching the primary benchmark metric).
3. Evaluated on RAW input features (before one-hot encoding) through the full pipeline,
   ensuring feature importance is directly comparable across models.
4. Aggregates mean and std across all outer folds and exports the top 15 features per model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import SEED
from src.pipelines import build_pipeline


def compute_permutation_importance(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    scoring: str = "average_precision",
    n_repeats: int = 5,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Compute permutation importance on raw DataFrame features for a fitted pipeline.

    Parameters
    ----------
    estimator : fitted Pipeline
    X_test : pd.DataFrame of shape (n_samples, n_features)
    y_test : array-like of shape (n_samples,)
    scoring : str, default='average_precision'
    n_repeats : int, default=5
    seed : int, default=SEED (42)

    Returns
    -------
    list of dict
        List of {'feature': str, 'importance_mean': float, 'importance_std': float}
        sorted descending by importance_mean.
    """
    res = permutation_importance(
        estimator=estimator,
        X=X_test,
        y=y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
    )

    feature_names = list(X_test.columns)
    records = []
    for i, col in enumerate(feature_names):
        mean_val = float(res.importances_mean[i])
        std_val = float(res.importances_std[i])
        records.append({
            "feature": col,
            "importance_mean": float(round(mean_val, 5)),
            "importance_std": float(round(std_val, 5)),
        })

    records.sort(key=lambda x: x["importance_mean"], reverse=True)
    return records


def aggregate_permutation_importances(
    fold_importances: list[list[dict[str, Any]]],
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Aggregate permutation importances across all outer folds.

    Parameters
    ----------
    fold_importances : list of list of dict
        List of fold results, where each element is a list of per-feature records.
    top_n : int, default=15

    Returns
    -------
    list of dict
        Top N features sorted by mean importance with rank, feature, importance_mean, importance_std.
    """
    if not fold_importances:
        return []

    # Map feature -> list of means and stds across folds
    feat_means: dict[str, list[float]] = {}

    for fold_list in fold_importances:
        for item in fold_list:
            feat = item["feature"]
            if feat not in feat_means:
                feat_means[feat] = []
            feat_means[feat].append(item["importance_mean"])

    aggregated = []
    for feat, means in feat_means.items():
        grand_mean = float(np.mean(means))
        grand_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        aggregated.append({
            "feature": feat,
            "importance_mean": float(round(grand_mean, 5)),
            "importance_std": float(round(grand_std, 5)),
        })

    aggregated.sort(key=lambda x: x["importance_mean"], reverse=True)

    # Assign ranks and slice top_n
    top_records = []
    for rank_idx, record in enumerate(aggregated[:top_n], start=1):
        top_records.append({
            "rank": rank_idx,
            **record,
        })

    return top_records


def run_explainability(
    models: list[str],
    cv_output: dict[str, Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_repeats: int = 5,
    seed: int = SEED,
    top_n: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """Run permutation importance across outer folds for all benchmark models.

    Parameters
    ----------
    models : list of str
    cv_output : dict from run_nested_cv()
    X_dev, y_dev : development set data
    n_repeats : int, default=5
    seed : int
    top_n : int, default=15

    Returns
    -------
    dict[str, list[dict]]
        Mapping of model_key -> top 15 features with importance means and stds.
    """
    output: dict[str, list[dict[str, Any]]] = {}
    models_dict = cv_output.get("models", {})
    imbalance_strategy = cv_output.get("meta", {}).get("imbalance_strategy", "balanced")

    for model_key in models:
        if model_key not in models_dict:
            continue

        folds = models_dict[model_key]
        fold_importances = []

        for f in folds:
            test_indices = f["test_indices"]
            best_params = f["best_params"]

            # Partition into train and test fold
            mask_test = X_dev.index.isin(test_indices)
            X_train_f = X_dev.loc[~mask_test]
            y_train_f = y_dev.loc[~mask_test]
            X_test_f = X_dev.loc[mask_test]
            y_test_f = y_dev.loc[mask_test]

            # Fit pipeline with best params on training fold
            pipe = build_pipeline(model_key, imbalance_strategy=imbalance_strategy)
            pipe.set_params(**best_params)
            pipe.fit(X_train_f, y_train_f)

            imp = compute_permutation_importance(
                estimator=pipe,
                X_test=X_test_f,
                y_test=y_test_f,
                scoring="average_precision",
                n_repeats=n_repeats,
                seed=seed,
            )
            fold_importances.append(imp)

        output[model_key] = aggregate_permutation_importances(fold_importances, top_n=top_n)

    return output
