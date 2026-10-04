"""
experiments.py — Imbalance handling experiments and learning curves for QuadVerdict.

Design rules (TRD §4.7, PRD §6.5):
1. Imbalance experiment:
   - Evaluates each model across all three strategies: 'none', 'balanced', 'smote'.
   - Runs cross-validation with a controlled budget to compare PR-AUC mean and std.
2. Learning curves:
   - Evaluates sample-efficiency using sklearn.model_selection.learning_curve.
   - Training size fractions: [0.1, 0.25, 0.5, 0.75, 1.0].
   - Scoring: 'average_precision' across 3-fold stratified cross-validation.
   - Reports train and validation PR-AUC mean and std at each step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

from src.config import MODEL_KEYS, SEED
from src.nested_cv import run_nested_cv
from src.pipelines import build_pipeline


def run_imbalance_experiment(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    models: list[str] | None = None,
    quick: bool = False,
    seed: int = SEED,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare performance across imbalance strategies: 'none', 'balanced', 'smote'.

    Parameters
    ----------
    X_dev, y_dev : pd.DataFrame, pd.Series
    models : list of str, optional
        Default is config.MODEL_KEYS.
    quick : bool, default=False
    seed : int, default=SEED

    Returns
    -------
    dict
        Mapping of model_key -> strategy -> {'pr_auc_mean', 'pr_auc_std', 'roc_auc_mean', 'roc_auc_std'}.
    """
    if models is None:
        models = MODEL_KEYS

    strategies = ["none", "balanced", "smote"]
    results: dict[str, dict[str, dict[str, float]]] = {m: {} for m in models}

    for strat in strategies:
        cv_out = run_nested_cv(
            X_dev=X_dev,
            y_dev=y_dev,
            models=models,
            imbalance_strategy=strat,
            quick=True if quick else True,  # Reduced budget per TRD §4.7
            resume=False,
            seed=seed,
            n_jobs=1,
        )

        for m in models:
            folds = cv_out["models"][m]
            pr_aucs = [f["pr_auc"] for f in folds]
            roc_aucs = [f["roc_auc"] for f in folds]

            results[m][strat] = {
                "pr_auc_mean": float(round(float(np.mean(pr_aucs)), 5)),
                "pr_auc_std": float(round(float(np.std(pr_aucs, ddof=1)) if len(pr_aucs) > 1 else 0.0, 5)),
                "roc_auc_mean": float(round(float(np.mean(roc_aucs)), 5)),
                "roc_auc_std": float(round(float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0, 5)),
            }

    return results


def run_learning_curves(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    models: list[str] | None = None,
    train_sizes: list[float] | None = None,
    cv_splits: int = 3,
    quick: bool = False,
    seed: int = SEED,
) -> dict[str, dict[str, list[Any]]]:
    """Generate empirical learning curves for benchmark classifiers (TRD §4.7).

    Parameters
    ----------
    X_dev, y_dev : pd.DataFrame, pd.Series
    models : list of str, optional
    train_sizes : list of float, optional
        Default [0.1, 0.25, 0.5, 0.75, 1.0].
    cv_splits : int, default=3
    quick : bool, default=False
    seed : int, default=SEED

    Returns
    -------
    dict
        Mapping of model_key -> {
            'train_sizes': list[int],
            'train_mean': list[float],
            'train_std': list[float],
            'val_mean': list[float],
            'val_std': list[float]
        }
    """
    if models is None:
        models = MODEL_KEYS

    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0] if not quick else [0.2, 0.5, 1.0]

    # Quick subsample for responsive curve calculation
    if quick and len(X_dev) > 1500:
        X_run, _, y_run, _ = train_test_split(
            X_dev, y_dev, train_size=1500, stratify=y_dev, random_state=seed
        )
    else:
        X_run, y_run = X_dev, y_dev

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    output: dict[str, dict[str, list[Any]]] = {}

    for model_key in models:
        pipe = build_pipeline(model_key, imbalance_strategy="balanced")

        sizes, train_scores, test_scores = learning_curve(
            estimator=pipe,
            X=X_run,
            y=y_run,
            train_sizes=train_sizes,
            cv=cv,
            scoring="average_precision",
            random_state=seed,
            n_jobs=1,
        )

        output[model_key] = {
            "train_sizes": [int(s) for s in sizes],
            "train_mean": [float(round(float(m), 5)) for m in np.mean(train_scores, axis=1)],
            "train_std": [float(round(float(s), 5)) for s in np.std(train_scores, axis=1)],
            "val_mean": [float(round(float(m), 5)) for m in np.mean(test_scores, axis=1)],
            "val_std": [float(round(float(s), 5)) for s in np.std(test_scores, axis=1)],
        }

    return output
