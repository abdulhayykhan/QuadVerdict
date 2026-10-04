"""
nested_cv.py — Deterministic, checkpointed nested cross-validation for QuadVerdict.

Design rules (TRD §4.3, PRD §7):
1. Outer CV: RepeatedStratifiedKFold(5 splits x 3 repeats = 15 folds, random_state=SEED).
   Quick mode: RepeatedStratifiedKFold(2 splits x 1 repeat = 2 folds, random_state=SEED).
2. Inner CV: StratifiedKFold(3 splits, shuffle=True, random_state=SEED) inside
   RandomizedSearchCV(scoring="average_precision", refit=True, random_state=SEED).
3. Leakage isolation: Outer-test data is NEVER visible to inner search or refit.
4. Checkpointing: Each completed (model, fold) is cached under results/_cache/.
   The --resume flag loads existing fold results to prevent wasted compute.
5. Determinism: All splits and search iterations strictly derive from config.SEED.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from src.config import (
    CACHE_DIR,
    INNER_N_SPLITS,
    N_ITER,
    OUTER_N_REPEATS,
    OUTER_N_SPLITS,
    QUICK_N_ITER,
    QUICK_OUTER_N_REPEATS,
    QUICK_OUTER_N_SPLITS,
    QUICK_SUBSAMPLE,
    SEED,
)
from src.pipelines import build_pipeline, get_param_distributions


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = 21,
) -> float:
    """Find the threshold in [0.1, 0.9] that maximizes F1 on validation/training data."""
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh


def evaluate_outer_fold(
    model_key: str,
    fold_idx: int,
    repeat_idx: int,
    X_train_fold: pd.DataFrame | np.ndarray,
    y_train_fold: pd.Series | np.ndarray,
    X_test_fold: pd.DataFrame | np.ndarray,
    y_test_fold: pd.Series | np.ndarray,
    test_indices: list[int] | np.ndarray,
    imbalance_strategy: str = "balanced",
    n_iter: int = 20,
    seed: int = SEED,
    cache_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute hyperparameter tuning on train fold and evaluate on outer test fold.

    Parameters
    ----------
    model_key : {'lr', 'dt', 'rf', 'svm'}
    fold_idx : int
        0-indexed outer fold identifier across all repeats.
    repeat_idx : int
        0-indexed repeat index.
    X_train_fold, y_train_fold : training fold inputs.
    X_test_fold, y_test_fold : test fold inputs (untouched until refitted evaluation).
    test_indices : original sample indices of the test fold.
    imbalance_strategy : {'none', 'balanced', 'smote'}, default='balanced'
    n_iter : int
        Number of RandomizedSearchCV parameter samples.
    seed : int
    cache_dir : Path, optional
    resume : bool

    Returns
    -------
    dict[str, Any]
        Per-fold evaluation metrics and predictions.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{model_key}_fold_{fold_idx}.joblib"
        if resume and cache_file.exists():
            try:
                cached_result = joblib.load(cache_file)
                return cached_result
            except Exception:
                pass  # Recompute if cache is corrupted

    t0 = time.perf_counter()

    # 1. Build pipeline and inner CV
    pipe = build_pipeline(model_key, imbalance_strategy=imbalance_strategy)
    param_dist = get_param_distributions(model_key)
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)

    # 2. Inner RandomizedSearchCV (scoring='average_precision', refit=True)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=inner_cv,
        refit=True,
        random_state=seed,
        n_jobs=1,  # Single-process per search to avoid oversubscription
    )

    search.fit(X_train_fold, y_train_fold)
    fit_time_s = time.perf_counter() - t0

    best_estimator = search.best_estimator_
    best_params = search.best_params_

    # 3. Outer test prediction
    y_test_arr = np.asarray(y_test_fold).astype(int)
    test_proba = best_estimator.predict_proba(X_test_fold)
    p_default = test_proba[:, 1]

    # Compute threshold on training fold predictions
    train_proba = best_estimator.predict_proba(X_train_fold)[:, 1]
    y_train_arr = np.asarray(y_train_fold).astype(int)
    opt_threshold = find_optimal_threshold(y_train_arr, train_proba)

    # Outer test metrics
    pr_auc = float(average_precision_score(y_test_arr, p_default))
    roc_auc = float(roc_auc_score(y_test_arr, p_default))
    brier = float(brier_score_loss(y_test_arr, p_default))

    preds_at_thresh = (p_default >= opt_threshold).astype(int)
    f1 = float(f1_score(y_test_arr, preds_at_thresh, zero_division=0))
    mcc = float(matthews_corrcoef(y_test_arr, preds_at_thresh))
    bal_acc = float(balanced_accuracy_score(y_test_arr, preds_at_thresh))
    accuracy = float(accuracy_score(y_test_arr, preds_at_thresh))

    result = {
        "model_key": model_key,
        "fold_idx": int(fold_idx),
        "repeat_idx": int(repeat_idx),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "f1": f1,
        "mcc": mcc,
        "bal_acc": bal_acc,
        "balanced_accuracy": bal_acc,
        "accuracy": accuracy,
        "optimal_threshold": opt_threshold,
        "best_params": best_params,
        "fit_time_s": float(fit_time_s),
        "test_indices": [int(idx) for idx in test_indices],
        "y_true": [int(v) for v in y_test_arr],
        "y_proba": [float(p) for p in p_default],
    }

    if cache_file is not None:
        joblib.dump(result, cache_file)

    return result


def run_nested_cv(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    models: list[str] | None = None,
    imbalance_strategy: str = "balanced",
    quick: bool = False,
    resume: bool = False,
    seed: int = SEED,
    cache_dir: Path | str = CACHE_DIR,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Execute complete nested cross-validation across all models and folds.

    Parameters
    ----------
    X_dev : pd.DataFrame
        Development set features (80% dev split).
    y_dev : pd.Series
        Development set binary labels.
    models : list of str, optional
        Subset of models to run, e.g. ['lr', 'dt', 'rf', 'svm']. Default runs all 4.
    imbalance_strategy : {'none', 'balanced', 'smote'}, default='balanced'
    quick : bool, default=False
        If True, use quick-mode overrides (2 folds, 2000 subsampled rows, n_iter=3).
    resume : bool, default=False
        If True, load already-cached folds from cache_dir.
    seed : int, default=SEED (42)
    cache_dir : Path or str, default=CACHE_DIR
    n_jobs : int, default=1
        Outer fold parallelism.

    Returns
    -------
    dict[str, Any]
        'models': mapping of model_key to list of per-fold result dicts.
        'meta': configuration and runtime metadata.
    """
    if models is None:
        models = ["lr", "dt", "rf", "svm"]

    valid_models = ("lr", "dt", "rf", "svm")
    for m in models:
        if m not in valid_models:
            raise ValueError(f"Unknown model '{m}'. Expected subset of {valid_models}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup budget and data size
    if quick:
        n_splits = QUICK_OUTER_N_SPLITS
        n_repeats = QUICK_OUTER_N_REPEATS
        iter_budget = QUICK_N_ITER

        # Stratified subsample for quick mode
        if len(X_dev) > QUICK_SUBSAMPLE:
            X_run, _, y_run, _ = train_test_split(
                X_dev,
                y_dev,
                train_size=QUICK_SUBSAMPLE,
                stratify=y_dev,
                random_state=seed,
            )
        else:
            X_run, y_run = X_dev, y_dev
    else:
        n_splits = OUTER_N_SPLITS
        n_repeats = OUTER_N_REPEATS
        iter_budget = N_ITER
        X_run, y_run = X_dev, y_dev

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    # Collect all fold tasks
    folds_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_run, y_run)):
        repeat_idx = fold_idx // n_splits
        X_train_fold = X_run.iloc[train_idx]
        y_train_fold = y_run.iloc[train_idx]
        X_test_fold = X_run.iloc[test_idx]
        y_test_fold = y_run.iloc[test_idx]
        test_indices = X_run.index[test_idx]

        folds_data.append((
            fold_idx,
            repeat_idx,
            X_train_fold,
            y_train_fold,
            X_test_fold,
            y_test_fold,
            test_indices,
        ))

    results_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in models}

    for model_key in models:
        model_n_iter = iter_budget[model_key]

        if n_jobs == 1:
            # Sequential execution
            for f_data in folds_data:
                res = evaluate_outer_fold(
                    model_key=model_key,
                    fold_idx=f_data[0],
                    repeat_idx=f_data[1],
                    X_train_fold=f_data[2],
                    y_train_fold=f_data[3],
                    X_test_fold=f_data[4],
                    y_test_fold=f_data[5],
                    test_indices=f_data[6],
                    imbalance_strategy=imbalance_strategy,
                    n_iter=model_n_iter,
                    seed=seed,
                    cache_dir=cache_dir,
                    resume=resume,
                )
                results_by_model[model_key].append(res)
        else:
            # Parallel outer fold execution
            delayed_tasks = [
                joblib.delayed(evaluate_outer_fold)(
                    model_key=model_key,
                    fold_idx=f_data[0],
                    repeat_idx=f_data[1],
                    X_train_fold=f_data[2],
                    y_train_fold=f_data[3],
                    X_test_fold=f_data[4],
                    y_test_fold=f_data[5],
                    test_indices=f_data[6],
                    imbalance_strategy=imbalance_strategy,
                    n_iter=model_n_iter,
                    seed=seed,
                    cache_dir=cache_dir,
                    resume=resume,
                )
                for f_data in folds_data
            ]
            fold_results = joblib.Parallel(n_jobs=n_jobs)(delayed_tasks)
            # Sort by fold_idx to maintain order
            fold_results.sort(key=lambda r: r["fold_idx"])
            results_by_model[model_key] = fold_results

    output = {
        "models": results_by_model,
        "meta": {
            "seed": int(seed),
            "quick": bool(quick),
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "total_folds": int(n_splits * n_repeats),
            "n_samples": int(len(X_run)),
            "imbalance_strategy": str(imbalance_strategy),
        },
    }
    return output


def summarize_nested_cv_results(cv_output: dict[str, Any]) -> pd.DataFrame:
    """Summarize nested CV results into a clean comparison DataFrame.

    Parameters
    ----------
    cv_output : dict
        Output from run_nested_cv().

    Returns
    -------
    pd.DataFrame
        Table with PR-AUC, ROC-AUC, Brier, F1, MCC, fit_time means and stds.
    """
    rows = []
    models_dict = cv_output.get("models", {})

    for model_key, folds in models_dict.items():
        pr_aucs = [f["pr_auc"] for f in folds]
        roc_aucs = [f["roc_auc"] for f in folds]
        briers = [f["brier"] for f in folds]
        f1s = [f["f1"] for f in folds]
        mccs = [f["mcc"] for f in folds]
        times = [f["fit_time_s"] for f in folds]

        rows.append({
            "model": model_key,
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs, ddof=1)) if len(pr_aucs) > 1 else 0.0,
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "roc_auc_std": float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0,
            "brier_mean": float(np.mean(briers)),
            "brier_std": float(np.std(briers, ddof=1)) if len(briers) > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)),
            "mcc_mean": float(np.mean(mccs)),
            "fit_time_s_mean": float(np.mean(times)),
            "n_folds": len(folds),
        })

    summary_df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    return summary_df
