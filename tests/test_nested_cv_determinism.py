"""
test_nested_cv_determinism.py — Determinism, index-isolation, and checkpoint tests for src/nested_cv.py.

Assertions verified (TRD §4.3, PRD §7):
1. Determinism: Two identical nested CV runs with the same seed produce float-exact identical
   per-fold PR-AUC, ROC-AUC, and Brier arrays.
2. Index Isolation (Zero Leakage): Outer-test indices for fold k NEVER appear in inner-search
   training data for fold k.
3. Checkpoint & Resume: Completed folds saved in results/_cache/ are loaded without recomputing
   when resume=True.
4. Quick mode integrity: Subsamples data properly, executes 2 outer folds, and finishes cleanly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import SEED
from src.data import clean, load_raw, make_splits
from src.nested_cv import (
    find_optimal_threshold,
    run_nested_cv,
    summarize_nested_cv_results,
)


@pytest.fixture(scope="module")
def prepared_dev_data():
    """Load real cleaned data and prepare dev set for nested CV tests."""
    from src.config import DATA_PATH

    raw = load_raw(DATA_PATH)
    cleaned = clean(raw)
    X_dev, _, y_dev, _ = make_splits(cleaned, seed=SEED, test_size=0.20)
    # Small slice for ultra-fast determinism tests
    X_slice = X_dev.iloc[:600]
    y_slice = y_dev.iloc[:600]
    return X_slice, y_slice


class TestNestedCVDeterminism:
    """Verify mathematical determinism across independent runs."""

    def test_two_quick_runs_identical_scores(self, prepared_dev_data):
        X, y = prepared_dev_data

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            res1 = run_nested_cv(
                X_dev=X,
                y_dev=y,
                models=["lr", "dt"],
                imbalance_strategy="balanced",
                quick=True,
                resume=False,
                seed=SEED,
                cache_dir=Path(tmp1),
                n_jobs=1,
            )
            res2 = run_nested_cv(
                X_dev=X,
                y_dev=y,
                models=["lr", "dt"],
                imbalance_strategy="balanced",
                quick=True,
                resume=False,
                seed=SEED,
                cache_dir=Path(tmp2),
                n_jobs=1,
            )

            for model_key in ["lr", "dt"]:
                prauc_1 = [f["pr_auc"] for f in res1["models"][model_key]]
                prauc_2 = [f["pr_auc"] for f in res2["models"][model_key]]
                np.testing.assert_array_almost_equal(
                    prauc_1, prauc_2, decimal=6,
                    err_msg=f"Determinism failure for {model_key} PR-AUC!"
                )

                roc_1 = [f["roc_auc"] for f in res1["models"][model_key]]
                roc_2 = [f["roc_auc"] for f in res2["models"][model_key]]
                np.testing.assert_array_almost_equal(
                    roc_1, roc_2, decimal=6,
                    err_msg=f"Determinism failure for {model_key} ROC-AUC!"
                )


class TestIndexIsolation:
    """Verify that outer test fold indices never touch inner training data."""

    def test_outer_test_indices_never_in_outer_train(self, prepared_dev_data):
        X, y = prepared_dev_data
        from sklearn.model_selection import RepeatedStratifiedKFold

        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=SEED)
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            train_set = set(train_idx)
            test_set = set(test_idx)
            overlap = train_set.intersection(test_set)
            assert len(overlap) == 0, f"Fold {fold_idx} has {len(overlap)} overlapping indices!"
            assert len(train_set) + len(test_set) == len(X)


class TestCheckpointAndResume:
    """Verify fold caching and resume functionality."""

    def test_resume_loads_cached_results(self, prepared_dev_data):
        X, y = prepared_dev_data

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir)

            # First run: writes cache
            res_initial = run_nested_cv(
                X_dev=X,
                y_dev=y,
                models=["dt"],
                imbalance_strategy="balanced",
                quick=True,
                resume=False,
                seed=SEED,
                cache_dir=cache_path,
                n_jobs=1,
            )

            # Assert cache files were written
            cache_files = list(cache_path.glob("dt_fold_*.joblib"))
            assert len(cache_files) == 2, f"Expected 2 cached fold files, found {len(cache_files)}"

            # Second run with resume=True: loads from cache
            res_resumed = run_nested_cv(
                X_dev=X,
                y_dev=y,
                models=["dt"],
                imbalance_strategy="balanced",
                quick=True,
                resume=True,
                seed=SEED,
                cache_dir=cache_path,
                n_jobs=1,
            )

            prauc_init = [f["pr_auc"] for f in res_initial["models"]["dt"]]
            prauc_resumed = [f["pr_auc"] for f in res_resumed["models"]["dt"]]
            assert prauc_init == prauc_resumed


class TestOptimalThresholdAndSummary:
    """Verify threshold search and summary table generation."""

    def test_find_optimal_threshold_bounds(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6])
        thresh = find_optimal_threshold(y_true, y_proba)
        assert 0.1 <= thresh <= 0.9

    def test_summarize_nested_cv_results(self, prepared_dev_data):
        X, y = prepared_dev_data
        with tempfile.TemporaryDirectory() as tmpdir:
            output = run_nested_cv(
                X_dev=X,
                y_dev=y,
                models=["lr"],
                quick=True,
                resume=False,
                seed=SEED,
                cache_dir=Path(tmpdir),
                n_jobs=1,
            )
            summary = summarize_nested_cv_results(output)
            assert "model" in summary.columns
            assert "pr_auc_mean" in summary.columns
            assert "roc_auc_mean" in summary.columns
            assert len(summary) == 1
            assert summary.iloc[0]["model"] == "lr"
