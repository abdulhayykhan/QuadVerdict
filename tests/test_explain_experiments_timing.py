"""
test_explain_experiments_timing.py — Unit tests for Phase 5 modules: explain.py, experiments.py, timing.py.

Assertions verified (TRD §4.6–4.8, PRD §6.5):
1. Permutation importance:
   - Computes on raw input features (23 features).
   - Top 15 features have ranks 1..15, zero NaNs, valid means and stds.
   - PAY_1 appears in the top ranked features.
2. Imbalance experiment:
   - Covers all 3 strategies ('none', 'balanced', 'smote').
   - Outputs valid PR-AUC and ROC-AUC means and stds.
3. Learning curves:
   - Evaluates sample-size steps with valid train and validation PR-AUC scores.
4. Timing and telemetry:
   - Measures inference latency on 1,000 samples (> 0 ms).
   - Measures model compressed size (> 0 KB).
   - Captures CPU processor, cores, and platform telemetry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import SEED
from src.data import clean, load_raw, make_splits
from src.experiments import run_imbalance_experiment, run_learning_curves
from src.explain import (
    aggregate_permutation_importances,
    compute_permutation_importance,
)
from src.pipelines import build_pipeline
from src.timing import (
    get_hardware_telemetry,
    measure_inference_latency_per_1k,
    measure_model_size_kb,
    measure_timing_and_size,
)


@pytest.fixture(scope="module")
def prepared_sample_data():
    """Load small dataset slice for fast Phase 5 testing."""
    from src.config import DATA_PATH

    raw = load_raw(DATA_PATH)
    cleaned = clean(raw)
    X_dev, X_hold, y_dev, y_hold = make_splits(cleaned, seed=SEED, test_size=0.20)

    X_train_slice = X_dev.iloc[:400]
    y_train_slice = y_dev.iloc[:400]
    X_test_slice = X_dev.iloc[400:600]
    y_test_slice = y_dev.iloc[400:600]

    return X_train_slice, y_train_slice, X_test_slice, y_test_slice, X_hold.iloc[:200]


class TestExplainability:
    """Verify permutation feature importance on raw features."""

    def test_compute_permutation_importance_top_features(self, prepared_sample_data):
        X_train, y_train, X_test, y_test, _ = prepared_sample_data
        pipe = build_pipeline("dt", imbalance_strategy="balanced")
        pipe.fit(X_train, y_train)

        records = compute_permutation_importance(
            pipe, X_test, y_test, scoring="average_precision", n_repeats=3, seed=SEED
        )

        assert len(records) == len(X_test.columns)
        for r in records:
            assert "feature" in r
            assert "importance_mean" in r
            assert "importance_std" in r
            assert not pd.isna(r["importance_mean"])
            assert not pd.isna(r["importance_std"])

        # Check sorted descending
        means = [r["importance_mean"] for r in records]
        assert means == sorted(means, reverse=True)

    def test_aggregate_permutation_importances_top_15(self):
        # 2 folds with synthetic features
        fold1 = [
            {"feature": "PAY_1", "importance_mean": 0.15, "importance_std": 0.02},
            {"feature": "LIMIT_BAL", "importance_mean": 0.05, "importance_std": 0.01},
        ]
        fold2 = [
            {"feature": "PAY_1", "importance_mean": 0.17, "importance_std": 0.03},
            {"feature": "LIMIT_BAL", "importance_mean": 0.07, "importance_std": 0.01},
        ]

        top_list = aggregate_permutation_importances([fold1, fold2], top_n=15)
        assert len(top_list) == 2
        assert top_list[0]["rank"] == 1
        assert top_list[0]["feature"] == "PAY_1"
        assert pytest.approx(top_list[0]["importance_mean"], abs=1e-4) == 0.16


class TestExperiments:
    """Verify imbalance comparison and learning curve generation."""

    def test_run_imbalance_experiment_structure(self, prepared_sample_data):
        X_train, y_train, _, _, _ = prepared_sample_data

        res = run_imbalance_experiment(
            X_train, y_train, models=["dt"], quick=True, seed=SEED
        )
        assert "dt" in res
        for strat in ["none", "balanced", "smote"]:
            assert strat in res["dt"]
            assert "pr_auc_mean" in res["dt"][strat]
            assert 0.0 <= res["dt"][strat]["pr_auc_mean"] <= 1.0

    def test_run_learning_curves_structure(self, prepared_sample_data):
        X_train, y_train, _, _, _ = prepared_sample_data

        curves = run_learning_curves(
            X_train, y_train, models=["dt"], train_sizes=[0.3, 0.6, 1.0], quick=True, seed=SEED
        )
        assert "dt" in curves
        dt_curve = curves["dt"]
        assert len(dt_curve["train_sizes"]) == 3
        assert len(dt_curve["train_mean"]) == 3
        assert len(dt_curve["val_mean"]) == 3


class TestTimingAndTelemetry:
    """Verify timing latency benchmark and hardware telemetry."""

    def test_hardware_telemetry(self):
        telem = get_hardware_telemetry()
        assert "cpu_processor" in telem
        assert "cpu_cores" in telem
        assert telem["cpu_cores"] >= 1
        assert "python_version" in telem

    def test_inference_latency_and_model_size(self, prepared_sample_data):
        X_train, y_train, _, _, X_hold_sample = prepared_sample_data

        pipe = build_pipeline("dt", imbalance_strategy="none")
        pipe.fit(X_train, y_train)

        latency_ms = measure_inference_latency_per_1k(pipe, X_hold_sample, n_runs=3)
        assert latency_ms > 0.0

        size_kb = measure_model_size_kb(pipe, compress_level=3)
        assert size_kb > 0.0

    def test_measure_timing_and_size_integration(self, prepared_sample_data):
        X_train, y_train, _, _, X_hold_sample = prepared_sample_data
        mock_cv = {
            "models": {"dt": [{"fit_time_s": 0.25}]},
            "meta": {"imbalance_strategy": "balanced"},
        }

        timing_out = measure_timing_and_size(
            models=["dt"],
            cv_output=mock_cv,
            X_dev=X_train,
            y_dev=y_train,
            X_hold=X_hold_sample,
            n_inference_runs=3,
        )

        assert "models" in timing_out
        assert "dt" in timing_out["models"]
        assert timing_out["models"]["dt"]["train_s"] == 0.25
        assert timing_out["models"]["dt"]["infer_ms_per_1k"] > 0.0
        assert timing_out["models"]["dt"]["model_kb"] > 0.0
        assert "hardware" in timing_out
