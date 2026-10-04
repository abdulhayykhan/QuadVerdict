"""
test_metrics.py — Unit tests for src/metrics.py (Phase 4).

Assertions verified (TRD §4.4, PRD §6.2–6.4):
1. Threshold table against hand-computed tiny arrays with known TP/FP/TN/FN.
2. Invariants:
   - tp + fn == n_pos for every threshold row.
   - fp + tn == n_neg for every threshold row.
   - tp and fp are monotonically non-increasing in threshold t.
3. Edge cases:
   - Precision edge case: tp + fp == 0 -> 1.0.
   - F1 edge case: 2*tp + fp + fn == 0 -> 0.0.
   - MCC edge case: zero denominator -> 0.0 (no exception or NaN).
4. Arc-length curve downsampling:
   - Downsamples to <= 200 points.
   - Strict preservation of curve endpoints.
5. Quantile calibration bins:
   - Quantile partitions, sum of n equals total samples.
   - Fractions and predictions within [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    build_calibration,
    build_roc_pr,
    build_threshold_table,
    compute_derived_threshold_metrics,
    compute_scalar_metrics,
    downsample_curve_arc_length,
)


class TestHandComputedMetrics:
    """Verify metrics match hand-calculated numbers on a tiny toy dataset."""

    def test_derived_threshold_metrics_hand_computed(self):
        # TP=2, FP=1, TN=6, FN=1 (Total = 10; n_pos = 3, n_neg = 7)
        # Precision = 2 / 3 = 0.66667
        # Recall    = 2 / 3 = 0.66667
        # F1        = 2*2 / (4 + 1 + 1) = 4 / 6 = 0.66667
        # MCC       = (2*6 - 1*1) / sqrt((3)*(3)*(7)*(7)) = 11 / 21 = 0.52381
        # Specificity = 6 / 7 = 0.85714
        # BalAcc    = (2/3 + 6/7) / 2 = (14/21 + 18/21) / 2 = 32 / 42 = 0.76190
        # Cost (c_fn=5, c_fp=1) = 5*1 + 1*1 = 6.0
        # Cost per 1k = 6 / 10 * 1000 = 600.0
        metrics = compute_derived_threshold_metrics(tp=2, fp=1, tn=6, fn=1, cost_ratio=5)

        assert pytest.approx(metrics["precision"], rel=1e-4) == 2 / 3
        assert pytest.approx(metrics["recall"], rel=1e-4) == 2 / 3
        assert pytest.approx(metrics["f1"], rel=1e-4) == 2 / 3
        assert pytest.approx(metrics["mcc"], rel=1e-4) == 11 / 21
        assert pytest.approx(metrics["balanced_accuracy"], rel=1e-4) == 16 / 21
        assert pytest.approx(metrics["cost"], rel=1e-4) == 6.0
        assert pytest.approx(metrics["cost_per_1k"], rel=1e-4) == 600.0

    def test_compute_scalar_metrics_hand_computed(self):
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.9, 0.8, 0.3, 0.1])
        # At threshold 0.5:
        # Preds: [1, 1, 0, 0]
        # TP = 1 (idx 0), FP = 1 (idx 1), FN = 1 (idx 2), TN = 1 (idx 3)
        res = compute_scalar_metrics(y_true, y_proba, threshold=0.5)

        assert res["tp"] == 1.0
        assert res["fp"] == 1.0
        assert res["tn"] == 1.0
        assert res["fn"] == 1.0
        assert res["precision"] == 0.5
        assert res["recall"] == 0.5
        assert res["f1"] == 0.5
        assert res["mcc"] == 0.0
        assert res["balanced_accuracy"] == 0.5


class TestEdgeCases:
    """Verify zero-division handling for extreme predictions."""

    def test_precision_zero_positive_predictions(self):
        # No samples predicted positive -> tp + fp == 0
        metrics = compute_derived_threshold_metrics(tp=0, fp=0, tn=8, fn=2)
        assert metrics["precision"] == 1.0  # Defined as 1.0 per TRD §4.4
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_mcc_zero_denominator(self):
        # All samples predicted negative (tp=0, fp=0)
        metrics = compute_derived_threshold_metrics(tp=0, fp=0, tn=10, fn=0)
        assert metrics["mcc"] == 0.0
        assert not np.isnan(metrics["mcc"])

    def test_f1_zero_denominator(self):
        metrics = compute_derived_threshold_metrics(tp=0, fp=0, tn=10, fn=0)
        assert metrics["f1"] == 0.0


class TestThresholdTableInvariants:
    """Verify threshold table construction and strict invariants (TRD §4.4)."""

    @pytest.fixture
    def mock_fold_results(self):
        # 2 repeats, 2 folds per repeat, 4 samples total (0, 1, 2, 3)
        # N = 4 samples: 2 positive, 2 negative
        return [
            # Repeat 0
            {
                "repeat_idx": 0, "fold_idx": 0,
                "test_indices": [0, 1], "y_true": [1, 0], "y_proba": [0.85, 0.20],
            },
            {
                "repeat_idx": 0, "fold_idx": 1,
                "test_indices": [2, 3], "y_true": [1, 0], "y_proba": [0.65, 0.40],
            },
            # Repeat 1
            {
                "repeat_idx": 1, "fold_idx": 2,
                "test_indices": [0, 2], "y_true": [1, 1], "y_proba": [0.90, 0.60],
            },
            {
                "repeat_idx": 1, "fold_idx": 3,
                "test_indices": [1, 3], "y_true": [0, 0], "y_proba": [0.15, 0.35],
            },
        ]

    def test_table_shape_and_keys(self, mock_fold_results):
        table = build_threshold_table(
            mock_fold_results, n_splits=2, n_repeats=2, n_steps=101, cost_ratio=5
        )
        assert len(table) == 101
        first_row = table[0]
        expected_keys = {
            "threshold", "tp", "fp", "tn", "fn",
            "precision", "recall", "f1", "mcc", "balanced_accuracy", "cost", "cost_per_1k"
        }
        assert expected_keys.issubset(first_row.keys())

    def test_invariants_hold_across_all_thresholds(self, mock_fold_results):
        table = build_threshold_table(
            mock_fold_results, n_splits=2, n_repeats=2, n_steps=101, cost_ratio=5
        )
        n_pos = 2.0
        n_neg = 2.0

        prev_tp = float("inf")
        prev_fp = float("inf")

        for row in table:
            tp = row["tp"]
            fp = row["fp"]
            tn = row["tn"]
            fn = row["fn"]

            # Invariant 1: tp + fn == n_pos
            assert pytest.approx(tp + fn, abs=1e-4) == n_pos
            # Invariant 2: fp + tn == n_neg
            assert pytest.approx(fp + tn, abs=1e-4) == n_neg

            # Invariant 3: tp and fp are monotonically non-increasing in threshold
            assert tp <= prev_tp + 1e-6, f"TP increased from {prev_tp} to {tp} at t={row['threshold']}"
            assert fp <= prev_fp + 1e-6, f"FP increased from {prev_fp} to {fp} at t={row['threshold']}"

            prev_tp = tp
            prev_fp = fp


class TestCurveDownsamplingAndCalibration:
    """Verify arc-length downsampling and quantile calibration."""

    def test_downsample_curve_arc_length(self):
        # 1000 synthetic curve points
        x = np.linspace(0, 1, 1000)
        y = np.sqrt(x)

        idx = downsample_curve_arc_length(x, y, max_points=200)
        assert len(idx) <= 200
        # Endpoints must be strictly preserved
        assert idx[0] == 0
        assert idx[-1] == 999
        # Monotonic index sequence
        assert np.all(np.diff(idx) > 0)

    def test_build_roc_pr_structure(self):
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=500, p=[0.75, 0.25])
        y_proba = np.random.uniform(0, 1, size=500)

        curves = build_roc_pr(y_true, y_proba, max_points=100)
        assert "roc" in curves
        assert "pr" in curves
        assert len(curves["roc"]) <= 100
        assert len(curves["pr"]) <= 100

        first_roc = curves["roc"][0]
        assert "fpr" in first_roc
        assert "tpr" in first_roc
        assert "threshold" in first_roc

    def test_build_calibration_structure(self):
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=1000, p=[0.78, 0.22])
        y_proba = np.random.beta(a=2, b=5, size=1000)

        bins = build_calibration(y_true, y_proba, n_bins=10)
        assert len(bins) <= 10
        total_n = sum(b["n"] for b in bins)
        assert total_n == 1000

        for b in bins:
            assert 0.0 <= b["frac_pos"] <= 1.0
            assert 0.0 <= b["mean_pred"] <= 1.0
            assert b["lower"] <= b["upper"]
