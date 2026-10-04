"""
test_significance.py — Unit tests for src/significance.py (Phase 4).

Assertions verified (TRD §4.5, PRD §6.5):
1. Nadeau-Bengio corrected t-test matches hand-computed reference within 1e-6.
2. Identical score vectors (all d_i = 0) -> t = 0.0, p = 1.0 (guards against zero-variance crash).
3. Wilcoxon signed-rank test matches scipy reference.
4. Holm-Bonferroni step-down correction matches manual calculations and enforces monotonicity.
5. Pairwise significance runner returns exactly 6 pairs for 4 models with strict p_adj < 0.05 winners.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.significance import (
    holm_bonferroni,
    nadeau_bengio_test,
    run_pairwise_significance,
    wilcoxon_test,
)


class TestNadeauBengioTest:
    """Verify Nadeau-Bengio corrected resampled t-test (TRD §4.5)."""

    def test_hand_computed_example(self):
        # Known synthetic fold differences
        # k = 5, n_train = 800, n_test = 200 -> r = 0.25
        # d = [0.05, 0.03, 0.04, 0.06, 0.02]
        # mean(d) = 0.04
        # var(d, ddof=1) = 0.00025
        # denom = sqrt((1/5 + 0.25) * 0.00025) = sqrt(0.45 * 0.00025) = sqrt(0.0001125) = 0.0106066
        # t = 0.04 / 0.0106066 = 3.771236
        # df = 4 -> p = 2 * (1 - t.cdf(3.771236, df=4)) = 0.019584
        scores_a = np.array([0.55, 0.53, 0.54, 0.56, 0.52])
        scores_b = np.array([0.50, 0.50, 0.50, 0.50, 0.50])

        t_stat, p_val = nadeau_bengio_test(scores_a, scores_b, n_train=800, n_test=200)

        assert pytest.approx(t_stat, abs=1e-5) == 3.771236
        assert pytest.approx(p_val, abs=1e-5) == 0.019584

    def test_identical_scores_zero_variance_guard(self):
        scores_a = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
        scores_b = np.array([0.50, 0.50, 0.50, 0.50, 0.50])

        t_stat, p_val = nadeau_bengio_test(scores_a, scores_b, n_train=800, n_test=200)
        assert t_stat == 0.0
        assert p_val == 1.0


class TestWilcoxonTest:
    """Verify non-parametric Wilcoxon signed-rank check."""

    def test_wilcoxon_matches_scipy(self):
        scores_a = np.array([0.55, 0.53, 0.54, 0.56, 0.52, 0.58])
        scores_b = np.array([0.50, 0.51, 0.49, 0.52, 0.50, 0.51])

        stat, p_val = wilcoxon_test(scores_a, scores_b)
        expected = stats.wilcoxon(scores_a - scores_b)

        assert pytest.approx(stat, rel=1e-5) == expected.statistic
        assert pytest.approx(p_val, rel=1e-5) == expected.pvalue

    def test_wilcoxon_identical_scores(self):
        scores_a = np.array([0.5, 0.5, 0.5])
        scores_b = np.array([0.5, 0.5, 0.5])

        stat, p_val = wilcoxon_test(scores_a, scores_b)
        assert stat == 0.0
        assert p_val == 1.0


class TestHolmBonferroni:
    """Verify Holm-Bonferroni FWER step-down correction."""

    def test_holm_known_vector(self):
        # Input p-values: [0.01, 0.04, 0.03]
        # Sorted: 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> running max = 0.06)
        # Expected adjusted: [0.03, 0.06, 0.06]
        raw_p = [0.01, 0.04, 0.03]
        adj = holm_bonferroni(raw_p)

        assert pytest.approx(adj[0], abs=1e-5) == 0.03
        assert pytest.approx(adj[1], abs=1e-5) == 0.06
        assert pytest.approx(adj[2], abs=1e-5) == 0.06

    def test_holm_caps_at_one(self):
        raw_p = [0.8, 0.9, 0.95]
        adj = holm_bonferroni(raw_p)
        assert all(p <= 1.0 for p in adj)

    def test_holm_single_value(self):
        assert holm_bonferroni([0.05]) == [0.05]


class TestPairwiseSignificanceRunner:
    """Verify all 6 pairwise comparisons for the 4 benchmark classifiers."""

    @pytest.fixture
    def mock_cv_output(self):
        np.random.seed(42)
        k = 15
        # Synthesize 15-fold PR-AUC distributions where RF > LR > DT > SVM
        return {
            "models": {
                "rf": [{"pr_auc": float(0.55 + np.random.normal(0, 0.01))} for _ in range(k)],
                "lr": [{"pr_auc": float(0.51 + np.random.normal(0, 0.01))} for _ in range(k)],
                "dt": [{"pr_auc": float(0.47 + np.random.normal(0, 0.01))} for _ in range(k)],
                "svm": [{"pr_auc": float(0.42 + np.random.normal(0, 0.01))} for _ in range(k)],
            }
        }

    def test_returns_exactly_six_pairs(self, mock_cv_output):
        comparisons = run_pairwise_significance(
            mock_cv_output, n_train=19200, n_test=4800, metric="pr_auc"
        )
        assert len(comparisons) == 6

        pairs = [(c["model_a"], c["model_b"]) for c in comparisons]
        expected_pairs = [
            ("lr", "dt"), ("lr", "rf"), ("lr", "svm"),
            ("dt", "rf"), ("dt", "svm"), ("rf", "svm"),
        ]
        assert pairs == expected_pairs

    def test_keys_and_significance_rule(self, mock_cv_output):
        comparisons = run_pairwise_significance(
            mock_cv_output, n_train=19200, n_test=4800, metric="pr_auc"
        )
        for c in comparisons:
            assert "p_adj_nb" in c
            assert "p_adj_wilcoxon" in c
            assert "winner" in c
            assert "significant" in c

            # Strict rule (TRD §4.5): significant ONLY when p_adj_nb < 0.05
            if c["p_adj_nb"] < 0.05:
                assert c["significant"] is True
                assert c["winner"] in (c["model_a"], c["model_b"])
            else:
                assert c["significant"] is False
                assert c["winner"] == "no_difference"
