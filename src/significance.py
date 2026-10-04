"""
significance.py — Statistical significance testing for QuadVerdict.

Design rules (TRD §4.5, PRD §6.5):
1. Nadeau-Bengio corrected resampled t-test:
   Accounts for the dependence between folds due to overlapping training sets:
       t = mean(d) / sqrt((1/k + n_test/n_train) * var(d, ddof=1))
       p = 2 * (1 - t_cdf(|t|, df=k-1))
2. Wilcoxon signed-rank test (scipy.stats.wilcoxon):
   Non-parametric paired check.
3. Holm-Bonferroni correction:
   Applied step-down across all 6 pairwise model comparisons to control FWER.
4. Words like 'significant' are asserted strictly when p_adj < 0.05.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy import stats

from src.config import MODEL_KEYS


def nadeau_bengio_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    n_train: int,
    n_test: int,
) -> tuple[float, float]:
    """Calculate the Nadeau-Bengio corrected resampled t-test (TRD §4.5).

    Parameters
    ----------
    scores_a, scores_b : array-like of shape (k,)
        Paired performance metrics across the same k outer folds.
    n_train : int
        Size of outer training fold.
    n_test : int
        Size of outer test fold.

    Returns
    -------
    t_stat : float
    p_value : float (two-sided)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")

    k = len(a)
    if k < 2:
        return 0.0, 1.0

    d = a - b
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))

    # Guard against zero-variance crash (identical performance)
    if var_d <= 1e-15 or math.isnan(var_d):
        return 0.0, 1.0

    r = float(n_test) / float(n_train)
    denom = math.sqrt((1.0 / k + r) * var_d)

    if denom <= 1e-15:
        return 0.0, 1.0

    t_stat = mean_d / denom
    df = k - 1
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))

    return float(t_stat), float(p_val)


def wilcoxon_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
) -> tuple[float, float]:
    """Calculate Wilcoxon signed-rank test on paired per-fold scores.

    Parameters
    ----------
    scores_a, scores_b : array-like of shape (k,)

    Returns
    -------
    stat : float
    p_value : float (two-sided)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    d = a - b
    # If all differences are zero, return p=1.0
    if np.allclose(d, 0.0, atol=1e-12):
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def holm_bonferroni(p_values: list[float] | np.ndarray) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.

    Parameters
    ----------
    p_values : list or array of float

    Returns
    -------
    list of float
        Adjusted p-values, preserving original input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n <= 1:
        return [float(x) for x in p]

    order = np.argsort(p)
    sorted_p = p[order]

    adj_sorted = np.empty(n, dtype=float)
    running_max = 0.0

    for i in range(n):
        multiplier = n - i
        val = min(1.0, sorted_p[i] * multiplier)
        running_max = max(running_max, val)
        adj_sorted[i] = running_max

    adj = np.empty(n, dtype=float)
    adj[order] = adj_sorted
    return [float(round(val, 6)) for val in adj]


def run_pairwise_significance(
    cv_output: dict[str, Any],
    n_train: int,
    n_test: int,
    metric: str = "pr_auc",
) -> list[dict[str, Any]]:
    """Compute all pairwise statistical tests across models for the given metric.

    Parameters
    ----------
    cv_output : dict
        Output from run_nested_cv().
    n_train : int
        Number of training samples per outer fold.
    n_test : int
        Number of test samples per outer fold.
    metric : str, default='pr_auc'
        Metric to evaluate ('pr_auc', 'roc_auc', 'brier', etc.).

    Returns
    -------
    list of dict
        6 pairwise comparison records with test statistics and Holm-adjusted p-values.
    """
    models_dict = cv_output.get("models", {})
    available_models = [m for m in MODEL_KEYS if m in models_dict]

    # Generate all unique pairs (6 pairs for 4 models)
    pairs = list(itertools.combinations(available_models, 2))

    comparisons: list[dict[str, Any]] = []
    nb_p_values = []
    wilcoxon_p_values = []

    for model_a, model_b in pairs:
        scores_a = [f[metric] for f in models_dict[model_a]]
        scores_b = [f[metric] for f in models_dict[model_b]]

        mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
        t_stat, p_nb = nadeau_bengio_test(scores_a, scores_b, n_train=n_train, n_test=n_test)
        w_stat, p_wilc = wilcoxon_test(scores_a, scores_b)

        nb_p_values.append(p_nb)
        wilcoxon_p_values.append(p_wilc)

        comparisons.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "mean_a": float(round(float(np.mean(scores_a)), 5)),
            "mean_b": float(round(float(np.mean(scores_b)), 5)),
            "mean_diff": float(round(mean_diff, 5)),
            "t_stat": float(round(t_stat, 4)),
            "p_value_nb": float(round(p_nb, 6)),
            "wilcoxon_stat": float(round(w_stat, 4)),
            "p_value_wilcoxon": float(round(p_wilc, 6)),
        })

    # Apply Holm correction across all pairs
    adj_nb = holm_bonferroni(nb_p_values)
    adj_wilcox = holm_bonferroni(wilcoxon_p_values)

    for i, comp in enumerate(comparisons):
        p_adj_nb = adj_nb[i]
        comp["p_adj_nb"] = p_adj_nb
        comp["p_adj_wilcoxon"] = adj_wilcox[i]

        # Significance declaration: strictly p_adj < 0.05
        is_sig = p_adj_nb < 0.05
        comp["significant"] = is_sig

        if is_sig:
            comp["winner"] = comp["model_a"] if comp["mean_diff"] > 0 else comp["model_b"]
        else:
            comp["winner"] = "no_difference"

    return comparisons
