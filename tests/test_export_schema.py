"""
test_export_schema.py — Schema validation and mutation tests for src/export.py.

Assertions verified (TRD §5, PRD §6.5):
1. Valid quick-mode and full-mode structures pass validate() cleanly.
2. Mutation tests verify that each violation raises SchemaValidationError:
   - Missing required model ('rf', 'svm', etc.)
   - Threshold table with != 101 rows
   - t not strictly increasing
   - tp + fn != n_pos
   - fp + tn != n_neg
   - tp or fp increasing with t
   - cv_scores length mismatch
   - Score values outside valid domain bounds
   - ROC/PR curves exceeding 200 points
   - Significance with != 6 pairwise comparisons
   - File size exceeding RESULTS_MAX_KB
3. Float rounding handles nested structures and guards against NaN/Inf.
4. evaluate_holdout_set produces valid scores on held-out data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import config
from src.export import (
    SchemaValidationError,
    assemble_results,
    evaluate_holdout_set,
    round_floats,
    validate,
    write,
)


def make_valid_mock_results(quick: bool = True) -> dict:
    """Generate a synthetically valid results dictionary conforming to TRD §5."""
    n_pos = 200
    n_neg = 800
    n_total = n_pos + n_neg
    cv_len = 2 if quick else 15

    models = {}
    for m in ["lr", "dt", "rf", "svm"]:
        # Generate 101 monotonic threshold rows
        t_vals = np.linspace(0.0, 1.0, 101)
        thresholds = []
        for t in t_vals:
            tp = float(round(n_pos * (1.0 - t), 4))
            fn = float(round(n_pos - tp, 4))
            fp = float(round(n_neg * (1.0 - t) * 0.8, 4))
            tn = float(round(n_neg - fp, 4))
            thresholds.append({"t": float(round(t, 2)), "tp": tp, "fp": fp, "tn": tn, "fn": fn})

        models[m] = {
            "label": m.upper(),
            "best_params": {"param": 1},
            "best_params_frequency": [{"params": {"param": 1}, "count": cv_len}],
            "cv_scores": {
                "pr_auc": [0.65] * cv_len,
                "roc_auc": [0.75] * cv_len,
                "f1": [0.55] * cv_len,
                "mcc": [0.45] * cv_len,
                "bal_acc": [0.70] * cv_len,
                "brier": [0.15] * cv_len,
                "accuracy": [0.80] * cv_len,
            },
            "cv_summary": {
                "pr_auc": {"mean": 0.65, "std": 0.02, "ci95": [0.61, 0.69]},
                "roc_auc": {"mean": 0.75, "std": 0.02, "ci95": [0.71, 0.79]},
                "f1": {"mean": 0.55, "std": 0.02, "ci95": [0.51, 0.59]},
                "mcc": {"mean": 0.45, "std": 0.02, "ci95": [0.41, 0.49]},
                "bal_acc": {"mean": 0.70, "std": 0.02, "ci95": [0.66, 0.74]},
                "brier": {"mean": 0.15, "std": 0.01, "ci95": [0.13, 0.17]},
                "accuracy": {"mean": 0.80, "std": 0.01, "ci95": [0.78, 0.82]},
            },
            "thresholds": thresholds,
            "roc": [[0.0, 0.0], [0.5, 0.8], [1.0, 1.0]],
            "pr": [[0.0, 0.8], [0.5, 0.6], [1.0, 0.2]],
            "calibration": [
                {"mean_pred": float(round(i * 0.1, 2)), "frac_pos": float(round(i * 0.08, 2)), "n": 100}
                for i in range(10)
            ],
            "timing": {"train_s": 1.2, "infer_ms_per_1k": 4.5, "model_kb": 12.3},
            "importance": {
                "permutation": [
                    {"feature": f"FEAT_{i}", "mean": 0.05, "std": 0.01}
                    for i in range(15)
                ]
            },
            "holdout": {"pr_auc": 0.64, "roc_auc": 0.74},
        }

    # Exactly 6 significance pairs
    pairs = [
        ("lr", "dt"), ("lr", "rf"), ("lr", "svm"),
        ("dt", "rf"), ("dt", "svm"), ("rf", "svm"),
    ]
    significance = []
    for a, b in pairs:
        significance.append({
            "a": a,
            "b": b,
            "metric": "pr_auc",
            "mean_diff": 0.02,
            "nadeau_bengio": {"t": 1.5, "p": 0.15, "p_adj": 0.45},
            "wilcoxon": {"stat": 10.0, "p": 0.20, "p_adj": 0.50},
        })

    imbalance_exp = [
        {"model": m, "strategy": s, "pr_auc_mean": 0.62, "pr_auc_std": 0.03}
        for m in ["lr", "dt", "rf", "svm"]
        for s in ["none", "balanced", "smote"]
    ]

    learning_curves = {
        m: {
            "sizes": [100, 200, 400],
            "train": [0.8, 0.75, 0.7],
            "val": [0.5, 0.55, 0.6],
            "train_std": [0.02, 0.02, 0.01],
            "val_std": [0.03, 0.02, 0.02],
        }
        for m in ["lr", "dt", "rf", "svm"]
    }

    meta = {
        "project": "QuadVerdict",
        "dataset": "UCI Default of Credit Card Clients",
        "n": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": float(round(n_pos / n_total, 4)),
        "seed": 42,
        "outer_cv": {"type": "RepeatedStratifiedKFold", "n_splits": 2 if quick else 5, "n_repeats": 1 if quick else 3},
        "inner_cv": {"type": "StratifiedKFold", "n_splits": 2 if quick else 3},
        "n_iter": {"lr": 3, "dt": 3, "rf": 3, "svm": 3},
        "svm_max_train": 8000,
        "default_cost_ratio": 5,
        "primary_metric": "pr_auc",
        "quick": quick,
        "versions": {
            "python": "3.11.0",
            "sklearn": "1.5.0",
            "imblearn": "0.12.0",
            "numpy": "1.26.0",
            "pandas": "2.2.0",
            "scipy": "1.13.0",
        },
        "hardware": {"cpu": "Mock CPU", "cores": 8},
        "built_at": "2026-10-04T00:00:00Z",
    }

    return {
        "meta": meta,
        "models": models,
        "imbalance_experiment": imbalance_exp,
        "significance": significance,
        "learning_curves": learning_curves,
    }


class TestValidResults:
    """Verify that structurally valid benchmarks pass validation."""

    def test_valid_quick_results(self):
        data = make_valid_mock_results(quick=True)
        # Should not raise
        validate(data)

    def test_valid_full_results(self):
        data = make_valid_mock_results(quick=False)
        validate(data)


class TestMutationSchemaValidation:
    """Mutation tests: verify that each schema or invariant violation is caught."""

    def test_missing_model_key(self):
        data = make_valid_mock_results()
        del data["models"]["rf"]
        with pytest.raises(SchemaValidationError, match="Missing required model 'rf'|'rf' is a required property"):
            validate(data)

    def test_threshold_table_row_count_not_101(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["thresholds"].pop()  # 100 rows
        with pytest.raises(SchemaValidationError, match="threshold table must have exactly 101 rows|is too short"):
            validate(data)

    def test_threshold_t_not_strictly_increasing(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["thresholds"][5]["t"] = data["models"]["lr"]["thresholds"][4]["t"]
        with pytest.raises(SchemaValidationError, match="threshold 't' is not strictly increasing"):
            validate(data)

    def test_conservation_invariant_tp_fn_mismatch(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["thresholds"][10]["tp"] += 5.0  # Breaks tp + fn == n_pos
        with pytest.raises(SchemaValidationError, match="tp \\+ fn .* != n_pos"):
            validate(data)

    def test_conservation_invariant_fp_tn_mismatch(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["thresholds"][10]["tn"] += 10.0  # Breaks fp + tn == n_neg
        with pytest.raises(SchemaValidationError, match="fp \\+ tn .* != n_neg"):
            validate(data)

    def test_monotonicity_tp_increasing(self):
        data = make_valid_mock_results()
        # Row 20 has higher tp than row 19 while preserving conservation
        new_tp = data["models"]["lr"]["thresholds"][19]["tp"] + 1.0
        data["models"]["lr"]["thresholds"][20]["tp"] = new_tp
        data["models"]["lr"]["thresholds"][20]["fn"] = float(data["meta"]["n_pos"] - new_tp)
        with pytest.raises(SchemaValidationError, match="tp .* increased relative to previous row"):
            validate(data)

    def test_monotonicity_fp_increasing(self):
        data = make_valid_mock_results()
        # Row 20 has higher fp than row 19 while preserving conservation
        new_fp = data["models"]["lr"]["thresholds"][19]["fp"] + 1.0
        data["models"]["lr"]["thresholds"][20]["fp"] = new_fp
        data["models"]["lr"]["thresholds"][20]["tn"] = float(data["meta"]["n_neg"] - new_fp)
        with pytest.raises(SchemaValidationError, match="fp .* increased relative to previous row"):
            validate(data)

    def test_cv_scores_length_mismatch(self):
        data = make_valid_mock_results(quick=False)
        data["models"]["lr"]["cv_scores"]["pr_auc"].pop()  # 14 folds instead of 15
        with pytest.raises(SchemaValidationError, match="cv_scores length must be 15"):
            validate(data)

    def test_score_outside_bounds(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["cv_scores"]["pr_auc"][0] = 1.25  # Exceeds 1.0
        with pytest.raises(SchemaValidationError, match="JSON Schema validation error"):
            validate(data)

    def test_mcc_score_outside_bounds(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["cv_scores"]["mcc"][0] = -1.5  # Below -1.0
        with pytest.raises(SchemaValidationError, match="JSON Schema validation error"):
            validate(data)

    def test_roc_points_exceed_limit(self):
        data = make_valid_mock_results()
        data["models"]["lr"]["roc"] = [[float(i) / 250.0, float(i) / 250.0] for i in range(250)]
        with pytest.raises(SchemaValidationError, match="ROC curve points exceed 200 limit|is too long"):
            validate(data)

    def test_significance_pairs_count(self):
        data = make_valid_mock_results()
        data["significance"].pop()  # 5 pairs instead of 6
        with pytest.raises(SchemaValidationError, match="Significance section must contain exactly 6 pairwise comparisons|is too short"):
            validate(data)


class TestExportUtilities:
    """Verify float rounding, file writing, and holdout evaluation."""

    def test_round_floats(self):
        sample = {
            "a": 0.1234567,
            "b": [0.9876543, {"c": 0.5555555}],
            "d": "text",
            "e": 10,
        }
        rounded = round_floats(sample, decimals=4)
        assert rounded["a"] == 0.1235
        assert rounded["b"][0] == 0.9877
        assert rounded["b"][1]["c"] == 0.5556
        assert rounded["d"] == "text"
        assert rounded["e"] == 10

    def test_round_floats_nan_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            round_floats({"bad": float("nan")})

    def test_write_and_size_guard(self, tmp_path):
        data = make_valid_mock_results(quick=True)
        out_file = tmp_path / "results.json"
        size = write(data, out_file, minified=True)
        assert size > 0
        assert out_file.is_file()

        # Check roundtrip
        with open(out_file, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["meta"]["project"] == "QuadVerdict"

    def test_write_oversized_raises(self, tmp_path, monkeypatch):
        data = make_valid_mock_results(quick=True)
        out_file = tmp_path / "results_oversized.json"
        # Temporarily lower RESULTS_MAX_KB to trigger limit
        monkeypatch.setattr(config, "RESULTS_MAX_KB", 0.1)
        with pytest.raises(SchemaValidationError, match="exceeds maximum allowed limit"):
            write(data, out_file)

    def test_assemble_results_integration(self):
        mock_data = make_valid_mock_results(quick=True)
        assembled = assemble_results(
            meta_info={
                "n": mock_data["meta"]["n"],
                "n_pos": mock_data["meta"]["n_pos"],
                "n_neg": mock_data["meta"]["n_neg"],
                "pos_rate": mock_data["meta"]["pos_rate"],
                "quick": True,
                "outer_splits": 2,
                "outer_repeats": 1,
                "inner_splits": 2,
            },
            models_data=mock_data["models"],
            imbalance_exp=mock_data["imbalance_experiment"],
            significance_data=mock_data["significance"],
            learning_curves_data=mock_data["learning_curves"],
        )
        assert "meta" in assembled
        assert "models" in assembled
        validate(assembled)

    def test_evaluate_holdout_set(self):
        from src.data import clean, load_raw, make_splits

        raw = load_raw(config.DATA_PATH)
        cleaned = clean(raw)
        X_dev, X_hold, y_dev, y_hold = make_splits(cleaned, seed=config.SEED, test_size=0.20)

        # Mock cv output with empty best_params
        mock_cv = {
            "models": {
                "dt": [{"best_params": {"classifier__max_depth": 3}}],
            },
            "meta": {"imbalance_strategy": "balanced"},
        }

        holdout_scores = evaluate_holdout_set(
            models=["dt"],
            cv_output=mock_cv,
            X_dev=X_dev.iloc[:300],
            y_dev=y_dev.iloc[:300],
            X_hold=X_hold.iloc[:100],
            y_hold=y_hold.iloc[:100],
        )

        assert "dt" in holdout_scores
        assert 0.0 <= holdout_scores["dt"]["pr_auc"] <= 1.0
        assert 0.0 <= holdout_scores["dt"]["roc_auc"] <= 1.0
