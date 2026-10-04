"""
export.py — Results assembly, JSON schema validation, and serialization.

Design rules (TRD §4.9, §5, PRD §6.5):
1. Assemble results into the standardized data contract defined in results/schema.json.
2. Enforce strict mathematical invariants:
   - Exactly 101 threshold rows; t strictly increasing from 0.00 to 1.00.
   - tp + fn == n_pos (within 1e-4 tolerance) and fp + tn == n_neg (within 1e-4 tolerance).
   - tp and fp are non-increasing in t.
   - Outer-fold cv_scores length == 15 (or 2 in quick mode).
   - ROC/PR curves <= 200 points.
   - Exactly 6 significance pairwise rows across the 4 models.
3. Validate against results/schema.json via jsonschema.
4. Write minified JSON with 5-decimal rounding to respect the 500 KB hard limit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import imblearn
import jsonschema
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure project root is in sys.path when invoked directly as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.pipelines import build_pipeline  # noqa: E402


class SchemaValidationError(Exception):
    """Raised when results fail JSON schema or semantic invariant validation."""
    pass


def round_floats(obj: Any, decimals: int = 5) -> Any:
    """Recursively round floating point values in data structures."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError("NaN or Inf encountered in benchmark export data")
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, decimals) for v in obj]
    if isinstance(obj, tuple):
        return [round_floats(v, decimals) for v in obj]
    if isinstance(obj, np.generic):
        py_val = obj.item()
        return round_floats(py_val, decimals)
    return obj


def evaluate_holdout_set(
    models: list[str],
    cv_output: dict[str, Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_hold: pd.DataFrame,
    y_hold: pd.Series,
    seed: int = config.SEED,
) -> dict[str, dict[str, float]]:
    """Fit each model on full dev set and evaluate once on hold-out set (TRD §4.9).

    Parameters
    ----------
    models : list of str
    cv_output : dict from run_nested_cv()
    X_dev, y_dev : full development set
    X_hold, y_hold : untouched hold-out set
    seed : int

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of model_key -> {'pr_auc': float, 'roc_auc': float}
    """
    results: dict[str, dict[str, float]] = {}
    models_dict = cv_output.get("models", {})
    imbalance_strategy = cv_output.get("meta", {}).get("imbalance_strategy", "balanced")

    for model_key in models:
        folds = models_dict.get(model_key, [])
        # Find most frequent best params across outer folds
        param_strings = [json.dumps(f["best_params"], sort_keys=True) for f in folds if "best_params" in f]
        if param_strings:
            most_common_json = Counter(param_strings).most_common(1)[0][0]
            best_params = json.loads(most_common_json)
        else:
            best_params = {}

        pipe = build_pipeline(model_key, imbalance_strategy=imbalance_strategy)
        pipe.set_params(**best_params)
        pipe.fit(X_dev, y_dev)

        y_proba = pipe.predict_proba(X_hold)[:, 1]
        pr_auc = float(average_precision_score(y_hold, y_proba))
        roc_auc = float(roc_auc_score(y_hold, y_proba))

        results[model_key] = {
            "pr_auc": float(round(pr_auc, 5)),
            "roc_auc": float(round(roc_auc, 5)),
        }

    return results


def assemble_results(
    meta_info: dict[str, Any],
    models_data: dict[str, Any],
    imbalance_exp: dict[str, Any] | list[dict[str, Any]],
    significance_data: list[dict[str, Any]],
    learning_curves_data: dict[str, Any],
) -> dict[str, Any]:
    """Assemble all sub-payloads into the canonical QuadVerdict data contract.

    Parameters
    ----------
    meta_info : dict
    models_data : dict
    imbalance_exp : dict or list of dict
    significance_data : list of dict
    learning_curves_data : dict

    Returns
    -------
    dict
        Standardized results dictionary matching results/schema.json.
    """
    # 1. Normalize meta
    quick = bool(meta_info.get("quick", False))
    built_at = meta_info.get("built_at", datetime.datetime.now(datetime.UTC).isoformat())

    meta = {
        "project": "QuadVerdict",
        "dataset": "UCI Default of Credit Card Clients",
        "n": int(meta_info["n"]),
        "n_pos": int(meta_info["n_pos"]),
        "n_neg": int(meta_info["n_neg"]),
        "pos_rate": float(round(float(meta_info["pos_rate"]), 4)),
        "seed": int(meta_info.get("seed", config.SEED)),
        "outer_cv": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": int(meta_info.get("outer_splits", 2 if quick else 5)),
            "n_repeats": int(meta_info.get("outer_repeats", 1 if quick else 3)),
        },
        "inner_cv": {
            "type": "StratifiedKFold",
            "n_splits": int(meta_info.get("inner_splits", 2 if quick else 3)),
        },
        "n_iter": meta_info.get(
            "n_iter",
            {"lr": 3 if quick else 20, "dt": 3 if quick else 20, "rf": 3 if quick else 20, "svm": 3 if quick else 12},
        ),
        "svm_max_train": int(config.SVM_MAX_TRAIN),
        "default_cost_ratio": float(config.COST_RATIO_DEFAULT),
        "primary_metric": "pr_auc",
        "quick": quick,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "imblearn": imblearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "hardware": {
            "cpu": str(platform.processor()) or "Unknown",
            "cores": int(os.cpu_count() or 1),
        },
        "built_at": built_at,
    }

    # 2. Format models dictionary
    labels = {
        "lr": "Logistic Regression",
        "dt": "Decision Tree",
        "rf": "Random Forest",
        "svm": "Support Vector Machine",
    }

    formatted_models: dict[str, Any] = {}
    for m in config.MODEL_KEYS:
        if m not in models_data:
            continue
        m_raw = models_data[m]

        # Extract best params and frequency
        folds = m_raw.get("folds", [])
        param_counts = Counter(
            json.dumps(f.get("best_params", {}), sort_keys=True) for f in folds if "best_params" in f
        )
        freq_list = []
        for p_str, count in param_counts.most_common():
            freq_list.append({"params": json.loads(p_str), "count": count})

        best_params = freq_list[0]["params"] if freq_list else m_raw.get("best_params", {})

        # CV scores and summary
        cv_scores = m_raw.get("cv_scores", {})
        if not cv_scores and folds:
            # Construct from folds list
            metric_keys = ["pr_auc", "roc_auc", "f1", "mcc", "bal_acc", "brier", "accuracy"]
            cv_scores = {k: [float(f[k]) for f in folds if k in f] for k in metric_keys}

        cv_summary = {}
        for metric, scores in cv_scores.items():
            arr = np.asarray(scores, dtype=float)
            mean_v = float(np.mean(arr)) if len(arr) > 0 else 0.0
            std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            sem = std_v / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            ci95 = [float(max(-1.0, mean_v - 1.96 * sem)), float(min(1.0, mean_v + 1.96 * sem))]
            cv_summary[metric] = {
                "mean": mean_v,
                "std": std_v,
                "ci95": ci95,
            }

        # Format importance: wrap in {"permutation": [...]}
        imp_raw = m_raw.get("importance", [])
        if isinstance(imp_raw, dict) and "permutation" in imp_raw:
            perm_list = imp_raw["permutation"]
        elif isinstance(imp_raw, list):
            perm_list = [
                {
                    "feature": item["feature"],
                    "mean": item.get("mean", item.get("importance_mean", 0.0)),
                    "std": item.get("std", item.get("importance_std", 0.0)),
                }
                for item in imp_raw
            ]
        else:
            perm_list = []

        formatted_models[m] = {
            "label": labels.get(m, m.upper()),
            "best_params": best_params,
            "best_params_frequency": freq_list,
            "cv_scores": cv_scores,
            "cv_summary": cv_summary,
            "thresholds": m_raw.get("thresholds", m_raw.get("threshold_table", [])),
            "roc": m_raw.get("roc", m_raw.get("curves", {}).get("roc", [])),
            "pr": m_raw.get("pr", m_raw.get("curves", {}).get("pr", [])),
            "calibration": m_raw.get("calibration", []),
            "timing": m_raw.get("timing", {"train_s": 0.0, "infer_ms_per_1k": 0.0, "model_kb": 0.0}),
            "importance": {"permutation": perm_list},
            "holdout": m_raw.get("holdout", {"pr_auc": 0.0, "roc_auc": 0.0}),
        }

    # 3. Format imbalance experiment
    if isinstance(imbalance_exp, list):
        formatted_imb = imbalance_exp
    else:
        formatted_imb = []
        for model_k, strats in imbalance_exp.items():
            for strat_k, s_info in strats.items():
                formatted_imb.append({
                    "model": model_k,
                    "strategy": strat_k,
                    "pr_auc_mean": float(s_info.get("pr_auc_mean", 0.0)),
                    "pr_auc_std": float(s_info.get("pr_auc_std", 0.0)),
                })

    # 4. Format significance comparisons
    formatted_sig = []
    for comp in significance_data:
        formatted_sig.append({
            "a": comp.get("a", comp.get("model_a", "")),
            "b": comp.get("b", comp.get("model_b", "")),
            "metric": comp.get("metric", "pr_auc"),
            "mean_diff": float(comp.get("mean_diff", 0.0)),
            "nadeau_bengio": {
                "t": float(comp.get("nadeau_bengio", {}).get("t", comp.get("t_stat", 0.0))),
                "p": float(comp.get("nadeau_bengio", {}).get("p", comp.get("p_value_nb", 1.0))),
                "p_adj": float(comp.get("nadeau_bengio", {}).get("p_adj", comp.get("p_adj_nb", 1.0))),
            },
            "wilcoxon": {
                "stat": float(comp.get("wilcoxon", {}).get("stat", comp.get("wilcoxon_stat", 0.0))),
                "p": float(comp.get("wilcoxon", {}).get("p", comp.get("p_value_wilcoxon", 1.0))),
                "p_adj": float(comp.get("wilcoxon", {}).get("p_adj", comp.get("p_adj_wilcoxon", 1.0))),
            },
        })

    # 5. Format learning curves
    formatted_curves = {}
    for m in config.MODEL_KEYS:
        if m in learning_curves_data:
            c = learning_curves_data[m]
            formatted_curves[m] = {
                "sizes": [int(s) for s in c.get("sizes", c.get("train_sizes", []))],
                "train": [float(v) for v in c.get("train", c.get("train_mean", []))],
                "val": [float(v) for v in c.get("val", c.get("val_mean", []))],
                "train_std": [float(v) for v in c.get("train_std", [])],
                "val_std": [float(v) for v in c.get("val_std", [])],
            }

    assembled = {
        "meta": meta,
        "models": formatted_models,
        "imbalance_experiment": formatted_imb,
        "significance": formatted_sig,
        "learning_curves": formatted_curves,
    }

    # Round all floating point numbers to 5 decimal places for clean storage
    return round_floats(assembled, decimals=5)


def validate(results: dict[str, Any], schema_path: Path | str | None = None) -> None:
    """Validate benchmark results against JSON schema and semantic invariants.

    Parameters
    ----------
    results : dict
    schema_path : Path or str, optional

    Raises
    ------
    SchemaValidationError
        If schema validation or mathematical invariant checks fail.
    """
    if schema_path is None:
        schema_path = config.RESULTS_DIR / "schema.json"
    else:
        schema_path = Path(schema_path)

    if not schema_path.is_file():
        raise SchemaValidationError(f"Schema file not found at: {schema_path}")

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    # 1. Structural validation via jsonschema
    try:
        jsonschema.validate(instance=results, schema=schema)
    except jsonschema.ValidationError as err:
        raise SchemaValidationError(f"JSON Schema validation error: {err.message}") from err

    meta = results.get("meta", {})
    models = results.get("models", {})
    n_pos = meta.get("n_pos")
    n_neg = meta.get("n_neg")
    is_quick = meta.get("quick", False)

    # 2. Required model coverage
    for expected_model in config.MODEL_KEYS:
        if expected_model not in models:
            raise SchemaValidationError(f"Missing required model '{expected_model}' in models payload.")

    # 3. Model-level invariants
    for model_key, model_data in models.items():
        # Check thresholds count
        thresholds = model_data.get("thresholds", [])
        if len(thresholds) != 101:
            raise SchemaValidationError(
                f"Model '{model_key}': threshold table must have exactly 101 rows, got {len(thresholds)}"
            )

        prev_t = -1.0
        prev_tp = float("inf")
        prev_fp = float("inf")

        for idx, row in enumerate(thresholds):
            t = row["t"]
            tp = row["tp"]
            fp = row["fp"]
            tn = row["tn"]
            fn = row["fn"]

            # Strictly increasing t
            if t <= prev_t:
                raise SchemaValidationError(
                    f"Model '{model_key}', row {idx}: threshold 't' is not strictly increasing ({t} <= {prev_t})"
                )
            prev_t = t

            # Conservation invariant: tp + fn == n_pos
            if abs((tp + fn) - n_pos) > 1e-4:
                raise SchemaValidationError(
                    f"Model '{model_key}', row {idx} (t={t}): tp + fn ({tp + fn}) != n_pos ({n_pos})"
                )

            # Conservation invariant: fp + tn == n_neg
            if abs((fp + tn) - n_neg) > 1e-4:
                raise SchemaValidationError(
                    f"Model '{model_key}', row {idx} (t={t}): fp + tn ({fp + tn}) != n_neg ({n_neg})"
                )

            # Monotonicity: tp and fp must be non-increasing with t
            # Allow tiny floating point slack 1e-4
            if tp > prev_tp + 1e-4:
                raise SchemaValidationError(
                    f"Model '{model_key}', row {idx}: tp ({tp}) increased relative to previous row ({prev_tp})"
                )
            if fp > prev_fp + 1e-4:
                raise SchemaValidationError(
                    f"Model '{model_key}', row {idx}: fp ({fp}) increased relative to previous row ({prev_fp})"
                )
            prev_tp = tp
            prev_fp = fp

        # cv_scores length verification
        expected_cv_length = 2 if is_quick else 15
        cv_scores = model_data.get("cv_scores", {})
        for metric, scores in cv_scores.items():
            if len(scores) != expected_cv_length:
                raise SchemaValidationError(
                    f"Model '{model_key}', metric '{metric}': cv_scores length must be {expected_cv_length} "
                    f"({'quick' if is_quick else 'full'} mode), got {len(scores)}"
                )

        # ROC and PR array sizes
        roc_pts = model_data.get("roc", [])
        pr_pts = model_data.get("pr", [])
        if len(roc_pts) > 200:
            raise SchemaValidationError(
                f"Model '{model_key}': ROC curve points exceed 200 limit ({len(roc_pts)})"
            )
        if len(pr_pts) > 200:
            raise SchemaValidationError(
                f"Model '{model_key}': PR curve points exceed 200 limit ({len(pr_pts)})"
            )

    # 4. Significance pairs invariant (exactly 6 comparisons)
    sig_records = results.get("significance", [])
    if len(sig_records) != 6:
        raise SchemaValidationError(
            f"Significance section must contain exactly 6 pairwise comparisons, got {len(sig_records)}"
        )


def write(results: dict[str, Any], out_path: Path | str, minified: bool = True) -> int:
    """Serialize and write results dictionary to JSON file.

    Parameters
    ----------
    results : dict
    out_path : Path or str
    minified : bool, default=True

    Returns
    -------
    file_size_bytes : int
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    separators = (",", ":") if minified else (", ", ": ")
    json_bytes = json.dumps(results, separators=separators, allow_nan=False).encode("utf-8")

    file_size_bytes = len(json_bytes)
    file_size_kb = file_size_bytes / 1024.0

    # Warning and hard constraint check
    if file_size_kb > config.RESULTS_WARN_KB:
        print(f"[WARNING] results.json size ({file_size_kb:.1f} KB) exceeds warning threshold ({config.RESULTS_WARN_KB} KB)")
    if file_size_kb > config.RESULTS_MAX_KB:
        raise SchemaValidationError(
            f"results.json size ({file_size_kb:.1f} KB) exceeds maximum allowed limit ({config.RESULTS_MAX_KB} KB)"
        )

    with open(out_path, "wb") as f:
        f.write(json_bytes)

    return file_size_bytes


def main() -> int:
    """CLI utility for schema validation."""
    parser = argparse.ArgumentParser(description="Validate QuadVerdict results JSON against schema.")
    parser.add_argument("--validate", type=Path, required=True, help="Path to results.json file.")
    parser.add_argument("--schema", type=Path, default=config.RESULTS_DIR / "schema.json", help="Path to schema.json.")
    args = parser.parse_args()

    if not args.validate.is_file():
        print(f"[ERROR] Target file does not exist: {args.validate}", file=sys.stderr)
        return 1

    try:
        with open(args.validate, encoding="utf-8") as f:
            data = json.load(f)
        validate(data, schema_path=args.schema)
        print(f"[OK] {args.validate} successfully validated against {args.schema}.")
        return 0
    except SchemaValidationError as err:
        print(f"[FAIL] Validation error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
