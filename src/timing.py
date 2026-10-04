"""
timing.py — Training time, inference latency, model compression size, and hardware telemetry.

Design rules (TRD §4.8, PRD §6.5):
1. train_s: mean wall time of outer-fold refit (from nested_cv).
2. infer_ms_per_1k: median over 10 timed single-process predict_proba runs on 1,000 held-out rows.
3. model_kb: size of joblib.dump(estimator, BytesIO(), compress=3) in kilobytes.
4. Hardware: captures CPU processor, physical/logical core counts, OS, and Python release.
"""

from __future__ import annotations

import io
import os
import platform
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_KEYS, SEED
from src.pipelines import build_pipeline


def get_hardware_telemetry() -> dict[str, Any]:
    """Capture hardware specifications to contextualize benchmark timings."""
    return {
        "cpu_processor": str(platform.processor()) or "Unknown",
        "cpu_architecture": str(platform.machine()),
        "cpu_cores": int(os.cpu_count() or 1),
        "os_name": str(platform.system()),
        "os_release": str(platform.release()),
        "python_version": str(platform.python_version()),
    }


def measure_inference_latency_per_1k(
    estimator: Any,
    X_sample: pd.DataFrame,
    n_runs: int = 10,
) -> float:
    """Measure median inference latency in milliseconds for 1,000 samples.

    Parameters
    ----------
    estimator : fitted Pipeline
    X_sample : pd.DataFrame of exactly or scaled to 1,000 rows
    n_runs : int, default=10

    Returns
    -------
    median_ms : float
    """
    n_rows = len(X_sample)
    if n_rows == 0:
        return 0.0

    # Warmup run
    estimator.predict_proba(X_sample)

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        estimator.predict_proba(X_sample)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Normalize to exactly 1,000 rows if sample count differs
        normalized_ms = elapsed_ms * (1000.0 / n_rows)
        latencies_ms.append(normalized_ms)

    return float(round(float(np.median(latencies_ms)), 3))


def measure_model_size_kb(estimator: Any, compress_level: int = 3) -> float:
    """Measure serialized, compressed size of a fitted model in kilobytes.

    Parameters
    ----------
    estimator : fitted Pipeline
    compress_level : int, default=3

    Returns
    -------
    size_kb : float
    """
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer, compress=compress_level)
    size_bytes = len(buffer.getvalue())
    return float(round(size_bytes / 1024.0, 2))


def measure_timing_and_size(
    models: list[str] | None,
    cv_output: dict[str, Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_hold: pd.DataFrame,
    n_inference_runs: int = 10,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compile comprehensive timing, latency, memory, and telemetry data (TRD §4.8).

    Parameters
    ----------
    models : list of str, optional
    cv_output : dict from run_nested_cv()
    X_dev, y_dev : training inputs
    X_hold : held-out dataset for inference benchmark
    n_inference_runs : int, default=10
    seed : int, default=SEED

    Returns
    -------
    dict
        'models': mapping of model_key -> {train_s, infer_ms_per_1k, model_kb}
        'hardware': hardware telemetry
    """
    if models is None:
        models = MODEL_KEYS

    models_dict = cv_output.get("models", {})
    imbalance_strategy = cv_output.get("meta", {}).get("imbalance_strategy", "balanced")

    # Held-out slice of 1000 rows for standardized inference benchmark
    n_infer_rows = min(1000, len(X_hold))
    X_infer_batch = X_hold.iloc[:n_infer_rows]

    results: dict[str, dict[str, float]] = {}

    for model_key in models:
        # 1. Training time (from nested CV fold results)
        if model_key in models_dict and models_dict[model_key]:
            fit_times = [f["fit_time_s"] for f in models_dict[model_key]]
            train_s = float(round(float(np.mean(fit_times)), 3))
        else:
            train_s = 0.0

        # 2. Fit a representative model on dev set to benchmark inference and size
        pipe = build_pipeline(model_key, imbalance_strategy=imbalance_strategy)
        # Use first fold's best params if available
        if model_key in models_dict and models_dict[model_key]:
            best_params = models_dict[model_key][0].get("best_params", {})
            pipe.set_params(**best_params)

        pipe.fit(X_dev, y_dev)

        # 3. Inference latency (median over 10 runs)
        infer_ms = measure_inference_latency_per_1k(
            pipe, X_infer_batch, n_runs=n_inference_runs
        )

        # 4. Model size (compressed joblib)
        model_kb = measure_model_size_kb(pipe, compress_level=3)

        results[model_key] = {
            "train_s": train_s,
            "infer_ms_per_1k": infer_ms,
            "model_kb": model_kb,
        }

    return {
        "models": results,
        "hardware": get_hardware_telemetry(),
    }
