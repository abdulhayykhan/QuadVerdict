"""
conftest.py — shared pytest configuration for QuadVerdict.

Phase 0: Contains minimal sanity tests so pytest exits 0 while all
         functional test files are stubs with no test functions yet.
Later phases will add fixtures here (e.g., a tiny DataFrame fixture).
"""

import importlib
import subprocess
import sys


def test_phase0_config_imports() -> None:
    """Skeleton sanity: src.config must be importable and key constants must be correct."""
    config = importlib.import_module("src.config")
    assert config.SEED == 42, "SEED must be 42 for reproducibility"
    assert config.SVM_MAX_TRAIN == 8_000
    assert config.DEFAULT_COST_RATIO == 5
    assert set(config.MODEL_KEYS) == {"lr", "dt", "rf", "svm"}
    assert config.N_THRESHOLD_STEPS == 101
    assert config.FLOAT_DECIMALS == 5


def test_phase0_run_benchmark_argparse() -> None:
    """run_benchmark.py must parse --quick, --resume, --seed without error."""
    result = subprocess.run(
        [sys.executable, "run_benchmark.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--quick" in result.stdout
    assert "--resume" in result.stdout
    assert "--seed" in result.stdout
