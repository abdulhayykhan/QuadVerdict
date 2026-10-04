"""
test_skeleton.py — Phase 0 sanity tests.

Verifies that the project skeleton is healthy:
- src.config imports and key constants are correct.
- run_benchmark.py --help works (argument parser is valid).

These two tests are the Phase 0 exit criterion.
They will remain in the suite permanently as a regression guard.
"""

import importlib
import subprocess
import sys


def test_config_constants() -> None:
    """src.config must be importable and key constants must match the TRD spec."""
    config = importlib.import_module("src.config")

    # Reproducibility: seed must be exactly 42 (TRD §10)
    assert config.SEED == 42, "SEED must be 42"

    # SVM runtime cap (TRD §4.2)
    assert config.SVM_MAX_TRAIN == 8_000

    # Cost model default (PRD §15, TRD §5)
    assert config.DEFAULT_COST_RATIO == 5

    # Threshold table size (TRD §5)
    assert config.N_THRESHOLD_STEPS == 101

    # Float precision for export (TRD §5)
    assert config.FLOAT_DECIMALS == 5

    # Model keys (TRD §5)
    assert set(config.MODEL_KEYS) == {"lr", "dt", "rf", "svm"}
    assert set(config.MODEL_LABELS.keys()) == {"lr", "dt", "rf", "svm"}

    # Outer CV: 5×3 = 15 folds (TRD §4.3)
    assert config.OUTER_N_SPLITS == 5
    assert config.OUTER_N_REPEATS == 3

    # Inner CV: 3 folds (TRD §4.3)
    assert config.INNER_N_SPLITS == 3

    # Export size budgets (TRD §5)
    assert config.RESULTS_MAX_KB == 500
    assert config.DIST_MAX_MB == 1.0


def test_run_benchmark_argparse() -> None:
    """run_benchmark.py must expose --quick, --resume and --seed via --help."""
    result = subprocess.run(
        [sys.executable, "run_benchmark.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    assert "--quick" in result.stdout, "--quick flag missing from argparse"
    assert "--resume" in result.stdout, "--resume flag missing from argparse"
    assert "--seed" in result.stdout, "--seed flag missing from argparse"


def test_run_benchmark_runs_as_stub() -> None:
    """run_benchmark.py (stub) must exit 0 and print phase markers."""
    result = subprocess.run(
        [sys.executable, "run_benchmark.py"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"Stub run failed:\n{result.stderr}"
    # Each phase marker must be present
    for phase_n in range(1, 11):
        marker = f"[Phase {phase_n}/10]"
        assert marker in result.stdout, f"Missing {marker} in output"
