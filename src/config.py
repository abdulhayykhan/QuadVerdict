"""
config.py — Single source of truth for all constants in QuadVerdict.

Rules (TRD §10):
- config.SEED is the ONLY source of randomness.
  Pass it explicitly to every estimator, splitter, SMOTE, RandomizedSearchCV,
  permutation_importance call, and joblib Parallel task.
- Do not change SEED after the first full run — it would invalidate reproducibility.
- N_ITER and SVM_MAX_TRAIN are documented in the README if changed from defaults.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Paths (all relative to the project root)
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = ROOT / "data" / "raw"
RESULTS_DIR: Path = ROOT / "results"
RESULTS_JSON: Path = RESULTS_DIR / "results.json"
SCHEMA_JSON: Path = RESULTS_DIR / "schema.json"
CACHE_DIR: Path = RESULTS_DIR / "_cache"
DIST_DIR: Path = ROOT / "dist"
WEB_DIR: Path = ROOT / "web"
WEB_INDEX: Path = WEB_DIR / "index.html"
MOCK_JSON: Path = WEB_DIR / "mock_results.json"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# UCI Default of Credit Card Clients
# Expected filename inside DATA_RAW_DIR:
DATA_FILENAME: str = "default_of_credit_card_clients.csv"
DATA_PATH: Path = DATA_RAW_DIR / DATA_FILENAME

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20          # 20% held-out, NEVER used in tuning or CV
STRATIFY: bool = True

# ---------------------------------------------------------------------------
# Nested CV settings (TRD §4.3)
# ---------------------------------------------------------------------------
# Outer CV: RepeatedStratifiedKFold
OUTER_N_SPLITS: int = 5
OUTER_N_REPEATS: int = 3         # → 15 outer folds total

# Inner CV: StratifiedKFold inside RandomizedSearchCV
INNER_N_SPLITS: int = 3

# Hyperparameter search iterations per model (full run)
N_ITER: dict[str, int] = {
    "lr": 20,
    "dt": 20,
    "rf": 20,
    "svm": 12,
}

# Quick-mode overrides (--quick flag)
QUICK_N_ITER: dict[str, int] = {"lr": 3, "dt": 3, "rf": 3, "svm": 3}
QUICK_OUTER_N_SPLITS: int = 2
QUICK_OUTER_N_REPEATS: int = 1   # → 2 outer folds
QUICK_SUBSAMPLE: int = 2_000     # rows subsampled from dev set in quick mode

# ---------------------------------------------------------------------------
# SVM runtime control (TRD §4.2)
# ---------------------------------------------------------------------------
# Kernel SVC is O(n²–n³). Cap training rows per fold to make nested CV practical.
# The README and UI must state this explicitly.
SVM_MAX_TRAIN: int = 8_000

# ---------------------------------------------------------------------------
# Cost model (TRD §4.4, PRD §6.3)
# ---------------------------------------------------------------------------
# Default FN:FP cost ratio shown on first load (slider covers 1–50).
DEFAULT_COST_RATIO: int = 5

# ---------------------------------------------------------------------------
# Threshold table
# ---------------------------------------------------------------------------
N_THRESHOLD_STEPS: int = 101     # np.linspace(0, 1, 101)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_N_BINS: int = 10     # quantile bins

# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------
PERM_N_REPEATS: int = 5          # repeats per fold
PERM_TOP_K: int = 15             # features exported per model

# ---------------------------------------------------------------------------
# Curves downsampling
# ---------------------------------------------------------------------------
ROC_PR_MAX_POINTS: int = 200     # uniform arc-length downsampling

# ---------------------------------------------------------------------------
# Export / size budgets
# ---------------------------------------------------------------------------
FLOAT_DECIMALS: int = 5          # round all floats before JSON export
RESULTS_WARN_KB: int = 400       # warn above this
RESULTS_MAX_KB: int = 500        # hard limit
DIST_MAX_MB: float = 1.0         # dist/index.html hard limit

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
# joblib.Parallel n_jobs for the (model, fold) outer loop.
# Do NOT set high n_jobs simultaneously on RF/LR to avoid oversubscription.
# -1 = all cores; set explicitly to a lower value if the machine is constrained.
JOBLIB_N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Model keys and labels
# ---------------------------------------------------------------------------
MODEL_KEYS: list[str] = ["lr", "dt", "rf", "svm"]
MODEL_LABELS: dict[str, str] = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "svm": "SVM (RBF, capped)",
}

# Imbalance strategies used in the experiment (TRD §4.7)
IMBALANCE_STRATEGIES: list[str] = ["none", "balanced", "smote"]

# Main benchmark strategy — chosen by inner CV; this is the fallback default.
# The imbalance experiment (src/experiments.py) determines the winner per model.
MAIN_STRATEGY: str = "balanced"

# Learning curve training-size fractions (TRD §4.7)
LEARNING_CURVE_FRACTIONS: list[float] = [0.10, 0.25, 0.50, 0.75, 1.00]
LEARNING_CURVE_CV: int = 3
