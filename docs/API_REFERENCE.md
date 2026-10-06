# QuadVerdict API & CLI Reference

This document provides technical reference documentation for the modules in `src/`, command-line interfaces, and configuration parameters of **QuadVerdict**.

---

## 1. CLI Interfaces

### 1.1 `run_benchmark.py`

The primary execution entry point for the benchmark pipeline.

```bash
python run_benchmark.py [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--quick` | flag | `False` | Run in rapid sanity mode: 1 outer fold, 2 inner CV iterations, limited hyperparameter candidates. Used for CI/CD smoke tests. |
| `--resume` | flag | `False` | Resume execution from existing cached folds in `results/_cache/`. Skips models and folds already fitted and serialized. |
| `--seed` | int | `42` | Global random state seed governing cross-validation splits and estimator initializations. |

**Phases Executed:**
1. Data Ingestion & Sealed 80/20 Hold-out Split
2. Pipeline Instantiation (4 paradigms: `lr`, `dt`, `rf`, `svm`)
3. Nested Cross-Validation (5 splits $\times$ 3 repeats = 15 folds)
4. Hold-Out Evaluation (Strictly evaluated once post-hoc)
5. Threshold Metric Sweeps & Financial Cost Optimization
6. Arc-Length Curve Downsampling & Platt Calibration Diagrams
7. Nadeau-Bengio & Wilcoxon Significance Testing with Holm-Bonferroni Correction
8. Permutation Feature Importance Analysis (15 outer folds $\times$ 23 features)
9. Class Imbalance Experiments (None vs Balanced Weights vs Pipeline SMOTE)
10. Computational Latency Telemetry & JSON Schema-Validated Export

---

### 1.2 `scripts/generate_figures.py`

Generates 300 DPI publication-grade static figures from canonical benchmark results.

```bash
python scripts/generate_figures.py [--results PATH] [--out-dir PATH]
```

**Options:**
- `--results`: Path to canonical `results.json` (default: `results/results.json`).
- `--out-dir`: Destination directory for PNG artifacts (default: `results/figures/`).

**Output Figures:**
- `cv_boxplots.png`: 15-fold PR-AUC & Brier score distributions with Okabe-Ito colors.
- `cost_curves.png`: Expected cost as a function of threshold $t \in [0, 1]$ across $C_{\text{FN}}:C_{\text{FP}} = 5:1$.
- `roc_pr_curves.png`: Dual-panel ROC and Precision-Recall pooled curves with chance lines.
- `calibration_curves.png`: 10-quantile reliability diagrams comparing empirical default frequency against predicted probability.
- `feature_importance.png`: Horizontal error-bar chart displaying permutation PR-AUC loss across 23 features.
- `imbalance_experiment.png`: Grouped bar chart comparing Vanilla, Balanced Weights, and SMOTE.
- `learning_curves.png`: Sample efficiency scaling across $N \in [1,600, 16,000]$.

---

### 1.3 `web/build_site.py`

Compiles the standalone interactive client dashboard into `dist/index.html`.

```bash
python web/build_site.py [--template PATH] [--results PATH] [--out PATH] [--mock]
```

**Options:**
- `--template`: Path to `web/index.html` template (default: `web/index.html`).
- `--results`: Path to JSON results payload (default: `results/results.json`).
- `--out`: Destination output HTML file (default: `dist/index.html`).
- `--mock`: Build using synthetic schema-compliant test fixture (`web/mock_results.json`).

**Validation Safeguards:**
- Asserts template contains exactly one `/*__RESULTS_JSON__*/` token.
- Validates results payload against `results/schema.json` via JSONSchema Draft-07.
- Escapes all `</` character pairs as `<\/` to eliminate premature HTML `<script>` breakout vulnerabilities.
- Asserts compiled bundle size is well under $1.0\text{ MB}$ (typically $\approx 167\text{ KB}$).

---

## 2. Core Python Modules (`src/`)

### 2.1 `src.config`

Global immutables, dataset constants, hyperparameter search spaces, and execution budgets.

```python
SEED = 42                     # Deterministic random seed
SVM_MAX_TRAIN = 8_000         # Maximum training samples for SVM per fold
DEFAULT_COST_RATIO = 5        # Baseline C_FN : C_FP loss ratio
N_THRESHOLD_STEPS = 101       # Discretized threshold resolution (0.00 to 1.00)
OUTER_N_SPLITS = 5            # Outer RepeatedStratifiedKFold splits
OUTER_N_REPEATS = 3           # Outer CV repeats (Total J = 15)
INNER_N_SPLITS = 3            # Inner CV tuning splits
RESULTS_MAX_KB = 500          # Maximum export payload size budget
DIST_MAX_MB = 1.0             # Maximum compiled bundle size budget
```

---

### 2.2 `src.data`

**`load_raw_data(data_path: Path | str) -> pd.DataFrame`**
- Ingests raw UCI Default of Credit Card Clients dataset (`.csv` or `.xls`).
- Renames legacy column aliases (`PAY_0` $\to$ `PAY_1`, `default payment next month` $\to$ `default`).
- Drops non-informative identifier columns (`ID`).
- Validates data shape ($30,000$ rows, $24$ columns) and column types.

**`split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]`**
- Executes a strictly stratified 80/20 train/hold-out split on the target `default`.
- Returns `(dev_df, holdout_df)` ($24,000$ dev rows, $6,000$ hold-out rows).

---

### 2.3 `src.pipelines`

**`build_pipeline(model_key: str, strategy: str = "none", random_state: int = 42) -> Pipeline`**
- Assembles scikit-learn or imblearn pipelines with data-leakage boundaries:
  - `model_key`: One of `"lr"`, `"dt"`, `"rf"`, `"svm"`.
  - `strategy`: Imbalance mitigation strategy (`"none"`, `"balanced"`, `"smote"`).
- Automatically incorporates `StandardScaler` for `lr` and `svm`. Omits scaler for `dt` and `rf`.
- Encapsulates SVM within `StratifiedSubsampleClassifier` to ensure training sets never exceed 8,000 rows while evaluating on full test sets.

**`get_param_distributions(model_key: str) -> dict`**
- Returns hyperparameter distributions optimized for `RandomizedSearchCV`:
  - `lr`: Regularization $C \in \text{LogUniform}(10^{-4}, 10^2)$, solver penalties.
  - `dt`: `max_depth` $\in [3, 10]$, `min_samples_split` $\in [10, 100]$, `min_samples_leaf` $\in [5, 50]$.
  - `rf`: `n_estimators` $\in [100, 300]$, `max_depth` $\in [5, 15]$, `max_features` $\in [0.2, 0.8]$.
  - `svm`: RBF kernel, $C \in \text{LogUniform}(10^{-2}, 10^2)$, $\gamma \in \text{LogUniform}(10^{-4}, 10^0)$.

---

### 2.4 `src.nested_cv`

**`run_nested_cv(dev_df: pd.DataFrame, pipelines: dict, quick: bool = False, resume: bool = False, seed: int = 42) -> dict`**
- Orchestrates the full $5 \times 3$ Repeated Stratified Cross-Validation loop across 15 outer folds.
- Evaluates out-of-fold probability predictions $\hat{p}_i$ and records scalar metrics for each fold.
- Implements atomic checkpoint serialization to `results/_cache/`.

---

### 2.5 `src.metrics`

**`compute_scalar_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.50) -> dict`**
- Computes PR-AUC (`average_precision_score`), ROC-AUC (`roc_auc_score`), Brier Score, F1-Score, and Matthews Correlation Coefficient (MCC).

**`compute_threshold_table(y_true: np.ndarray, y_proba: np.ndarray, cost_ratio: float = 5.0) -> list[dict]`**
- Sweeps 101 thresholds $t \in [0.00, 1.00]$.
- Calculates confusion matrix counts ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$), Precision, Recall, Specificity, F1, and financial Expected Cost.
- Verifies mathematical conservation invariants ($\text{TP} + \text{FN} = P$, $\text{FP} + \text{TN} = N$).

**`downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> tuple[np.ndarray, np.ndarray]`**
- Executes Euclidean arc-length downsampling to compact ROC and PR curves into $\le 200$ high-fidelity coordinate pairs.

---

### 2.6 `src.significance`

**`nadeau_bengio_test(diffs: np.ndarray, n_train: int = 19200, n_test: int = 4800) -> tuple[float, float]`**
- Computes the Nadeau-Bengio corrected resampled $t$-statistic and two-tailed $p$-value across cross-validation differences.

**`wilcoxon_test(diffs: np.ndarray) -> tuple[float, float]`**
- Computes non-parametric Wilcoxon signed-rank test statistic and $p$-value with zero-difference handling.

**`holm_bonferroni(p_values: np.ndarray) -> np.ndarray`**
- Applies step-down family-wise error rate correction to an array of raw $p$-values.

**`run_pairwise_significance(cv_scores: dict[str, list[float]]) -> list[dict]`**
- Executes all 6 pairwise comparisons across `rf`, `dt`, `svm`, `lr` for both PR-AUC and ROC-AUC.
- Emits structured significance verdicts (`significant: bool`, `better_model: str`).

---

### 2.7 `src.explain`

**`compute_permutation_importance(estimator, X_test: pd.DataFrame, y_test: np.ndarray, n_repeats: int = 5, random_state: int = 42) -> dict`**
- Evaluates out-of-fold PR-AUC degradation across all 23 features.
- Computes mean drop and standard error across repeats and outer folds.

---

### 2.8 `src.export`

**`validate_results_schema(results_data: dict, schema_path: Path | str = "results/schema.json") -> None`**
- Validates compiled benchmark results against the official JSON schema. Raises `ValidationError` if any metric or structure is non-compliant.

**`write_results(results_data: dict, out_path: Path | str = "results/results.json") -> Path`**
- Rounds floating point values to 5 decimal places (`config.FLOAT_DECIMALS`).
- Writes compact JSON with size budget enforcement ($< 500\text{ KB}$).
