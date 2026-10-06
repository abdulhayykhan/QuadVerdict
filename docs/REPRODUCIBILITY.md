# QuadVerdict Audit & Reproducibility Guide

This guide details the procedure for independently replicating, verifying, and auditing all numerical metrics, statistical conclusions, static figures, and web application builds of **QuadVerdict**.

---

## 1. Reproducibility Mandate & Determinism Guarantees

Every step of the QuadVerdict benchmark is mathematically deterministic:

1. **Global Seed:** Set explicitly to `42` (`src/config.py:SEED`).
2. **Deterministic Data Partition:**
   - Raw records ($N = 30,000$) are partitioned with `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
   - The Development Set ($N = 24,000$) has exactly $5,309$ defaults ($22.12\%$).
   - The Hold-out Set ($N = 6,000$) has exactly $1,327$ defaults ($22.12\%$).
3. **Cross-Validation Split Identifiers:**
   - `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`.
   - Each outer fold has exactly 19,200 training rows and 4,800 validation rows.
4. **Estimator Random States:**
   - All randomized algorithms (`RandomForestClassifier`, `DecisionTreeClassifier`, `RandomizedSearchCV`) receive `random_state=42` or fold-derived deterministic seeds.

---

## 2. Environment Setup

### 2.1 Python Environment

QuadVerdict supports **Python 3.11, 3.12, 3.13, and 3.14**.

```bash
# 1. Clone repository
git clone https://github.com/abdulhayykhan/QuadVerdict.git
cd QuadVerdict

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Benchmark

### 3.1 Rapid Sanity Check Mode (`--quick`)
Executes a rapid smoke test (1 outer fold, 2 inner CV iterations) to confirm environment health in $< 30$ seconds:

```bash
python run_benchmark.py --quick
```

### 3.2 Full Publication Benchmark (15 Folds)
Executes the canonical benchmark across all 4 machine learning paradigms, 15 outer folds, and 3 inner folds:

```bash
python run_benchmark.py --resume
```

*Note on Compute Requirements:*
- **Random Forest (`rf`):** $\approx 350\text{ s}$ fit time.
- **Decision Tree (`dt`):** $\approx 18\text{ s}$ fit time.
- **Logistic Regression (`lr`):** $\approx 1,097\text{ s}$ fit time.
- **Support Vector Machine (`svm`):** $\approx 3,395\text{ s}$ fit time (due to RBF kernel Gram matrices and Platt calibration).
- Total runtime on a modern 4-core CPU is approximately 1.5 to 2 hours.
- If interrupted, the `--resume` flag automatically loads existing completed folds from `results/_cache/` without recalculating.

---

## 4. Verification and Audit Steps

### 4.1 Running the Automated Test Suite

QuadVerdict includes **136 automated regression, invariant, and leakage tests**:

```bash
pytest
```

**Key Test Suites:**
- `tests/test_pipelines_leakage.py`: Proves test sets are never transformed during training and SMOTE never alters test distributions.
- `tests/test_nested_cv_determinism.py`: Proves outer test indices have zero overlap with outer training folds.
- `tests/test_significance.py`: Validates Nadeau-Bengio corrected variance math against known analytical test vectors.
- `tests/test_metrics.py`: Verifies confusion matrix conservation invariants ($\text{TP} + \text{FN} = P$, $\text{FP} + \text{TN} = N$).
- `tests/test_export_schema.py`: Verifies compliance with `results/schema.json`.
- `tests/test_build_site.py`: Confirms site compilation, script escaping, and standalone HTML validity.

### 4.2 Verifying Output Cryptographic Hashes

After benchmark completion, verify the SHA-256 integrity of the export artifact:

```powershell
# Windows PowerShell:
Get-FileHash results/results.json -Algorithm SHA256

# Linux / macOS:
sha256sum results/results.json
```

---

## 5. Generating Publication Visualizations

To reproduce the seven 300 DPI static figures in `results/figures/`:

```bash
python scripts/generate_figures.py --results results/results.json --out-dir results/figures/
```

Figures produced:
- `cv_boxplots.png`
- `cost_curves.png`
- `roc_pr_curves.png`
- `calibration_curves.png`
- `feature_importance.png`
- `imbalance_experiment.png`
- `learning_curves.png`

---

## 6. Compiling and Verifying the Web Dashboard

Rebuild the standalone single-page HTML client:

```bash
python web/build_site.py --results results/results.json --out dist/index.html
```

You can view the compiled dashboard locally by opening `dist/index.html` in any web browser (`file:///.../dist/index.html`) or viewing the live deployment at [https://quad-verdict.vercel.app/](https://quad-verdict.vercel.app/).
