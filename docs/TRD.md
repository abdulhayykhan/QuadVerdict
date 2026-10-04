# QuadVerdict: Technical Requirements Document (TRD)

| Field | Value |
|---|---|
| Project | QuadVerdict |
| Owner | Abdul Hayy Khan |
| Status | Draft v1.0 |
| Companion doc | `PRD.md` (what and why). This document is the how. |

---

## 1. Architecture Overview

Two decoupled halves joined by one JSON file:

```
┌──────────────────────────── OFFLINE (Python) ────────────────────────────┐
│ data.py → pipelines.py → nested_cv.py → metrics/significance/explain.py  │
│                                  │                                       │
│                              export.py                                   │
│                                  ▼                                       │
│                        results/results.json                              │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ build_site.py (string injection)
                                   ▼
┌──────────────────────────── ONLINE (static) ─────────────────────────────┐
│ web/index.html (CSS+JS, placeholder) ──► dist/index.html (data embedded) │
│                                   │                                      │
│                                   ▼  Vercel static hosting               │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key decisions**

| Decision | Rationale |
|---|---|
| ML runs offline in Python only | Reimplementing sklearn in JS would be slow, wrong and uncredible |
| Browser receives threshold tables, not per-row predictions | Exact and tiny, and it enables live threshold/cost recomputation |
| Single `index.html`, no CDN | Works offline via `file://`, trivially hostable, easy to audit |
| Vanilla JS and inline SVG for charts | Satisfies "vanilla" constraint; no library lock-in |
| Vercel deploys pre-built `dist/` | Nested CV takes hours; Vercel builds are short-lived and should never run it |

**Note on "vanilla HTML/CSS":** interactivity requires JavaScript. "Vanilla" here means HTML + CSS + plain JS, with no frameworks, bundlers or libraries.

---

## 2. Repository Structure

```
quadverdict/
├── README.md
├── PRD.md
├── TRD.md
├── vercel.json
├── requirements.txt            # pinned versions
├── pyproject.toml              # pytest + ruff config
├── .gitignore
├── .gitattributes              # mark notebooks / dist as generated where relevant
├── run_benchmark.py            # one-command pipeline entry point
├── data/
│   ├── raw/                    # downloaded CSV (see §4.1 on licensing)
│   └── download.py             # fetches dataset if not present
├── src/
│   ├── __init__.py
│   ├── config.py               # SEED, cost ratio, CV settings, paths
│   ├── data.py                 # load, clean, split
│   ├── pipelines.py            # pipeline factories per model + strategy
│   ├── nested_cv.py            # outer/inner CV, OOF predictions
│   ├── metrics.py              # scalar metrics + threshold table + calibration bins
│   ├── significance.py         # Nadeau-Bengio, Wilcoxon
│   ├── explain.py              # permutation importance (+ SHAP stretch)
│   ├── experiments.py          # imbalance experiment, learning curves
│   ├── timing.py               # train/inference timing, model size
│   └── export.py               # assemble + validate + write results.json
├── notebooks/                  # exploration only (see §11)
│   ├── 01_eda.ipynb
│   ├── 02_pipeline_sanity.ipynb
│   └── 03_results_analysis.ipynb
├── web/
│   ├── index.html              # SOURCE: all CSS + JS, contains data placeholder
│   ├── mock_results.json       # hand-made, schema-valid data for UI development
│   └── build_site.py           # injects results.json into index.html → dist/
├── dist/
│   └── index.html              # BUILD OUTPUT, committed so Vercel can serve it
├── results/
│   ├── results.json            # canonical benchmark output
│   ├── schema.json             # JSON Schema for results.json
│   └── figures/                # optional static PNG exports for README
└── tests/
    ├── test_data.py
    ├── test_pipelines_leakage.py
    ├── test_nested_cv_determinism.py
    ├── test_metrics.py
    ├── test_significance.py
    ├── test_export_schema.py
    └── test_build_site.py
```

**Why two `index.html` files:** `web/index.html` is the editable source containing a placeholder token; `dist/index.html` is the generated file with real data embedded. Never edit `dist/` by hand. Both names are `index.html` so Vercel serves `dist/` with no routing config.

---

## 3. Technology Stack

| Layer | Choice | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | Pin in `requirements.txt` header comment |
| ML | scikit-learn (pin exact), imbalanced-learn | Pin versions; results can shift between releases |
| Data | pandas, numpy | |
| Stats | scipy | Wilcoxon, t distribution |
| Explain | scikit-learn `permutation_importance`; `shap` (P2) | |
| Schema | jsonschema | Validate output |
| Tests | pytest, ruff | |
| Notebooks | jupyter, nbstripout | Strip outputs before commit |
| Frontend | HTML5, CSS3 (custom properties, grid), ES2020 JS | No framework, no CDN |
| Hosting | Vercel (static) | |

`requirements.txt` must contain exact pins (`==`). Capture with `pip freeze` after the first clean install, then trim to direct dependencies plus transitive pins that affect numerics.

---

## 4. Offline Pipeline Design

### 4.1 Data (`src/data.py`, `data/download.py`)

- Load CSV; drop `ID`; rename target to `default`; rename `PAY_0` to `PAY_1` for consistency.
- Clean quirks: map `EDUCATION` ∈ {0, 5, 6} → 4 ("others"); map `MARRIAGE` 0 → 3 ("others"). Document in code and README.
- Feature groups:
  - Categorical (one-hot): `SEX`, `EDUCATION`, `MARRIAGE`
  - Numeric (scaled for LR/SVM only): `LIMIT_BAL`, `AGE`, `BILL_AMT1..6`, `PAY_AMT1..6`, `PAY_1`, `PAY_2..6` (ordinal, treated as numeric)
- **Hold-out split:** stratified 80/20 with `SEED`. The 20% hold-out is **never touched** by tuning or CV. It is used once at the end as a sanity check (a single reported score per model, shown in the README, not in the leaderboard).
- The 80% development set feeds nested CV.
- **Licensing:** confirm UCI terms. If redistribution is unclear, gitignore `data/raw/` and rely on `download.py`.

### 4.2 Pipelines (`src/pipelines.py`)

Factory: `build_pipeline(model_key, imbalance_strategy) -> Pipeline`.

Preprocessing via `ColumnTransformer`:

| Model | Categorical | Numeric |
|---|---|---|
| LR | OneHotEncoder(handle_unknown="ignore") | StandardScaler |
| SVM | OneHotEncoder | StandardScaler |
| DT | OneHotEncoder | passthrough |
| RF | OneHotEncoder | passthrough |

Imbalance strategies:

| Strategy | Mechanism |
|---|---|
| `none` | nothing |
| `balanced` | `class_weight="balanced"` on the estimator |
| `smote` | `imblearn.pipeline.Pipeline` with `SMOTE(random_state=SEED)` after preprocessing, before the estimator |

Use `imblearn.pipeline.Pipeline` everywhere so SMOTE is only applied to training folds during fit.

**Models and search spaces** (randomized search, `n_iter` from config):

| Model | Estimator | Search space |
|---|---|---|
| LR | `LogisticRegression(solver="saga", max_iter=2000)` | `C`: loguniform(1e-3, 1e2); `penalty`: {l1, l2, elasticnet}; `l1_ratio` when elasticnet |
| DT | `DecisionTreeClassifier` | `max_depth`: {3..20, None}; `min_samples_leaf`: {1..50}; `ccp_alpha`: loguniform(1e-6, 1e-2) |
| RF | `RandomForestClassifier(n_jobs=-1)` | `n_estimators`: {100..500}; `max_depth`: {5..30, None}; `max_features`: {sqrt, log2, 0.3}; `min_samples_leaf`: {1..20} |
| SVM | `CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=3)` | `C`: loguniform(1e-2, 1e2); `gamma`: loguniform(1e-4, 1e0) (set on the inner SVC via `estimator__` prefix) |

If `saga` with elasticnet proves unstable or slow, restrict penalties to {l1, l2} and document it.

**SVM runtime control (critical).** Kernel SVC scales roughly O(n²) to O(n³). With ~24k development rows, nested CV on full data is impractical. Policy:

- Config constant `SVM_MAX_TRAIN = 8000`. Inside each outer-training fold, take a stratified subsample of at most `SVM_MAX_TRAIN` rows for SVM fitting (both search and refit). Implement as a custom sklearn-compatible wrapper (`StratifiedSubsampleClassifier`) so the subsampling happens **inside** the fit and sees only training data.
- Evaluate on the full outer test fold (no subsampling on the test side).
- The README and UI must state this plainly: "SVM trained on a stratified subsample of at most 8,000 rows".
- Optional P1: also report `LinearSVC` (calibrated) on the full data as `svm_linear`, clearly labeled, so readers can see the effect of the cap. If added, extend the data contract with the extra model key.

### 4.3 Nested CV (`src/nested_cv.py`)

- **Outer:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)` → 15 outer folds.
- **Inner:** `StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)` inside `RandomizedSearchCV(scoring="average_precision", n_iter=cfg.N_ITER, refit=True, random_state=SEED)`.
- For each outer fold and each model:
  1. Run inner search on outer-train only.
  2. Predict probabilities on outer-test with the refit best estimator.
  3. Compute per-fold scalar metrics (PR-AUC, ROC-AUC, F1 at the threshold maximizing F1 on the inner OOF, MCC at the same threshold, balanced accuracy at that threshold, Brier).
  4. Store outer-test probabilities, labels and indices.
  5. Record best params and fit time.
- **Determinism:** every `random_state` derives from `config.SEED`. `n_jobs` on RF does not break determinism when `random_state` is set. Verify with a test.
- **Parallelism:** parallelize across (model, outer-fold) with `joblib`, not inside every estimator at once, to avoid oversubscription. Document the thread settings.
- **Runtime budget:** target under 4 hours total. Suggested starting values: `N_ITER = 20` (LR, DT, RF), `N_ITER = 12` (SVM). Measure one outer fold first, extrapolate (×15), then adjust and record the final values in the README.
- **Checkpointing:** persist per-fold results to `results/_cache/` (gitignored) after each fold so a crash or interrupt doesn't lose hours. `run_benchmark.py --resume` skips completed folds.

### 4.4 Out-of-Fold Aggregation and Threshold Tables (`src/metrics.py`)

With 3 repeats, each sample has 3 OOF probabilities (one per repeat). Handling:

- Build one OOF probability vector **per repeat** (each sample appears exactly once per repeat).
- For each repeat and each threshold `t` in `np.linspace(0, 1, 101)`, count TP/FP/TN/FN with prediction `p >= t`.
- **Average the counts across repeats** (floats allowed) to give the final table. Do **not** average probabilities across repeats before thresholding, since that would create an implicit ensemble no single model represents.
- Record `n_pos` and `n_neg` in meta so the UI can verify `tp + fn = n_pos`.

Derived metrics the browser recomputes from counts:

```
precision = tp / (tp + fp)            (define as 1 when tp+fp = 0, flag as undefined in UI)
recall    = tp / (tp + fn)
f1        = 2·tp / (2·tp + fp + fn)
mcc       = (tp·tn − fp·fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))   (0 when denominator is 0)
bal_acc   = (recall + tn/(tn+fp)) / 2
cost      = c_fn·fn + c_fp·fp          (c_fp = 1, c_fn = ratio)
cost_per_1k = cost / n · 1000
```

**ROC and PR curves:** compute from the pooled per-repeat OOF predictions (repeat 0 for the curve shape; per-fold AUCs carry the uncertainty). Downsample to ≤200 points using uniform sampling over the curve's cumulative arc length, keeping the endpoints.

**Calibration:** 10 quantile bins on repeat-0 OOF probabilities (`strategy="quantile"`), with `mean_pred`, `frac_pos`, `n`. Brier score per fold plus overall.

### 4.5 Significance (`src/significance.py`)

For each model pair (A, B) and the primary metric PR-AUC, with per-fold paired differences `d_i` over the same 15 outer folds:

**Nadeau-Bengio corrected resampled t-test**

```
k      = number of paired folds (15)
n_train/n_test from the outer split sizes
t      = mean(d) / sqrt( (1/k + n_test/n_train) · var(d, ddof=1) )
p      = 2 · (1 − t_cdf(|t|, df = k − 1))
```

Also report Wilcoxon signed-rank (`scipy.stats.wilcoxon`) as a non-parametric check. Report both p-values. Apply a Holm correction across the 6 pairwise comparisons and report adjusted p-values (`p_adj`).

Caveat to state in the README: with overlapping training sets across folds, no resampling test gives exact type-I error; the corrected t-test is an approximation. Words like "significant" are used at `p_adj < 0.05` only.

### 4.6 Explainability (`src/explain.py`)

- Permutation importance on each **outer test fold** with the fold's fitted best estimator, `scoring="average_precision"`, `n_repeats=5`. Aggregate mean and std across folds.
- Importance is computed on the **raw input features** (before one-hot) by running permutation through the whole pipeline, so features are comparable across models.
- Export top 15 per model.
- SHAP for DT/RF: P2, added as a separate JSON section.

### 4.7 Experiments (`src/experiments.py`)

- **Imbalance experiment:** for each model, repeat nested CV with a reduced budget (smaller `n_iter`, 5x1 outer) for each of `none`, `balanced`, `smote`. Report PR-AUC mean ± std. The main benchmark uses the winning strategy per model, chosen by the **inner** CV only, to avoid selection leakage. State this in the README. (Simpler alternative: fix the main benchmark to `balanced` and report the experiment as a side study.)
- **Learning curves:** `learning_curve` with fixed training-size fractions {0.1, 0.25, 0.5, 0.75, 1.0}, `scoring="average_precision"`, 3-fold stratified, best params from the main benchmark's most frequent configuration. Report mean train and validation scores.

### 4.8 Timing (`src/timing.py`)

- `train_s`: mean wall time of the outer-fold refit.
- `infer_ms_per_1k`: median over 10 timed runs of `predict_proba` on 1,000 held-out rows, single-process.
- `model_kb`: size of the `joblib.dumps` byte string (compressed 3).
- Record the CPU model and core count in meta so timings are interpretable.

### 4.9 Hold-out sanity check

After nested CV, refit each model with its most frequent best params on the full 80% development set, score once on the untouched 20%, and report PR-AUC and ROC-AUC in the README only (also in the JSON `holdout` section, not in the leaderboard). Large disagreement with the CV estimate must be investigated.

### 4.10 Orchestration (`run_benchmark.py`)

```
python run_benchmark.py [--resume] [--quick] [--seed 42]
```

- `--quick`: tiny budget (`n_iter=3`, 2x1 outer, subsampled data) for smoke tests and CI. Output flagged `meta.quick = true`; the UI shows a visible "QUICK RUN, NOT FOR PUBLICATION" banner when set.
- Steps: load → split → nested CV → aggregate → significance → explain → experiments → timing → holdout → export → validate.
- Logs progress with elapsed time. Writes `results/results.json` only after the schema validates.

---

## 5. Data Contract (`results/schema.json` and `results/results.json`)

JSON Schema is the source of truth; the following is the logical shape. Model keys: `lr`, `dt`, `rf`, `svm` (plus optional `svm_linear`).

```json
{
  "meta": {
    "project": "QuadVerdict",
    "dataset": "UCI Default of Credit Card Clients",
    "n": 24000,
    "n_pos": 5309,
    "n_neg": 18691,
    "pos_rate": 0.2212,
    "seed": 42,
    "outer_cv": {"type": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": 3},
    "inner_cv": {"type": "StratifiedKFold", "n_splits": 3},
    "n_iter": {"lr": 20, "dt": 20, "rf": 20, "svm": 12},
    "svm_max_train": 8000,
    "default_cost_ratio": 5,
    "primary_metric": "pr_auc",
    "quick": false,
    "versions": {"python": "", "sklearn": "", "imblearn": "", "numpy": "", "pandas": "", "scipy": ""},
    "hardware": {"cpu": "", "cores": 0},
    "built_at": "ISO-8601 UTC"
  },
  "models": {
    "lr": {
      "label": "Logistic Regression",
      "best_params": {},
      "best_params_frequency": [{"params": {}, "count": 0}],
      "cv_scores": {
        "pr_auc":   [0.0],
        "roc_auc":  [0.0],
        "f1":       [0.0],
        "mcc":      [0.0],
        "bal_acc":  [0.0],
        "brier":    [0.0],
        "accuracy": [0.0]
      },
      "cv_summary": {"pr_auc": {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}},
      "thresholds": [{"t": 0.00, "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}],
      "roc": [[0.0, 0.0]],
      "pr": [[0.0, 0.0]],
      "calibration": [{"mean_pred": 0.0, "frac_pos": 0.0, "n": 0}],
      "timing": {"train_s": 0.0, "infer_ms_per_1k": 0.0, "model_kb": 0.0},
      "importance": {"permutation": [{"feature": "", "mean": 0.0, "std": 0.0}]},
      "holdout": {"pr_auc": 0.0, "roc_auc": 0.0}
    }
  },
  "imbalance_experiment": [
    {"model": "lr", "strategy": "none|balanced|smote", "pr_auc_mean": 0.0, "pr_auc_std": 0.0}
  ],
  "significance": [
    {"a": "rf", "b": "lr", "metric": "pr_auc", "mean_diff": 0.0,
     "nadeau_bengio": {"t": 0.0, "p": 0.0, "p_adj": 0.0},
     "wilcoxon": {"stat": 0.0, "p": 0.0, "p_adj": 0.0}}
  ],
  "learning_curves": {
    "lr": {"sizes": [0], "train": [0.0], "val": [0.0], "train_std": [0.0], "val_std": [0.0]}
  }
}
```

**Validation rules (enforced in `export.py` and tests):**

- Every model has exactly 101 threshold rows; `t` strictly increasing 0.00 to 1.00.
- For every row: `tp + fn` equals `n_pos` and `fp + tn` equals `n_neg` (within 1e-6, since counts are repeat-averaged).
- `tp` and `fp` are non-increasing in `t`.
- Every `cv_scores` list has length 15; all values within valid ranges (AUCs in [0,1], MCC in [-1,1]).
- ROC and PR arrays have ≤200 points and include their endpoints.
- Six significance rows for four models (all unordered pairs).
- File size ≤ 500 KB (warn at 400 KB). Round floats to 5 decimals on export to control size.

---

## 6. Frontend Design (`web/index.html`)

### 6.1 File structure

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuadVerdict</title>
  <style> /* tokens, layout, components, dark mode */ </style>
</head>
<body>
  <header>…</header>
  <main> <section id="leaderboard">…</section> … </main>
  <noscript>This page needs JavaScript to render the interactive results.</noscript>
  <script id="results-data" type="application/json">/*__RESULTS_JSON__*/</script>
  <script> /* app code */ </script>
</body>
</html>
```

The placeholder token is exactly `/*__RESULTS_JSON__*/`. In `web/index.html` standalone (without build), the app falls back to loading `mock_results.json` **only when served over http with a `?mock=1` query**; it does not fetch anything in production. (For `file://`, use `build_site.py --mock` to bake the mock into a dev `dist`.)

Escape rule: `build_site.py` must replace `</` with `<\/` inside the injected JSON so the data can't terminate the script tag.

### 6.2 Application architecture (plain JS)

```
const DATA = JSON.parse(document.getElementById('results-data').textContent);
const state = { model:'rf', threshold:0.5, costRatio: DATA.meta.default_cost_ratio, theme:'auto', visibleModels:Set };
```

- **Pure functions** (no DOM): `metricsFromCounts(row)`, `costAt(row, ratio)`, `optimalThreshold(model, ratio)`, `bestModelAtRatio(ratio)`, `interpolateRow(model, t)`.
- **Renderers**: one function per panel, taking `(container, DATA, state)`, rebuilding SVG. Re-render only the affected panels on state change.
- **Event handling**: `input` events on sliders update state, then `requestAnimationFrame` coalesces redraws.
- **Threshold snapping:** the slider has step 0.01 matching the 101 table rows, so no interpolation is needed. `interpolateRow` exists only to guard against future finer grids.

### 6.3 Panels

| # | Panel | Chart type | Notes |
|---|---|---|---|
| 1 | Leaderboard and significance | HTML table + 4x4 matrix | mean ± std, sort by PR-AUC, matrix cells show `p_adj` and an explicit "n.s." state; text states the claim only if `p_adj < 0.05` |
| 2 | Threshold and cost lab | SVG line (cost vs t), HTML confusion matrix | two sliders, optimum marker, best-model readout, copy-link button encoding state in URL hash |
| 3 | ROC and PR | SVG lines | legend toggles; diagonal and prevalence baselines |
| 4 | Calibration | SVG scatter and line | y = x reference, bin size encoded in point radius, Brier in legend |
| 5 | CV variance | SVG box plots | median, IQR, whiskers at 1.5 IQR, jittered points |
| 6 | Imbalance experiment | SVG grouped bars | error bars = std |
| 7 | Feature importance | SVG horizontal bars | model tabs, top 10, error bars |
| 8 | Efficiency | table + SVG learning curves | train/val curves per model with std bands |
| 9 | Methodology and limitations | prose | includes the SVM subsampling disclosure |

### 6.4 Visual and interaction spec

- **Design tokens** in `:root`: `--bg`, `--surface`, `--text`, `--muted`, `--border`, `--accent`, and one categorical palette of four colors that is color-blind safe (for example Okabe-Ito: orange `#E69F00`, sky `#56B4E9`, green `#009E73`, vermillion `#D55E00`).
- Dark mode via `@media (prefers-color-scheme: dark)` with `:root:not([data-theme="light"])`, and `:root[data-theme="dark"]`; manual toggle persists in `localStorage` wrapped in try/catch.
- **Never encode meaning by color alone.** Pair with line dash patterns and direct labels.
- Layout: CSS grid, single column below 720 px, 2-column cards above; charts are SVGs with `viewBox` and `width:100%`.
- Typography: system font stack only (no web fonts), so there are no external requests.
- Numbers: tabular figures (`font-variant-numeric: tabular-nums`).

### 6.5 Accessibility requirements

- All interactive controls are native elements (`input[type=range]`, `button`, `select`) so keyboard behavior is free.
- `aria-label` on every slider; live region (`aria-live="polite"`) announcing the confusion matrix and cost after changes (debounced).
- Each SVG has `role="img"` plus `<title>`/`<desc>`, and a visually hidden data table fallback for the main charts.
- Focus outlines never removed. Contrast at least WCAG AA in both themes.
- Respect `prefers-reduced-motion`: no transitions beyond 0 ms when set.

### 6.6 Failure handling

- If the JSON is missing, still the placeholder, unparsable or fails basic shape checks (models present, 101 threshold rows), show a clear message in place of the panels: what's wrong and how to rebuild. No partial rendering of wrong data.
- If `meta.quick` is true, show the non-publication banner.

### 6.7 Performance budgets

- `dist/index.html` ≤ 1 MB (target ~600 KB including JSON).
- First render < 300 ms on a mid-range laptop; slider-drag redraw < 16 ms (one panel).
- No network requests at runtime (verify in the DevTools network tab: zero).

---

## 7. Build Script (`web/build_site.py`)

```
python web/build_site.py [--results results/results.json] [--mock] [--out dist/index.html]
```

Steps:

1. Read `web/index.html`; fail if the placeholder is absent or appears more than once.
2. Read results JSON (or `web/mock_results.json` with `--mock`), validate against `results/schema.json`.
3. Minify JSON (`separators=(",", ":")`), escape `</` → `<\/`.
4. Replace the placeholder; write `dist/index.html`.
5. Print size and warn above 1 MB.
6. Add `<!-- built: <timestamp> results-sha256: <hash> -->` comment at the top of `dist/index.html` for traceability.

`tests/test_build_site.py` asserts: placeholder replaced, JSON round-trips, no `</script>` leakage, output parses.

---

## 8. Notebooks (`notebooks/`)

Notebooks live in `notebooks/` at the repo root, next to `src/`.

| Notebook | Purpose | Rules |
|---|---|---|
| `01_eda.ipynb` | Class imbalance, distributions, correlations, outliers, dataset quirks | Imports `src.data`; no cleaning logic duplicated in the notebook |
| `02_pipeline_sanity.ipynb` | Demonstrates leakage check (fit-on-all vs in-pipeline), SVM scaling effect, SMOTE inside vs outside | Educational; mirrors what the tests assert |
| `03_results_analysis.ipynb` | Loads `results/results.json`, reproduces key figures with matplotlib, sanity-checks the UI numbers | Reads only the JSON; never retrains |

Rules:

1. **Notebooks are not the source of truth.** All logic that produces reported numbers lives in `src/` and is executed by `run_benchmark.py`.
2. Notebooks import from `src` (`pip install -e .` or `sys.path` insertion in the first cell).
3. Strip outputs before commit with `nbstripout --install`, so diffs stay clean and stale outputs don't mislead. Exception: `03_results_analysis.ipynb` may keep outputs if they were generated from the committed `results.json`.
4. Notebook figures for the README are exported to `results/figures/` by a notebook cell and committed as PNGs.
5. Notebooks are not tested in CI, but must run top to bottom with `jupyter nbconvert --execute` before release (manual checklist item).

---

## 9. Testing Strategy

| Test file | What it proves |
|---|---|
| `test_data.py` | Quirk handling; no ID column; split is stratified and seeded; hold-out disjoint from the development set |
| `test_pipelines_leakage.py` | (1) Pipelines contain scaler/SMOTE as steps; (2) fitting a pipeline on fold-train and predicting fold-test never calls `fit` on test data (spy transformer); (3) SMOTE only changes the training matrix, and the test matrix passes through unchanged; (4) trees pipelines contain no scaler |
| `test_nested_cv_determinism.py` | Two runs with `--quick` and the same seed give identical per-fold scores; outer-test indices never appear in the matching inner search data |
| `test_metrics.py` | Threshold table against hand-computed tiny arrays; MCC/F1/precision edge cases (zero denominators); `tp+fn = n_pos` invariant |
| `test_significance.py` | Nadeau-Bengio against a hand-computed example; identical score vectors give p = 1 (guarding zero variance); Holm correction on a known case |
| `test_export_schema.py` | Valid output passes; each validation rule in §5 fails when violated (mutation tests) |
| `test_build_site.py` | See §7 |

Additional manual checks (release checklist):

- Pick one model and threshold; compute confusion matrix and cost by hand from the JSON; match the UI.
- Disable network, open `dist/index.html` via `file://`; everything works.
- Keyboard-only walkthrough of the whole page.

Optional frontend test: a tiny Node script (no framework) that evaluates the pure functions extracted from the file against the mock JSON. This is not required for v1.

---

## 10. Reproducibility

- `config.SEED = 42` is the single source of randomness; passed explicitly to every estimator, splitter, SMOTE, `permutation_importance`, `RandomizedSearchCV`, and `joblib` task.
- Package versions pinned; recorded in `meta.versions`.
- `results.json` hash recorded in the build comment of `dist/index.html`.
- Document in README that bit-exact reproduction assumes the same package versions and CPU architecture; small float differences across platforms are expected.
- `.gitignore`: `__pycache__/`, `.ipynb_checkpoints/`, `results/_cache/`, `.venv/`, `data/raw/` (conditional per §4.1). **Do not ignore `dist/` or `results/results.json`.**

---

## 11. Deployment on Vercel

### 11.1 Strategy

Vercel hosts the **pre-built static file**. It never runs the Python benchmark (hours of compute, not suited to a build step) and doesn't need Python at all.

Flow: run the benchmark locally → `python web/build_site.py` → commit `dist/index.html` and `results/results.json` → push → Vercel serves `dist/`.

### 11.2 `vercel.json`

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": null,
  "buildCommand": null,
  "installCommand": null,
  "outputDirectory": "dist",
  "cleanUrls": true,
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy", "value": "no-referrer" },
        { "key": "X-Frame-Options", "value": "DENY" },
        {
          "key": "Content-Security-Policy",
          "value": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"
        }
      ]
    }
  ]
}
```

Notes:

- The CSP is strict because the page makes no external requests. `'unsafe-inline'` is required since everything is inline. If you later move to hashed inline scripts you can tighten it. Test the CSP in the browser console after deploy; if anything is blocked, loosen only the specific directive.
- `framework: null` plus null build/install commands tell Vercel to skip building and just serve `dist/`. If the dashboard overrides conflict, set them explicitly in Project Settings: **Framework Preset = Other**, **Build Command = empty**, **Output Directory = dist**.
- Verify the exact `vercel.json` property behavior against current Vercel docs when you set it up, since project config options change over time.

### 11.3 Deployment steps

**Option A, Git integration (recommended)**

1. Push the repo to GitHub with `dist/index.html` committed.
2. Vercel dashboard → *Add New Project* → import the repo.
3. Framework Preset: *Other*. Build Command: blank. Output Directory: `dist`. Install Command: blank.
4. Deploy. Every push to `main` redeploys.
5. Add a preview check: pull requests get preview URLs automatically.

**Option B, CLI**

```bash
npm i -g vercel
vercel login
vercel          # preview deployment
vercel --prod   # production deployment
```

### 11.4 Release routine

```
python run_benchmark.py            # hours; produces results/results.json
pytest                             # must pass
python web/build_site.py           # produces dist/index.html
# open dist/index.html via file:// offline; run manual checks
git add results/results.json dist/index.html && git commit && git push
```

### 11.5 Guardrails

- A CI check (GitHub Actions, optional) runs `pytest`, then `build_site.py --results results/results.json`, then fails if `dist/index.html` differs from the committed one. This guarantees `dist/` always matches the committed JSON.
- CI uses `run_benchmark.py --quick` only to smoke test the pipeline; it never overwrites `results/results.json`.
- Never deploy a `dist/` built from a `--quick` or `--mock` run. The banner in §6.6 plus a pre-push check (`meta.quick == false`) enforce this.

---

## 12. AI Coding Agent Execution Plan

Since the owner builds with an AI coding agent, work in **one phase per session**, committing after each. The TRD and PRD are the briefing documents. Hand the agent the relevant sections, not the whole repo at once.

### 12.1 Standing rules to give the agent in every session

1. Read `PRD.md` §7 (methodology) and TRD §4, §5 and the relevant section before changing code.
2. Never fit scalers, encoders, imputers or SMOTE outside a `Pipeline`.
3. Never tune and evaluate on the same folds; the hold-out set is untouched until §4.9.
4. All randomness comes from `config.SEED`.
5. Threshold tables, curves and calibration come from OOF predictions only.
6. SVM probabilities come only through `CalibratedClassifierCV`; SVM uses the capped subsample wrapper.
7. Do not fabricate numbers anywhere (code, comments, README). Results come from `run_benchmark.py` only.
8. Frontend: one file, vanilla HTML/CSS/JS, no framework, no CDN, no runtime network requests.
9. Every function in `src/` gets a test; run `pytest` and `ruff` before declaring done.
10. If a requirement is ambiguous, ask; don't assume.
11. Don't change the data contract (§5) without being asked.

### 12.2 Phases

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Skeleton, pinned `requirements.txt`, `config.py`, `pyproject.toml`, `.gitignore`, `vercel.json`, empty test files | `pip install` and `pytest` run clean |
| 1 | `data/download.py`, `data.py`, `01_eda.ipynb`, `test_data.py` | Split is seeded and stratified; quirks handled |
| 2 | `pipelines.py` + SVM subsample wrapper, `test_pipelines_leakage.py`, `02_pipeline_sanity.ipynb` | Leakage tests pass |
| 3 | `nested_cv.py` with checkpointing, `run_benchmark.py --quick`, determinism test | Two quick runs are identical |
| 4 | `metrics.py`, `significance.py` and tests | Tests match hand-computed values |
| 5 | `explain.py`, `experiments.py`, `timing.py` | Sections present in output, quick mode |
| 6 | `export.py`, `schema.json`, `test_export_schema.py` | Quick-run `results.json` validates |
| 7 | `web/index.html` against `web/mock_results.json` (hand-made, schema-valid) | All panels render; accessibility checklist passes on the mock |
| 8 | `web/build_site.py`, `test_build_site.py` | `dist/index.html` works via `file://` |
| 9 | Full benchmark run (owner runs, hours); owner reviews numbers; hold-out check | Final `results.json` produced |
| 10 | README (owner writes limitations), notebook 03, figures, Vercel deploy, optional CI | Live URL matches `dist/` |
| 11 | Stretch: multi-dataset Friedman/Nemenyi, distribution shift | Added as new JSON sections; schema versioned |

Phase 7 runs in parallel with the long benchmark by design: building the UI on a schema-valid mock prevents the UI from drifting toward invented numbers.

### 12.3 Prompt template per phase

```
CONTEXT: Read PRD.md §7 and TRD.md §<sections>. Current phase: <n>. Files already done: <list>.
TASK: <specific deliverable, file names, function signatures>.
CONSTRAINTS: <leakage rules, seed rule, data contract rule, size/perf limits>.
VERIFY: <tests to write and run; commands; what to report>.
REPORT: What you changed, test results, runtime, and any deviation from TRD.md.
```

Example (phase 3):

```
CONTEXT: Read TRD.md §4.2–4.4 and §10. Phases 0–2 are done; pipelines.py exposes build_pipeline().
TASK: Implement src/nested_cv.py with outer RepeatedStratifiedKFold(5x3), inner
RandomizedSearchCV (3-fold, average_precision), per-fold result checkpointing under
results/_cache/, and run_benchmark.py with --quick and --resume.
CONSTRAINTS: no leakage; every random_state from config.SEED; parallelize across
(model, fold) with joblib, not nested n_jobs; SVM uses the subsample wrapper.
VERIFY: pytest asserting two --quick runs give identical per-fold scores; pytest asserting
outer-test indices never appear in inner-search data. Report time per outer fold.
REPORT: Changes, test output, measured seconds per outer fold per model.
```

### 12.4 Manual review points (do not delegate)

1. Leakage: read `pipelines.py` and `nested_cv.py` yourself and confirm that SMOTE and scalers are pipeline steps and that no `fit` sees test data.
2. SVM runtime and subsampling: confirm the cap is applied inside training folds only.
3. Cost curve and confusion matrix: hand-verify one value against the JSON and the UI.
4. Significance wording: every "X beats Y" sentence needs `p_adj < 0.05` behind it.
5. Hold-out vs CV agreement.
6. Limitations section of the README: write it yourself.

---

## 13. Risks (Technical)

| Risk | Mitigation |
|---|---|
| Benchmark exceeds time budget | Measure one fold, scale ×15, adjust `n_iter`/`SVM_MAX_TRAIN`; checkpoint and `--resume` |
| Averaging threshold counts across repeats confuses readers | Explain in the methodology panel; counts are averages over 3 repeats, which is why they can be non-integers |
| Elasticnet LR instability | Fall back to {l1, l2}, document |
| SMOTE with one-hot features creates impossible rows | Note in limitations; consider `SMOTENC` (needs a categorical index) as a variant |
| Package-version drift changes results | Exact pins, recorded in meta |
| JSON injection breaks the HTML | `</` escape plus build test |
| Vercel header CSP blocks the page | Test after deploy; relax the specific directive only |
| Frontend recomputation drifts from Python metrics | Cross-check test: Python-computed metrics at t=0.5 vs the JS formulas on the same counts (done in notebook 03) |

---

## 14. Definition of Done

All items in PRD §14 plus:

- [ ] `pytest` and `ruff` clean; leakage, determinism, metrics, significance, schema and build tests pass.
- [ ] `results.json` is from a full (non-quick) run, `meta.quick == false`.
- [ ] `dist/index.html` is built from that exact `results.json` (hash in the build comment).
- [ ] Zero runtime network requests in the browser.
- [ ] Vercel production URL serves the same file as local `dist/index.html`.
- [ ] README states: the SVM subsample cap, the final `n_iter` values, the cost-ratio assumption, and the limitations.
