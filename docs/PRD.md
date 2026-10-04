# QuadVerdict: Product Requirements Document (PRD)

| Field | Value |
|---|---|
| Project | QuadVerdict, a cost-sensitive benchmark of four classifiers |
| Owner | Abdul Hayy Khan |
| Status | Draft v1.0 |
| Companion doc | `TRD.md` (technical design, data contract, execution plan) |

---

## 1. Summary

QuadVerdict compares **Logistic Regression, Decision Tree, Random Forest and SVM** on a single imbalanced credit-default dataset using rigorous methodology: leakage-free pipelines, nested cross-validation, calibration analysis, statistical significance testing and **cost-based threshold selection**. Results are published as one self-contained `index.html` (vanilla HTML/CSS/JS, no frameworks, no CDN) deployed on Vercel, where visitors can change the decision threshold and the false-negative:false-positive cost ratio and watch every metric update live.

**Core thesis:** the interesting question isn't "which model has the highest accuracy". It is "which model costs the least when a missed default is more expensive than a false alarm, and is that difference statistically real?"

## 2. Problem Statement

Most "compare 4 classifiers" projects share the same flaws:

1. They report accuracy on an imbalanced dataset (~78% accuracy is achievable by predicting "no default" for everyone).
2. They tune and evaluate on the same folds, which inflates scores.
3. They fit scalers or SMOTE before splitting (data leakage).
4. They declare a winner from a 0.3% gap with no significance test.
5. They compare at the default 0.5 threshold, which is wrong for cost-asymmetric problems.
6. The output is a static notebook nobody opens.

QuadVerdict exists to avoid all six, and to make the result inspectable by a non-technical visitor in a browser.

## 3. Goals and Non-Goals

### Goals
- G1. Produce a methodologically defensible benchmark of the four models.
- G2. Make evaluation honest: nested CV, out-of-fold predictions only, significance tests, calibration.
- G3. Compare models on **expected cost** at an adjustable threshold, not only on ranking metrics.
- G4. Ship a single-file, offline-capable, interactive results site.
- G5. Be fully reproducible with one command and a fixed seed.
- G6. Serve as a portfolio piece showing methodology depth, not model-count.

### Non-Goals
- NG1. Not a production credit-scoring system. No claims about real lending decisions.
- NG2. Not running ML inference in the browser. The site displays precomputed results.
- NG3. Not a model zoo. No gradient boosting or neural nets (may be mentioned as future work).
- NG4. Not multi-user. No accounts, backend, database or uploads.
- NG5. No frontend frameworks, bundlers or CDN dependencies.

## 4. Target Users

| Persona | Need | How the product serves them |
|---|---|---|
| **Recruiter / reviewer** | Judge skill quickly | Leaderboard, live demo link, README with limitations |
| **ML student / peer** | Learn proper benchmarking | Readable code, notebooks, documented methodology |
| **ABDI (owner)** | Portfolio, GSoC/internship credibility | Reproducible repo, deployed demo |
| **Technical interviewer** | Probe rigor | Significance matrix, leakage tests, calibration plots |

## 5. User Stories

- US1. As a visitor, I want to see all four models ranked on PR-AUC with uncertainty, so I can see who leads and whether the lead is real.
- US2. As a visitor, I want to drag a threshold slider and see the confusion matrix and metrics change, so I understand the precision/recall trade-off.
- US3. As a visitor, I want to set how much costlier a missed default is than a false alarm, and see the optimal threshold and best model change.
- US4. As a visitor, I want to see calibration plots, so I know whether predicted probabilities can be trusted.
- US5. As a visitor, I want to see when differences are **not** statistically significant, so I don't over-read the leaderboard.
- US6. As a reviewer, I want to see which features drive each model, using a method comparable across models.
- US7. As a reviewer, I want the site to work offline from a single file, so I can download and inspect it.
- US8. As the owner, I want one command to regenerate all results and the site, so the project is reproducible.

## 6. Functional Requirements

Priority: **P0** = must ship, **P1** = should ship, **P2** = stretch.

### 6.1 ML Benchmark (offline, Python)

| ID | Requirement | Priority |
|---|---|---|
| FR-ML-1 | Load and clean the UCI *Default of Credit Card Clients* dataset (30k rows, ~22% positive) | P0 |
| FR-ML-2 | Train four models: Logistic Regression, Decision Tree, Random Forest, SVM | P0 |
| FR-ML-3 | All preprocessing (scaling, encoding, SMOTE) lives inside sklearn/imblearn `Pipeline`s | P0 |
| FR-ML-4 | Hyperparameter tuning by **nested CV** (inner randomized search, outer repeated stratified k-fold) | P0 |
| FR-ML-5 | SVM probabilities obtained via calibration (`CalibratedClassifierCV`), never raw SVC scores | P0 |
| FR-ML-6 | Metrics: PR-AUC (primary), ROC-AUC, F1, MCC, balanced accuracy, Brier score; accuracy reported only to show why it misleads | P0 |
| FR-ML-7 | Threshold tables (TP/FP/TN/FN at 101 thresholds) from out-of-fold predictions only | P0 |
| FR-ML-8 | Pairwise significance tests (Nadeau-Bengio corrected t-test, Wilcoxon signed-rank) on PR-AUC | P0 |
| FR-ML-9 | Imbalance strategy experiment: none vs `class_weight="balanced"` vs SMOTE | P1 |
| FR-ML-10 | Permutation importance for all four models, on held-out folds | P1 |
| FR-ML-11 | Calibration curves (reliability bins) per model | P0 |
| FR-ML-12 | Efficiency metrics: train time, inference latency, serialized model size | P1 |
| FR-ML-13 | Learning curves per model | P1 |
| FR-ML-14 | Export everything to one `results.json` conforming to the schema in the TRD | P0 |
| FR-ML-15 | One command (`python run_benchmark.py`) reproduces everything from a fixed seed | P0 |
| FR-ML-16 | SHAP values for tree-based models | P2 |
| FR-ML-17 | Multi-dataset Friedman + Nemenyi test | P2 |
| FR-ML-18 | Distribution-shift simulation (train early rows, test later rows) | P2 |

### 6.2 Web Frontend (single `index.html`)

| ID | Requirement | Priority |
|---|---|---|
| FR-UI-1 | Single file `index.html`: all CSS and JS inline, results JSON embedded, no external requests | P0 |
| FR-UI-2 | **Leaderboard panel:** per-model PR-AUC, ROC-AUC, F1, MCC, Brier as mean ± std, plus pairwise significance matrix with a visible "not significant" state | P0 |
| FR-UI-3 | **Threshold and cost lab:** model selector, threshold slider (0 to 1), cost-ratio slider (FN:FP, 1 to 50), live confusion matrix, live metrics, cost-vs-threshold curve with the optimum marked, "best model at this cost ratio" indicator | P0 |
| FR-UI-4 | **Curves panel:** ROC and PR overlays with per-model toggles | P0 |
| FR-UI-5 | **Calibration panel:** reliability diagram per model with Brier score | P0 |
| FR-UI-6 | **CV variance panel:** box plots of per-fold scores | P1 |
| FR-UI-7 | **Imbalance experiment panel:** grouped bars (model x strategy) | P1 |
| FR-UI-8 | **Feature importance panel:** per-model permutation importance, top 10 features | P1 |
| FR-UI-9 | **Efficiency panel:** train time, latency, learning curves | P1 |
| FR-UI-10 | Dark and light theme following `prefers-color-scheme`, with a manual toggle | P1 |
| FR-UI-11 | Responsive layout (360 px to 1440 px wide) | P0 |
| FR-UI-12 | Keyboard-accessible sliders and toggles, ARIA labels, visible focus | P0 |
| FR-UI-13 | Fallback message when embedded data is missing or malformed | P0 |
| FR-UI-14 | Methodology and limitations section stated plainly on the page | P0 |
| FR-UI-15 | Charts as hand-written inline SVG, with no chart library | P0 |

### 6.3 Repository and Delivery

| ID | Requirement | Priority |
|---|---|---|
| FR-REPO-1 | Structured repo with `src/`, `notebooks/`, `tests/`, `web/`, `results/`, `dist/` | P0 |
| FR-REPO-2 | Notebooks for EDA, pipeline sanity checks and results analysis (exploration only, not the source of truth) | P0 |
| FR-REPO-3 | Unit tests for metrics, significance, leakage and export schema | P0 |
| FR-REPO-4 | README with results table, methodology, reproduction steps and honest limitations | P0 |
| FR-REPO-5 | Deployed to **Vercel** as a static site from the pre-built `dist/` | P0 |
| FR-REPO-6 | README footer per the owner's standard format | P0 |

## 7. Methodology Requirements (Non-Negotiable)

These are product requirements because the project's credibility depends on them.

1. **No leakage.** No fitting of any transformer outside a pipeline. A test must prove it.
2. **No tuning/evaluation overlap.** Hyperparameters are chosen on inner folds only.
3. **OOF-only reporting.** Every reported number comes from held-out predictions.
4. **Primary metric is PR-AUC**, justified by class imbalance.
5. **Significance before claims.** The README and UI may only say "X outperforms Y" when the test supports it.
6. **Cost before accuracy.** Final model comparison uses expected cost at each model's optimal threshold.
7. **Fixed seed.** Rerunning gives identical numbers.
8. **Documented deviations.** Any shortcut (for example SVM training-set subsampling) is disclosed in the README and on the site.

## 8. UX Overview

**Page layout (top to bottom):**

1. Header: title, one-line thesis, link to repo, theme toggle.
2. Key findings strip: 3 sentences generated from the data (winner on PR-AUC, whether significant, best model at default cost ratio).
3. Panel 1, Leaderboard and significance matrix.
4. Panel 2, Threshold and cost lab (the hero interaction).
5. Panel 3, ROC/PR curves.
6. Panel 4, Calibration.
7. Panel 5, CV variance.
8. Panel 6, Imbalance experiment.
9. Panel 7, Feature importance.
10. Panel 8, Efficiency and learning curves.
11. Methodology and limitations.
12. Footer: reproducibility info (seed, dataset, versions, build date).

**Interaction principles:** every number on screen traces to the embedded JSON. No fake smoothing, and no animation that hides values. Uncertainty (std) is always shown next to means.

## 9. Dataset

- **Source:** UCI Machine Learning Repository, *Default of Credit Card Clients* (Taiwan, 2005).
- **Size:** 30,000 rows, 23 features plus target `default payment next month`.
- **Class balance:** ~22% positive.
- **Known quirks to handle and document:** undocumented category values in `EDUCATION` (0, 5, 6) and `MARRIAGE` (0); the column named `PAY_0` (no `PAY_1`); `ID` column to drop.
- **Why this dataset:** imbalanced enough to make accuracy meaningless, small enough for a kernel SVM, tabular, and widely known so reviewers can sanity-check results.
- **Licensing:** verify the UCI license terms before committing raw data. If in doubt, gitignore it and provide a download script.

## 10. Success Metrics

### Project success
| Metric | Target |
|---|---|
| Benchmark reproducible from clean clone | `python run_benchmark.py` yields identical `results.json` hash across two runs with the same seed and package versions |
| Leakage tests | 100% pass |
| `results.json` schema validation | Pass |
| Site payload | `dist/index.html` under 1 MB |
| Offline | Opens and fully works via `file://` |
| Lighthouse (desktop) | Accessibility 90+ and Performance 90+ |
| Claims audit | Zero README/UI claims unsupported by the significance tests |

### Portfolio success (soft)
- The repo's README can be understood in 3 minutes.
- The live demo loads in under 2 seconds on a typical connection.

## 11. Constraints and Assumptions

- Runs on a laptop-class machine. The full benchmark should finish in **under about 4 hours**; if not, reduce `n_iter` or the SVM subsample cap and document it.
- The owner is a BS-AI student: the stack is Python 3.11+, scikit-learn and imbalanced-learn, with plain web tech for the frontend.
- Development uses an AI coding agent, one phase per session.
- No budget for hosting beyond Vercel's free tier.

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kernel SVM nested CV takes many hours | High | High | Cap SVM training subsample; limit `n_iter`; document it |
| Silent leakage introduced by the agent | Medium | Very high | Dedicated leakage tests; manual review of `pipelines.py` and `nested_cv.py` |
| Differences between models are not significant | High | Low | Report it honestly. It is itself a finding |
| UI built on invented numbers | Medium | High | Build the UI against a mock JSON validated by the same schema |
| `results.json` too large for a single file | Medium | Medium | Threshold tables instead of per-row predictions; downsample curves |
| README overclaims | Medium | High | Owner writes the limitations section by hand |
| Scope creep into P2 items | High | Medium | P2 items only after all P0/P1 are done |
| Vercel build tries to run Python | Low | Medium | Commit `dist/`; no build command on Vercel |

## 13. Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| M0 | Repo skeleton and pinned environment | `pip install -r requirements.txt` works, `pytest` runs |
| M1 | Data and EDA | Seeded stratified split, EDA notebook complete |
| M2 | Pipelines | Leakage test passes |
| M3 | Nested CV | Deterministic across runs |
| M4 | Metrics and significance | Unit tests match hand-computed values |
| M5 | Explainability, imbalance experiment, learning curves | Outputs present in the data contract |
| M6 | Export | `results.json` validates, under 500 KB |
| M7 | Frontend on mock data | All panels render |
| M8 | Frontend on real data | `dist/index.html` works offline |
| M9 | Docs and deployment | Live on Vercel, README final |
| M10 (stretch) | Multi-dataset and distribution shift | Added as new JSON sections |

## 14. Acceptance Criteria (Release Checklist)

- [ ] All P0 requirements implemented; P1 items implemented or explicitly deferred in README.
- [ ] `pytest` passes, including leakage, determinism, metrics and schema tests.
- [ ] `run_benchmark.py` completes and produces `results/results.json`.
- [ ] `python web/build_site.py` produces `dist/index.html` embedding the JSON.
- [ ] Spot check: one threshold's confusion matrix and cost computed by hand matches the UI.
- [ ] Every statement of the form "X beats Y" is backed by a significance result.
- [ ] Site works via `file://` with network disabled.
- [ ] Keyboard-only navigation works for all sliders and toggles.
- [ ] Live Vercel URL loads and matches the local `dist/index.html`.
- [ ] README includes results table, methodology, limitations, reproduction steps and the standard footer.

## 15. Open Questions

1. Cost ratio default: 5:1 is assumed. Is a different default more defensible for the story (say 10:1)? The slider covers both, but the headline number depends on it.
2. SVM variant: RBF on a capped subsample (more interesting, less data) or LinearSVC on the full set (faster, less interesting)? The TRD defaults to RBF-on-subsample, and the README must state it.
3. Custom domain or default `*.vercel.app` URL?
4. Should the stretch Friedman/Nemenyi multi-dataset analysis be in scope for v1, or a v1.1 release?

## 16. Out of Scope for v1 (Future Work)

Gradient boosting comparison, fairness analysis across `SEX`/`AGE`, drift monitoring, a live "paste a row, get a prediction" demo (would require a backend or ONNX-in-browser), and multi-dataset benchmarking.
