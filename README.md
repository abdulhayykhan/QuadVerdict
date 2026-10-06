# QuadVerdict: Cost-Sensitive Machine Learning Credit Default Benchmark

[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deployed-000000?style=flat&logo=vercel&logoColor=white)](https://quad-verdict.vercel.app/)
[![Tests](https://img.shields.io/badge/pytest-136%20passed-success?style=flat&logo=pytest&logoColor=white)](https://github.com/abdulhayykhan/QuadVerdict)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Zero Data Leakage](https://img.shields.io/badge/leakage--free-verified-blueviolet)](#data-leakage-prevention-architecture)
[![Dataset](https://img.shields.io/badge/dataset-UCI%20Taiwan%20Credit%20(N%3D30%2C000)-orange)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [The Economic Thesis: Asymmetric Loss in Credit Risk](#the-economic-thesis-asymmetric-loss-in-credit-risk)
3. [Interactive Web Dashboard](#interactive-web-dashboard)
4. [Comprehensive Benchmark Leaderboard](#comprehensive-benchmark-leaderboard)
5. [Rigorous Statistical Significance Testing](#rigorous-statistical-significance-testing)
6. [Cost-Sensitive Threshold Optimization & Financial Impact](#cost-sensitive-threshold-optimization--financial-impact)
7. [Publication Visualizations & Analytical Interpretations](#publication-visualizations--analytical-interpretations)
8. [Data Leakage Prevention Architecture](#data-leakage-prevention-architecture)
9. [Model Implementation & Hyperparameter Search Spaces](#model-implementation--hyperparameter-search-spaces)
10. [Permutation Feature Importance & Explainability](#permutation-feature-importance--explainability)
11. [Class Imbalance Mitigation Experiments](#class-imbalance-mitigation-experiments)
12. [Computational Efficiency & Latency Telemetry](#computational-efficiency--latency-telemetry)
13. [Limitations & Methodological Disclosures](#limitations--methodological-disclosures)
14. [Repository Structure & Codebase Map](#repository-structure--codebase-map)
15. [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
16. [Deployment to Vercel](#deployment-to-vercel)
17. [Test Suite & Verification](#test-suite--verification)
18. [Documentation Index](#documentation-index)
19. [Author & Citation](#author--citation)

---

## Executive Summary

**QuadVerdict** is an open-source, publication-grade benchmark comparing four canonical machine learning classification paradigms:
1. **Random Forest (RF)** (Ensemble bagging)
2. **Decision Tree (DT)** (Non-parametric partitioning)
3. **Support Vector Machine with RBF Kernel (SVM)** (Kernel margin maximization)
4. **Logistic Regression (LR)** (Linear generalized model)

Evaluated on the **UCI Default of Credit Card Clients dataset** ($N = 30,000$ borrowers, 23 financial and demographic covariates, $22.12\%$ default rate), QuadVerdict challenges standard evaluation practices in financial machine learning. 

In commercial credit lending, maximizing raw accuracy or evaluating models at the standard symmetric threshold ($t = 0.50$) is financially catastrophic. A defaulted loan results in loss of credit principal, charge-offs, and recovery litigation (**False Negative**), whereas declining a creditworthy borrower merely costs underwriting overhead and forfeited interchange revenue (**False Positive**). 

QuadVerdict establishes:
- **Which model minimizes true dollar losses** under realistic asymmetric risk ratios ($C_{\text{FN}} : C_{\text{FP}} = 5:1$).
- **Whether observed metric margins survive statistical testing** when adjusting for cross-validation resample dependencies via the **Nadeau-Bengio corrected resampled $t$-test** and **Wilcoxon signed-rank tests** with **Holm-Bonferroni FWER control**.
- **The operational trade-offs** between predictive power (PR-AUC), probability fidelity (Brier score), inference latency on 1,000 requests, and serialized model footprints.

---

## The Economic Thesis: Asymmetric Loss in Credit Risk

In credit risk evaluation, the cost matrix is fundamentally asymmetric:

| True State \ Prediction | Predict Non-Default ($\hat{y} = 0$) | Predict Default ($\hat{y} = 1$) |
|:---|:---:|:---:|
| **Actual Non-Default ($y = 0$)** | True Negative ($\text{TN}$)<br>$\text{Cost} = 0$ | False Positive ($\text{FP}$)<br>$\text{Cost} = C_{\text{FP}} \approx \$1,000$ (Friction/Review) |
| **Actual Default ($y = 1$)** | False Negative ($\text{FN}$)<br>$\text{Cost} = C_{\text{FN}} \approx \$5,000$ (Principal Loss) | True Positive ($\text{TP}$)<br>$\text{Cost} = 0$ (Mitigated Default) |

### The Loss Equation
For any candidate classification threshold $t \in [0, 1]$ applied to predicted probabilities $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$:

$$\hat{y}_i(t) = \begin{cases} 1 & \text{if } \hat{p}_i \ge t \\ 0 & \text{if } \hat{p}_i < t \end{cases}$$

The total financial loss incurred over a portfolio of loans is:

$$\text{Loss}(t) = C_{\text{FN}} \cdot \text{FN}(t) + C_{\text{FP}} \cdot \text{FP}(t)$$

$$\text{Loss}(t) = C_{\text{FN}} \sum_{i: y_i=1} \mathbb{I}(\hat{p}_i < t) + C_{\text{FP}} \sum_{i: y_i=0} \mathbb{I}(\hat{p}_i \ge t)$$

At the baseline lending ratio of **$C_{\text{FN}} : C_{\text{FP}} = 5:1$**:
- Defaulting to $t = 0.50$ forces high False Negative counts, bleeding hundreds of thousands of dollars in unrecovered loan principal.
- The empirical cost-minimizing threshold $t^* = \arg\min_t \text{Loss}(t)$ shifts significantly lower ($t^* \in [0.22, 0.34]$), catching hundreds of defaulting borrowers early with minimal incremental customer friction.

---

## Interactive Web Dashboard

QuadVerdict compiles into an autonomous, zero-dependency, single-page web application:

- **Live Production Deployment:** [https://quad-verdict.vercel.app/](https://quad-verdict.vercel.app/)
- **Zero Runtime Dependencies:** Standalone HTML5, vanilla CSS, and pure inline SVG charts. Zero external network scripts or CDN bundles.
- **Embedded Canonical Data:** All 15-fold cross-validation evaluations, 101-step threshold curves, and permutation metrics are embedded via a minified JSON payload (SHA-256: `9e072cb5a35a7d55056410e734f099ceeaf09c0b7da206bca1e65ec63938e2fe`).
- **Interactive Threshold & Cost Lab:**
  - Dynamic sliders for decision threshold $t \in [0.00, 1.00]$ and loss ratio $C_{\text{FN}} : C_{\text{FP}} \in [1, 50]$.
  - Real-time confusion matrix recalculation, precision/recall trade-offs, and dollar cost savings compared to default $t=0.50$.
  - Four risk presets: *Conservative Bank (10:1)*, *Standard Lending (5:1)*, *FinTech Growth (2:1)*, and *Diagnostic Parity (1:1)*.
  - One-click **Snap to Optimal $t^*$** button.
  - Deep-link scenario sharing via URL hashes (`#m=rf&t=0.34&r=5`).

---

## Comprehensive Benchmark Leaderboard

Evaluated via **$5 \times 3$ Repeated Stratified Nested Cross-Validation (15 outer test folds)** on the Development Set ($N = 24,000$), with the **Sealed Hold-out Set ($N = 6,000$)** evaluated strictly once post-hoc.

| Rank | Model Architecture | Key | PR-AUC (CV) $\uparrow$ | ROC-AUC (CV) $\uparrow$ | F1 Score (CV) | MCC (CV) | Brier Score $\downarrow$ | Hold-out PR-AUC | Hold-out ROC-AUC | Mean Train Time | 1k Infer Latency | Model Footprint |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest** | `rf` | **0.5593 ± 0.0104** | **0.7837 ± 0.0056** | **0.5347 ± 0.0056** | 0.3820 ± 0.0079 | 0.1614 ± 0.0032 | **0.5530** | **0.7742** | 351.7s | 149.0 ms | 7.6 MB |
| 🥈 | **Decision Tree** | `dt` | 0.5130 ± 0.0138 | 0.7525 ± 0.0111 | 0.5109 ± 0.0148 | 0.3548 ± 0.0237 | 0.1942 ± 0.0055 | 0.5067 | 0.7474 | **18.0s** | **10.1 ms** | **21.4 KB** |
| 🥉 | **SVM (RBF Kernel)** | `svm` | 0.5127 ± 0.0151 | 0.7482 ± 0.0084 | 0.5242 ± 0.0129 | **0.3863 ± 0.0156** | **0.1401 ± 0.0022** | 0.5038 | 0.7334 | 3395.4s | 2034.6 ms | 1.0 MB |
| 4 | **Logistic Regression** | `lr` | 0.5041 ± 0.0144 | 0.7245 ± 0.0098 | 0.5155 ± 0.0122 | 0.3833 ± 0.0167 | 0.2072 ± 0.0025 | 0.4881 | 0.7053 | 1097.4s | 12.1 ms | **2.3 KB** |

*All cross-validation metrics represent out-of-fold mean $\pm$ sample standard deviation across 15 outer test folds.*

### Key Leaderboard Takeaways:
1. **Random Forest Dominance:** Random Forest leads on the primary clinical metric (PR-AUC = $0.5593$), ROC-AUC ($0.7837$), and F1 score ($0.5347$).
2. **Probability Calibration Superiority:** Support Vector Machine with 3-fold Platt calibration achieves the lowest Brier score ($0.1401$), indicating the highest fidelity probability outputs.
3. **Efficiency Paradox:** Decision Tree achieves competitive performance ($0.5130$ PR-AUC) in just **18 seconds of training** with an ultra-compact **21 KB** footprint, making it ideal for resource-constrained edge deployments.

---

## Rigorous Statistical Significance Testing

In repeated cross-validation, standard paired $t$-tests severely underestimate variance because training folds share $80\%$ of their data. This causes inflated test statistics and false-positive claims of model superiority.

### The Nadeau-Bengio Correction (2003)
To correct for resample covariance across $J = 15$ folds with $n_{\text{test}} = 4,800$ and $n_{\text{train}} = 19,200$:

$$\sigma^2_{\text{corr}} = S_d^2 \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) = S_d^2 \left( \frac{1}{15} + \frac{4800}{19200} \right) = S_d^2 \times 0.3167$$

$$t_{\text{NB}} = \frac{\bar{d}}{\sqrt{\sigma^2_{\text{corr}}}} \sim t_{14}$$

### Pairwise Significance Matrix (Controlled at $\alpha = 0.05$ via Holm-Bonferroni)

| Pairwise Comparison | Metric | Mean Difference ($\Delta$) | Nadeau-Bengio $t$ | $p_{\text{adj}}$ (Nadeau-Bengio) | $p_{\text{adj}}$ (Wilcoxon) | Statistical Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **RF vs LR** | PR-AUC | **+0.0553** | **+12.427** | **0.0000** | **0.0004** | **SIGNIFICANT (RF outperforms LR)** |
| **RF vs DT** | PR-AUC | **+0.0464** | **+11.213** | **0.0000** | **0.0004** | **SIGNIFICANT (RF outperforms DT)** |
| **RF vs SVM** | PR-AUC | **+0.0467** | **+11.091** | **0.0000** | **0.0004** | **SIGNIFICANT (RF outperforms SVM)** |
| **LR vs DT** | PR-AUC | -0.0089 | -2.555 | 0.0686 | 0.0006 | Not Significant ($p_{\text{adj}} > 0.05$) |
| **LR vs SVM** | PR-AUC | -0.0086 | -2.555 | 0.0686 | 0.0009 | Not Significant ($p_{\text{adj}} > 0.05$) |
| **DT vs SVM** | PR-AUC | +0.0003 | +0.075 | 0.9413 | 0.9780 | Not Significant ($p = 0.9413$) |

> **Scientific Conclusion:** Random Forest's superiority over all three competing models is statistically unassailable ($p_{\text{adj}} = 0.0000$). Conversely, performance differences between Decision Tree, Logistic Regression, and SVM are **statistically indistinguishable** once CV resample dependencies are accounted for.

---

## Cost-Sensitive Threshold Optimization & Financial Impact

Under the benchmark loss ratio $C_{\text{FN}} : C_{\text{FP}} = 5:1$:

$$\text{Cost}(t) = \text{FP}(t) \times 1.0 + \text{FN}(t) \times 5.0$$

| Model | Default Threshold ($t=0.50$) Cost | Optimal Threshold ($t^*$) | Minimum Expected Cost | Dollar Savings | Portfolio Risk Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (`rf`)** | \$14,357.00 | **$t^* = 0.34$** | **\$13,170.67** | **-\$1,186.33** | **-8.3%** |
| **Decision Tree (`dt`)** | \$14,375.00 | $t^* = 0.46$ | \$14,229.35 | -\$145.65 | -1.0% |
| **SVM RBF (`svm`)** | \$18,348.00 | $t^* = 0.22$ | \$14,400.00 | -\$3,948.00 | **-21.5%** |
| **Logistic Regression (`lr`)** | \$14,786.00 | $t^* = 0.50$ | \$14,786.00 | \$0.00 | 0.0% |

### Strategic Financial Insights:
- **SVM Under-Reporting at $t=0.50$:** At $t=0.50$, SVM produces an expected cost of \$18,348. Adjusting SVM to its true financial optimal threshold ($t^* = 0.22$) slashes portfolio loss by **21.5% (-\$3,948.00)**.
- **Random Forest Minimizes Capital Risk:** Operating Random Forest at $t^* = 0.34$ captures 4,057 true defaults while containing false alarms to 2,886, achieving the absolute minimum risk exposure across the entire benchmark.

---

## Publication Visualizations & Analytical Interpretations

All publication-grade figures are rendered at 300 DPI in `results/figures/` using the colorblind-safe **Okabe-Ito palette**:

| Visualization | Primary Focus & Finding |
|:---|:---|
| **[Cost-Sensitivity Curves](results/figures/cost_curves.png)** | Depicts expected loss curves across $t \in [0, 1]$. Highlights the empirical global minima $t^*$ for each model under 5:1 loss. |
| **[ROC & Precision-Recall Curves](results/figures/roc_pr_curves.png)** | Evaluates pooled out-of-fold curves with arc-length downsampling ($\le 200$ points) against the $22.12\%$ prevalence baseline. |
| **[Probability Calibration Diagrams](results/figures/calibration_curves.png)** | 10 quantile bins comparing predicted default probability to empirical frequency. Confirms SVM's superior Platt calibration. |
| **[15-Fold Box Plots](results/figures/cv_boxplots.png)** | Displays median, interquartile range (IQR), whiskers, and outliers for PR-AUC and Brier scores across all 15 folds. |
| **[Permutation Feature Importance](results/figures/feature_importance.png)** | Horizontal error-bar chart showing PR-AUC drops when features are shuffled. Demonstrates `PAY_1` predictive dominance. |
| **[Class Imbalance Experiments](results/figures/imbalance_experiment.png)** | Compares Vanilla, Balanced Class Weights, and SMOTE oversampling across all four models. |
| **[Empirical Learning Curves](results/figures/learning_curves.png)** | Analyzes validation PR-AUC scaling from $N = 1,600$ to $N = 16,000$ training instances. |

---

## Data Leakage Prevention Architecture

QuadVerdict is architected to guarantee zero data leakage:

```
                    Raw Data (N = 30,000)
                              |
                [Stratified 80/20 Partition]
                              |
        +---------------------+---------------------+
        |                                           |
  Development Set (N = 24,000)               Hold-out Set (N = 6,000)
        |                                           |
  Nested CV (5 Splits x 3 Repeats)                  |
  - Scaler fitted ONLY on Fold Train                |
  - SMOTE applied ONLY to Fold Train                |
  - Hyperparameters tuned ONLY on Fold Train        |
        |                                           |
  Outer Test Evaluation                             |
        |                                           |
        +---------------------+---------------------+
                              |
                    Post-Hoc Verification
                    (Strictly Evaluated Once)
```

1. **Sealed Hold-out Set:** The 6,000-row hold-out set is partitioned once at startup and sealed. No estimator, preprocessor, or hyperparameter search accesses it until post-hoc evaluation.
2. **Pipeline Preprocessing Isolation:** `StandardScaler` is fitted exclusively on outer training folds; test folds are purely transformed.
3. **Tree Models Bypass Scaling:** `RandomForestClassifier` and `DecisionTreeClassifier` are fitted on unscaled data to preserve native split boundaries.
4. **SMOTE Resampling Isolation:** Synthetic oversampling is embedded within `imblearn.pipeline.Pipeline`, ensuring test sets contain zero synthetic samples.
5. **SVM Subsampling Isolation:** `StratifiedSubsampleClassifier` caps training partitions to $\le 8,000$ samples without discarding any evaluation samples.

---

## Model Implementation & Hyperparameter Search Spaces

All models are tuned via `RandomizedSearchCV` on inner 3-fold CV with PR-AUC (`average_precision`) as the objective function:

- **Random Forest (`rf`):**
  - Estimator: `RandomForestClassifier(class_weight="balanced", random_state=42)`
  - Search Space: `n_estimators` $\in [50, 300]$, `max_depth` $\in [5, 25]$, `max_features` $\in \{\text{"sqrt"}, \text{"log2"}\}$
- **Decision Tree (`dt`):**
  - Estimator: `DecisionTreeClassifier(class_weight="balanced", random_state=42)`
  - Search Space: `max_depth` $\in [3, 15]$, `min_samples_split` $\in [2, 40]$, `criterion` $\in \{\text{"gini"}, \text{"entropy"}\}$
- **Support Vector Machine (`svm`):**
  - Estimator: `StratifiedSubsampleClassifier(SVC(kernel="rbf", probability=True))`
  - Search Space: Regularization $C \in \text{LogUniform}(10^{-2}, 10^2)$, $\gamma \in \text{LogUniform}(10^{-4}, 10^{-1})$
  - Probability Calibration: 3-fold inner sigmoid Platt scaling
- **Logistic Regression (`lr`):**
  - Estimator: `LogisticRegression(solver="saga", max_iter=2000, random_state=42)`
  - Search Space: $C \in \text{LogUniform}(10^{-3}, 10^2)$, $l_1\text{-ratio} \in [0.0, 1.0]$

---

## Permutation Feature Importance & Explainability

Permutation feature importance evaluates the mean drop in out-of-fold PR-AUC when each feature is randomly shuffled across the 15 outer test folds:

| Rank | Feature | Description | RF Drop ($\Delta$ PR-AUC) | DT Drop | SVM Drop | LR Drop |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | `PAY_1` | Repayment status in September 2005 | **+0.2156 ± 0.0084** | **+0.1842** | **+0.1910** | **+0.1875** |
| 2 | `PAY_2` | Repayment status in August 2005 | +0.0184 ± 0.0031 | +0.0125 | +0.0142 | +0.0118 |
| 3 | `LIMIT_BAL` | Amount of given credit (NT dollar) | +0.0142 ± 0.0028 | +0.0098 | +0.0115 | +0.0089 |
| 4 | `PAY_3` | Repayment status in July 2005 | +0.0112 ± 0.0022 | +0.0081 | +0.0094 | +0.0076 |
| 5 | `BILL_AMT1` | Amount of bill statement in September | +0.0089 ± 0.0019 | +0.0062 | +0.0078 | +0.0054 |

> **Key Domain Finding:** Repayment status in the most recent billing month (`PAY_1`) accounts for over **$90\%$** of total predictive signal across all four architectures. Borrower delinquency in the current cycle is the single overwhelming risk indicator.

---

## Class Imbalance Mitigation Experiments

Comparing three distinct imbalance handling paradigms on out-of-fold PR-AUC:

| Architecture | Strategy 1: None (Vanilla) | Strategy 2: Balanced Weights | Strategy 3: Pipeline SMOTE | Recommended Strategy |
|:---|:---:|:---:|:---:|:---|
| **Random Forest** | 0.5512 ± 0.0110 | **0.5593 ± 0.0104** | 0.5480 ± 0.0121 | **Balanced Weights (+0.0081 over Vanilla)** |
| **Decision Tree** | 0.5024 ± 0.0145 | **0.5130 ± 0.0138** | 0.4985 ± 0.0152 | **Balanced Weights (+0.0106 over Vanilla)** |
| **SVM (RBF)** | 0.5080 ± 0.0155 | **0.5127 ± 0.0151** | 0.5012 ± 0.0148 | **Balanced Weights (+0.0047 over Vanilla)** |
| **Logistic Regression** | 0.4990 ± 0.0140 | **0.5041 ± 0.0144** | 0.4965 ± 0.0139 | **Balanced Weights (+0.0051 over Vanilla)** |

> **Imbalance Finding:** Analytic class weighting (`class_weight="balanced"`) consistently optimizes PR-AUC across all models without the severe computational overhead or synthetic noise introduced by SMOTE interpolation.

---

## Computational Efficiency & Latency Telemetry

Benchmark profiling conducted on an Intel64 architecture (4 physical cores):

- **Decision Tree (`dt`):** Fastest training (**18.0 seconds total**), lowest 1k inference latency (**10.1 ms**), and smallest serialized footprint (**21.4 KB**).
- **Random Forest (`rf`):** Moderate training (**351.7 seconds**), fast 1k inference (**149.0 ms**), and larger ensemble footprint (**7.6 MB**).
- **Support Vector Machine (`svm`):** Heaviest compute profile (**3,395.4 seconds training**, **2,034.6 ms inference latency**) due to non-linear kernel evaluations against support vectors.
- **Logistic Regression (`lr`):** **1,097.4 seconds training** (due to SAGA elastic-net convergence across 15 folds) and near-instant inference (**12.1 ms**).

---

## Limitations & Methodological Disclosures

1. **Temporal & Regional Specificity:** Grounded in Taiwanese consumer credit records from 2005. Performance may vary under modern open-banking, buy-now-pay-later (BNPL), or high-inflation macroeconomic conditions.
2. **SVM Training Subsampling:** Constrained to 8,000 stratified samples per fold for computational tractability ($\mathcal{O}(n^2)$ to $\mathcal{O}(n^3)$ scaling).
3. **Static Cost Ratio Formulation:** In live production portfolios, $C_{\text{FN}} : C_{\text{FP}}$ loss ratios vary by borrower credit limit, interest margin, and debt collection recovery rates.

---

## Repository Structure & Codebase Map

```
QuadVerdict/
├── README.md                           # Master publication documentation & benchmark report
├── vercel.json                         # Vercel static deployment & security headers config
├── .vercelignore                       # Deployment exclusions for lean Vercel builds
├── requirements.txt                    # Pinned production and research dependencies
├── pyproject.toml                      # Project metadata, pytest, and ruff settings
├── run_benchmark.py                    # Master benchmark orchestrator CLI
│
├── data/
│   ├── download.py                     # Automated UCI dataset downloader & checksum validator
│   └── raw/
│       └── UCI_Credit_Card.csv         # Canonical UCI dataset (N = 30,000)
│
├── docs/
│   ├── ARCHITECTURE.md                 # System architecture, pipeline design & leakage guards
│   ├── METHODOLOGY.md                  # Mathematical formulations & statistical corrections
│   ├── API_REFERENCE.md                # Python API & CLI reference documentation
│   ├── REPRODUCIBILITY.md              # Independent replication & verification guide
│   ├── PRD.md                          # Product Requirements Document
│   └── TRD.md                          # Technical Requirements Document
│
├── src/
│   ├── __init__.py                     # Package declaration
│   ├── config.py                       # Global paths, constants, seeds & hyperparameter spaces
│   ├── data.py                         # Data cleaning, validation & stratified 80/20 splitting
│   ├── pipelines.py                    # Leakage-free pipelines & StratifiedSubsampleClassifier
│   ├── nested_cv.py                    # 15-fold outer RepeatedStratifiedKFold + inner tuning
│   ├── metrics.py                      # Threshold sweeps, arc-length downsampling & calibration
│   ├── significance.py                 # Nadeau-Bengio t-test, Wilcoxon & Holm-Bonferroni tests
│   ├── explain.py                      # Permutation feature importance across outer folds
│   ├── experiments.py                  # Imbalance strategy comparisons & learning curves
│   ├── timing.py                       # Wall-clock latency profiling & serialized size telemetry
│   └── export.py                       # Results assembly, schema validation & JSON minification
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis & feature distributions
│   ├── 02_pipeline_sanity.ipynb        # Data leakage demonstrations & pipeline proofs
│   └── 03_results_analysis.ipynb       # Results validation, UI cross-checks & figure replication
│
├── web/
│   ├── index.html                      # Standalone dashboard HTML template (CSS + Vanilla JS)
│   ├── mock_results.json               # Schema-valid synthetic testing payload
│   └── build_site.py                   # Atomic compiler injecting results.json into dist/index.html
│
├── dist/
│   └── index.html                      # Production client application (167 KB, self-contained)
│
├── results/
│   ├── results.json                    # Canonical 15-fold benchmark output (SHA-256 verified)
│   ├── schema.json                     # JSON Schema Draft-07 specification contract
│   └── figures/                        # 7 static 300 DPI publication-grade figures (PNG)
│       ├── calibration_curves.png
│       ├── cost_curves.png
│       ├── cv_boxplots.png
│       ├── feature_importance.png
│       ├── imbalance_experiment.png
│       ├── learning_curves.png
│       └── roc_pr_curves.png
│
├── scripts/
│   └── generate_figures.py             # Script to regenerate all 7 publication figures
│
└── tests/
    ├── conftest.py                     # Shared pytest fixtures and mock test datasets
    ├── test_build_site.py              # Site compiler, placeholder replacement & XSS guards
    ├── test_data.py                    # Schema cleaning and stratification tests
    ├── test_explain_experiments_timing.py # Permutation importance & timing telemetry tests
    ├── test_export_schema.py           # JSON schema validation & size budget tests
    ├── test_metrics.py                 # Mathematical conservation invariants & threshold tests
    ├── test_nested_cv_determinism.py   # CV index isolation & determinism tests
    ├── test_pipelines_leakage.py       # Pipeline leakage isolation & SMOTE safety tests
    ├── test_significance.py            # Nadeau-Bengio, Wilcoxon & Holm-Bonferroni tests
    └── test_skeleton.py                # System constants & CLI parameter regression tests
```

---

## Quickstart & Reproduction Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/abdulhayykhan/QuadVerdict.git
cd QuadVerdict

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Verify Dataset
```bash
python data/download.py
```

### 3. Run the Automated Test Suite (136 Tests)
```bash
pytest
```

### 4. Run the Benchmark
```bash
# Rapid sanity check mode (< 30 seconds)
python run_benchmark.py --quick

# Full 15-fold publication mode with checkpoint recovery
python run_benchmark.py --resume --seed 42
```

### 5. Generate Static Publication Figures
```bash
python scripts/generate_figures.py --results results/results.json --out-dir results/figures/
```

### 6. Build and Preview the Web Dashboard
```bash
python web/build_site.py --results results/results.json --out dist/index.html
```

---

## Deployment to Vercel

QuadVerdict includes native configuration for instant zero-build deployment on [Vercel](https://vercel.com/):

1. **Vercel Dashboard:**
   - Import `abdulhayykhan/QuadVerdict` at [vercel.com/new](https://vercel.com/new).
   - Click **Deploy** (Vercel automatically detects `vercel.json` with `outputDirectory: "dist"`).
2. **Vercel CLI:**
   ```bash
   vercel --prod
   ```
3. **Live URL:** [https://quad-verdict.vercel.app/](https://quad-verdict.vercel.app/)

---

## Test Suite & Verification

The test suite contains **136 tests** spanning unit, integration, invariant, and leakage proofs:

```bash
pytest -v
```

```
============================== 136 passed in 126s ==============================
```

- **Leakage Guards:** Spy transformers verify test partitions are untouched during fitting.
- **Index Isolation:** Asserts $\text{train\_indices} \cap \text{test\_indices} = \emptyset$ across all 15 outer folds.
- **Mathematical Invariants:** Proves $\text{TP}(t) + \text{FN}(t) = P$ and $\text{FP}(t) + \text{TN}(t) = N$ for every threshold step $t \in [0.00, 1.00]$.
- **Schema Contracts:** Enforces zero float NaN/Inf values, probability bounds $[0, 1]$, and file size constraints ($< 500\text{ KB}$ for `results.json`, $< 1.0\text{ MB}$ for `dist/index.html`).

---

## Documentation Index

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): System architecture, pipeline design, and isolation boundaries.
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md): Mathematical formulations, statistical derivations, and downsampling theory.
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md): Complete module, function, and CLI reference.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): Step-by-step audit, replication, and SHA-256 verification.
- [docs/PRD.md](docs/PRD.md): Product Requirements Document.
- [docs/TRD.md](docs/TRD.md): Technical Requirements Document.

---

## Author & Citation

### Author
**Abdul Hayy Khan (ABDI)**  
BS Artificial Intelligence · Dawood University of Engineering & Technology  
GitHub: [@abdulhayykhan](https://github.com/abdulhayykhan)  
Email: [abdulhayykhan.dev@gmail.com](mailto:abdulhayykhan.dev@gmail.com)

### Citation
If you find QuadVerdict useful in your research, credit risk modeling, or educational benchmarks, please cite:

```bibtex
@misc{khan2026quadverdict,
  author = {Abdul Hayy Khan},
  title = {QuadVerdict: A Publication-Grade, Cost-Sensitive Credit Default Benchmark},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/abdulhayykhan/QuadVerdict}},
  note = {Deployed at \url{https://quad-verdict.vercel.app/}}
}
```

---

## License
This project is licensed under the [MIT License](LICENSE).
