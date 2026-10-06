# QuadVerdict System Architecture

This document details the architectural design, data pipelines, isolation boundaries, and execution models of the **QuadVerdict** credit default benchmark.

---

## 1. High-Level System Overview

QuadVerdict is engineered as a zero-leakage, publication-grade benchmark and web reporting engine. The architecture consists of four distinct operational layers:

```
+-------------------------------------------------------------------------------+
|                             1. Data Ingestion Layer                           |
|  - UCI Dataset Ingestion & Type Enforcement (src/data.py)                    |
|  - Stratified 80/20 Train/Hold-out Split (Dev: 24,000 | Hold-out: 6,000)      |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
|                         2. Benchmark & Execution Layer                        |
|  - Pipeline Construction & Scaling Isolation (src/pipelines.py)               |
|  - StratifiedSubsampleClassifier for SVM Runtime Capping                     |
|  - Nested Cross-Validation (5x3 = 15 Outer Folds, 3 Inner Folds)             |
|  - Incremental Checkpoint & Resume Engine (results/_cache/)                  |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
|                        3. Analytics & Export Engine                           |
|  - Cost-Sensitive Loss Metrics & Threshold Sweeps (src/metrics.py)           |
|  - Nadeau-Bengio & Wilcoxon Significance Correction (src/significance.py)    |
|  - Permutation Importance & Imbalance Studies (src/explain.py, src/exp.py)   |
|  - Arc-Length Downsampling & JSON Schema Validation (src/export.py)           |
+---------------------------------------+---------------------------------------+
                                        |
+---------------------------------------v---------------------------------------+
|                    4. Client Presentation & Web Engine                        |
|  - Atomic Compiler: results.json -> web/index.html -> dist/index.html         |
|  - Zero-Dependency Client App with Pure Vanilla HTML5/CSS3/Inline SVG        |
|  - Interactive Cost Optimization Lab & Dynamic Presets                        |
|  - Edge-Ready Vercel Configuration with A+ Security Headers (vercel.json)    |
+-------------------------------------------------------------------------------+
```

---

## 2. Data Flow and Leakage Safeguards

A foundational imperative of QuadVerdict is **mathematical and operational proof of zero data leakage**.

### 2.1 The Two-Tier Isolation Boundary

1. **Development vs. Hold-Out Split:**
   - Raw records ($N = 30,000$) are partitioned once at startup (`src/data.py:split_data`) into:
     - **Development Set ($N = 24,000$, 80%)**: The *only* data accessible to pipeline fitting, hyperparameter optimization, and cross-validation.
     - **Hold-Out Set ($N = 6,000$, 20%)**: Stored independently and strictly sealed. No estimator, scaler, or hyperparameter search inspects or fits on this split until post-hoc evaluation.
2. **Outer Fold Isolation:**
   - Inside the $5 \times 3$ Repeated Stratified Cross-Validation loop, each outer fold splits the 24,000 development rows into:
     - Outer Training: 19,200 rows ($80\%$)
     - Outer Test: 4,800 rows ($20\%$)
   - Inner hyperparameter searches (`RandomizedSearchCV`, 3 folds) execute solely on the 19,200 outer training rows.
   - Outer test fold indices are verified to have **null intersection** with outer training rows.

### 2.2 Pipeline Scaling Architecture

Feature normalization (scaling) before cross-validation is a prevalent vector of subtle data leakage. QuadVerdict adheres to structural scaling rules:

```
                    +------------------------+
                    | Raw Categorical/Numeric |
                    +-----------+------------+
                                |
        +-----------------------+-----------------------+
        |                                               |
  Tree-Based Pipelines                            Linear / Kernel Pipelines
  [Random Forest, Decision Tree]                  [Logistic Regression, SVM]
        |                                               |
  NO StandardScaler                               StandardScaler fitted ONLY
  (Preserves step functions)                      on fold training indices
        |                                               |
        +-----------------------+-----------------------+
                                |
                    Estimator Fit & Predict
```

- **Trees (`dt`, `rf`):** Fitted on raw values. No `StandardScaler` is present in their pipeline definition.
- **Linear / Kernel (`lr`, `svm`):** `StandardScaler` is encapsulated *inside* the scikit-learn pipeline step. Statistics ($\mu, \sigma$) are computed exclusively on training folds; test folds are transformed using the frozen training parameters.

---

## 3. Specialized Pipeline Components

### 3.1 StratifiedSubsampleClassifier for SVM Runtime Containment

Kernel Support Vector Machines with RBF kernels exhibit computational complexity between $\mathcal{O}(n^2)$ and $\mathcal{O}(n^3)$. On $N = 19,200$ rows with Platt scaling and hyperparameter tuning, standard SVM execution exceeds practical computational budgets.

To solve this rigorously without data leakage, QuadVerdict introduces `StratifiedSubsampleClassifier` (`src/pipelines.py`):

```python
class StratifiedSubsampleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, max_train_samples=8_000, random_state=42):
        self.estimator = estimator
        self.max_train_samples = max_train_samples
        self.random_state = random_state
```

- **Mechanism:** Implemented as a valid scikit-learn `BaseEstimator` and `ClassifierMixin`. When `fit(X, y)` is invoked:
  - If $N_{\text{train}} > 8,000$, it executes a stratified subsample down to exactly 8,000 samples using the pipeline fold seed.
  - If $N_{\text{train}} \le 8,000$, it fits on all available samples.
  - Test/evaluation calls (`predict`, `predict_proba`) execute on $100\%$ of test samples without subsampling.
- **Cloneability:** Fully compliant with `sklearn.base.clone()`, supporting seamless nested hyperparameter tuning inside `RandomizedSearchCV`.

### 3.2 SMOTE Resampling Isolation

When evaluating synthetic oversampling (`imblearn.over_sampling.SMOTE` in Panel 6), standard scripts often oversample the dataset before splitting, which artificially leaks synthetic copies of test distributions into training sets.

In QuadVerdict:
- SMOTE is injected exclusively via `imblearn.pipeline.Pipeline`.
- Synthetic interpolation operates **only** on the training folds during `fit_resample`.
- Outer and inner test sets remain $100\%$ authentic, empirical borrower records.

---

## 4. Checkpoint & Resume Architecture

The nested cross-validation run across 4 models, 15 folds, and inner search spaces requires substantial compute (Random Forest fits hundreds of trees; SVM fits thousands of support vectors).

QuadVerdict implements an **atomic checkpointing engine** (`src/nested_cv.py`):

1. **Storage:** Cached in `results/_cache/{model}_fold_{fold_idx}.joblib`.
2. **Determinism:** Each fold result stores outer test indices, true labels $y_{\text{true}}$, probability predictions $\hat{y}_{\text{proba}}$, best inner hyperparameters, and timing telemetry.
3. **Resumption:** Launching `python run_benchmark.py --resume` checks `results/_cache/`. Existing completed folds are loaded in memory in $< 0.1$ seconds, allowing interrupted runs to recover instantaneously without recomputing completed folds.

---

## 5. Web Compilation Engine

QuadVerdict avoids heavy node/webpack dependencies by utilizing an atomic Python site compiler (`web/build_site.py`):

```
+---------------------+     +-----------------------+
|   results/results.json  |     |   web/index.html      |
|  (Canonical Export) |     | (Structural Template) |
+----------+----------+     +-----------+-----------+
           |                            |
           +--------------+-------------+
                          |
             [ web/build_site.py ]
             - Schema Validation (results/schema.json)
             - SHA-256 Hash Computation
             - '</' Script Escape Guard
             - Atomic Token Substitution
                          |
                          v
               +----------------------+
               |   dist/index.html    |
               | (Self-Contained App) |
               +----------------------+
```

- **Output Invariants:**
  - File size must remain strictly $< 1.0\text{ MB}$ (current build: $\approx 167\text{ KB}$).
  - Standalone portability: Can be served over HTTP, CDN, or opened directly via `file://` protocol.
  - Zero external JavaScript libraries: No React, Vue, D3, or CDN bundles. Visualizations are rendered as raw, highly-optimized inline SVG elements driven by pure JavaScript.
