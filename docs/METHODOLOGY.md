# QuadVerdict Scientific Methodology & Statistical Theory

This document provides the mathematical foundations, statistical corrections, decision-theoretic loss formulations, and algorithmic specifications governing **QuadVerdict**.

---

## 1. The Core Economic Thesis: Asymmetric Loss in Credit Lending

In conventional machine learning benchmarks, classification models are evaluated using symmetric accuracy or standard ROC-AUC curves evaluated at an arbitrary decision threshold of $t = 0.50$. In commercial retail lending, this standard practice is fundamentally decoupled from financial reality.

### 1.1 The Asymmetric Loss Equation

Consider an institution extending a revolving credit line (such as credit cards in the UCI Taiwan dataset):

- **False Negative ($\text{FN}$ - Missed Default):** The classifier predicts that a borrower will repay ($y=0$), but the borrower defaults ($y=1$). The lender loses the outstanding principal, charge-off loss, and legal recovery expenses ($C_{\text{FN}}$).
- **False Positive ($\text{FP}$ - False Alert / Rejected Good Borrower):** The classifier predicts default ($y=1$), but the borrower would have repaid in full ($y=0$). The lender incurs customer friction, underwriting verification cost, and forfeited interchange/interest margin ($C_{\text{FP}}$).

Empirically, principal loss far exceeds operational underwriting cost:

$$\frac{C_{\text{FN}}}{C_{\text{FP}}} \in [2, 50]$$

In QuadVerdict, the default benchmark operating ratio is established at **$5 : 1$** ($C_{\text{FN}} = 5.0, C_{\text{FP}} = 1.0$), with neutral parameters $C_{\text{TP}} = 0, C_{\text{TN}} = 0$.

### 1.2 Total Expected Financial Cost

For any candidate decision threshold $t \in [0, 1]$ applied to predicted probabilities $\hat{p}_i = P(y_i = 1 \mid \mathbf{x}_i)$:

$$\hat{y}_i(t) = \mathbb{I}(\hat{p}_i \ge t)$$

The empirical loss on a test set of size $N$ is:

$$\text{Cost}(t) = C_{\text{FP}} \cdot \text{FP}(t) + C_{\text{FN}} \cdot \text{FN}(t)$$

$$\text{Cost}(t) = 1.0 \times \sum_{i: y_i = 0} \mathbb{I}(\hat{p}_i \ge t) + 5.0 \times \sum_{i: y_i = 1} \mathbb{I}(\hat{p}_i < t)$$

The optimal threshold $t^*$ is the global minimizer over the discretized threshold sweep $t \in \{0.00, 0.01, \dots, 1.00\}$:

$$t^* = \arg\min_{t \in [0, 1]} \text{Cost}(t)$$

---

## 2. Statistical Significance Testing & Resample Corrections

Standard paired Student's $t$-tests assume independent, identically distributed ($i.i.d.$) samples. In cross-validation (especially repeated $k$-fold cross-validation), training partitions overlap heavily across folds. Standard $t$-tests underestimate the true sampling variance of the differences, producing inflated test statistics and false-positive significance rates exceeding $40\%$.

QuadVerdict addresses this through a dual-test framework:

### 2.1 The Nadeau-Bengio Corrected Resampled t-Test

Let $J$ be the total number of outer folds ($J = K \times R = 5 \text{ splits} \times 3 \text{ repeats} = 15 \text{ folds}$).
For each fold $j \in \{1, \dots, J\}$, let $d_j = M_{A, j} - M_{B, j}$ represent the performance difference (in PR-AUC) between Model $A$ and Model $B$.

The sample mean difference is:

$$\bar{d} = \frac{1}{J} \sum_{j=1}^J d_j$$

The standard sample variance across the $J$ folds is:

$$S_d^2 = \frac{1}{J - 1} \sum_{j=1}^J (d_j - \bar{d})^2$$

To account for the covariance introduced by overlapping training sets, **Nadeau and Bengio (2003)** derived the corrected variance estimator:

$$\sigma^2_{\text{corr}} = S_d^2 \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right)$$

In QuadVerdict:
- $J = 15$
- $n_{\text{test}} = 4,800$
- $n_{\text{train}} = 19,200$
- $\frac{n_{\text{test}}}{n_{\text{train}}} = \frac{4,800}{19,200} = 0.25$

The correction multiplier is:

$$\frac{1}{15} + 0.25 = 0.0667 + 0.25 = 0.3167$$

The Nadeau-Bengio test statistic is:

$$t_{\text{NB}} = \frac{\bar{d}}{\sqrt{S_d^2 \left( \frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \right)}} \sim t_{J - 1}$$

With degrees of freedom $\nu = J - 1 = 14$. The two-tailed $p$-value is:

$$p = 2 \cdot \left( 1 - F_{t_{14}}(|t_{\text{NB}}|) \right)$$

### 2.2 Non-Parametric Wilcoxon Signed-Rank Test

To safeguard against violations of normality in cross-validation differences, QuadVerdict simultaneously runs the non-parametric Wilcoxon signed-rank test on the paired vector of differences $(d_1, \dots, d_{15})$.

### 2.3 Holm-Bonferroni Family-Wise Error Control

Across the 4 model paradigms, there are $\binom{4}{2} = 6$ pairwise comparisons (`rf vs dt`, `rf vs svm`, `rf vs lr`, `dt vs svm`, `dt vs lr`, `svm vs lr`). Testing 6 hypotheses at nominal $\alpha = 0.05$ creates substantial risk of Type I error inflation.

QuadVerdict controls the Family-Wise Error Rate (FWER) via the **Holm-Bonferroni step-down procedure**:

1. Sort the unadjusted $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$ where $m = 6$.
2. For rank $k = 1, \dots, m$, compute adjusted $p$-value:
   $$p_{(k)}^{\text{adj}} = \min\left(1.0, \max_{j \le k} \left( (m - j + 1) \cdot p_{(j)} \right)\right)$$
3. A pairwise difference is declared statistically significant if and only if:
   $$p_{(k)}^{\text{adj}} < 0.05$$

---

## 3. High-Fidelity Arc-Length Downsampling

Raw pooled cross-validation probability outputs contain thousands of unique coordinates for ROC and Precision-Recall curves. Embedding uncompressed arrays into a client-side bundle would bloat file size beyond 5 MB and cause browser rendering lag.

Uniform subsampling (e.g. taking every 50th point) severely degrades curve fidelity near sharp inflection points (such as high-precision thresholds near $(0, 1)$).

QuadVerdict utilizes **Arc-Length Preserving Downsampling** (`src/metrics.py:downsample_curve`):

1. The cumulative Euclidean arc length $S$ along the curve points $(x_i, y_i)_{i=1}^M$ is computed:
   $$s_0 = 0, \quad s_i = s_{i-1} + \sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2}$$
2. Target arc-length increments $\Delta s = \frac{s_M}{N_{\text{target}} - 1}$ are generated for $N_{\text{target}} \le 200$.
3. For each step, the point closest to the arc length target is selected.
4. The exact endpoints $(x_1, y_1)$ and $(x_M, y_M)$ are pinned unconditionally.

This reduces curve coordinate counts by **$96\%$** while preserving curvature and area under the curve to within $\pm 0.0001$.

---

## 4. Probability Calibration & Reliability Assessment

Classifiers optimize discriminative rankings, which does not guarantee that predicted scores $\hat{p}$ equal the true probability of default:

$$\mathbb{E}[y \mid \hat{p} = p] \stackrel{?}{=} p$$

### 4.1 Brier Score Metric

The mean squared error of probability forecasts measures both discrimination and calibration:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

### 4.2 Reliability Diagrams (10 Quantile Bins)

To evaluate calibration across diverse risk strata:
1. Predictions $\hat{p}_i$ on out-of-fold test sets are sorted and partitioned into 10 equally-sized quantile bins $B_1, \dots, B_{10}$.
2. For each bin $B_k$, the mean predicted probability $\bar{p}_k$ and observed empirical fraction of defaults $\bar{y}_k$ are computed:
   $$\bar{p}_k = \frac{1}{|B_k|} \sum_{i \in B_k} \hat{p}_i, \quad \bar{y}_k = \frac{1}{|B_k|} \sum_{i \in B_k} y_i$$
3. Perfect calibration is represented by the diagonal identity line $\bar{y}_k = \bar{p}_k$.
4. SVM incorporates 3-fold inner sigmoid Platt scaling ($P(y=1 \mid f) = \frac{1}{1 + \exp(A f + B)}$), yielding the benchmark's superior Brier score ($0.1401$).

---

## 5. Permutation Feature Importance

To understand model decision drivers without reliance on model-specific heuristics (e.g. tree Gini impurity which is biased toward high-cardinality features), QuadVerdict utilizes **Permutation Feature Importance** (`src/explain.py`):

1. For each outer fold $j \in \{1, \dots, 15\}$, evaluate baseline out-of-fold test metric $\text{PR-AUC}_j(\text{baseline})$.
2. For each feature column $c \in \{1, \dots, 23\}$:
   - Permute the values of column $c$ randomly across the outer test set, breaking the association between $x_c$ and target $y$ while preserving marginal feature distributions.
   - Re-evaluate performance $\text{PR-AUC}_{j, c}(\text{shuffled})$.
   - Compute feature importance delta:
     $$\Delta_{j, c} = \text{PR-AUC}_j(\text{baseline}) - \text{PR-AUC}_{j, c}(\text{shuffled})$$
3. Compute the mean importance $\bar{\Delta}_c$ and sample standard error across the 15 outer folds.
4. **Dominant Finding:** `PAY_1` (repayment status in recent month) accounts for $+0.2156 \pm 0.0084$ drop in PR-AUC, exceeding all remaining 22 financial features combined.
