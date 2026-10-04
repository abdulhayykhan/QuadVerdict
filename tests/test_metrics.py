"""
test_metrics.py — Unit tests for src/metrics.py

Assertions (Phase 4, to be filled in):
- Threshold table against hand-computed tiny arrays (known proba + labels → known TP/FP/TN/FN).
- MCC edge case: zero denominator → 0.0 (no exception).
- Precision edge case: tp+fp=0 → 1.0.
- F1 edge case: 2*tp+fp+fn=0 → 0.0.
- Invariant: tp + fn == n_pos (within 1e-6) for every row.
- Invariant: fp + tn == n_neg (within 1e-6) for every row.
- tp and fp are non-increasing in t.
"""
# Tests will be implemented in Phase 4.
