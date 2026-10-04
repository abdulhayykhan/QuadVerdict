"""
test_significance.py — Unit tests for src/significance.py

Assertions (Phase 4, to be filled in):
- Nadeau-Bengio t and p match hand-computed example (within 1e-6).
- Identical score vectors (d_i = 0 for all i) → p = 1.0 (no zero-variance crash).
- Wilcoxon on known differences → stat and p match scipy reference.
- Holm correction on a known p-value vector → correct adjusted values.
- Exactly 6 pairwise rows returned for 4 models.
"""
# Tests will be implemented in Phase 4.
