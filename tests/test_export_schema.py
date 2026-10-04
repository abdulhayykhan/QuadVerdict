"""
test_export_schema.py — Schema validation and mutation tests for src/export.py

Assertions (Phase 6, to be filled in — mutation tests: each broken input must be caught):
- Valid quick-run output passes validate() without error.
- Missing model key → ValidationError.
- Threshold table with != 101 rows → ValidationError.
- t not strictly increasing → ValidationError.
- tp + fn != n_pos (beyond 1e-6 tolerance) → ValidationError.
- fp + tn != n_neg (beyond 1e-6 tolerance) → ValidationError.
- tp not non-increasing in t → ValidationError.
- cv_scores list length != 15 → ValidationError.
- AUC value outside [0, 1] → ValidationError.
- MCC value outside [-1, 1] → ValidationError.
- ROC/PR with > 200 points → ValidationError.
- significance with != 6 rows → ValidationError.
- File size > RESULTS_MAX_KB → warning/error.
"""
# Tests will be implemented in Phase 6.
