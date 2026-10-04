"""
test_pipelines_leakage.py — Leakage tests for src/pipelines.py

Assertions (Phase 2, to be filled in):
1. Pipeline steps contain scaler/SMOTE as named steps (not bare transformers).
2. Spy transformer: fitting the pipeline on fold-train never calls transform() on fold-test
   rows during fit().
3. SMOTE only changes the training matrix — test count is unchanged.
4. DT and RF pipelines contain no StandardScaler step.
"""
# Tests will be implemented in Phase 2.
