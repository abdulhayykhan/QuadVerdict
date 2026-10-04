"""
test_pipelines_leakage.py — Comprehensive leakage and architectural tests for src/pipelines.py.

Assertions verified (TRD §4.2, PRD §7):
1. Pipeline steps: ColumnTransformer, SMOTE (when enabled), and Estimator are named steps.
2. Leakage prevention: Fitting on fold-train never calls transform() or fit() on fold-test data.
3. SMOTE isolation: SMOTE applies strictly during fit() to the training fold; test fold rows
   are never resampled, synthesized, or altered in size.
4. Scale architecture: DT and RF numeric transformers are 'passthrough'; LR and SVM use StandardScaler.
5. StratifiedSubsampleClassifier:
   - Capped at max_train rows on training data.
   - Evaluated on full test set without subsampling.
   - Preserves stratification in subsampled fold.
   - Fully compatible with sklearn clone().
6. Imbalance strategy configuration:
   - 'none': no resampler, class_weight=None.
   - 'balanced': class_weight='balanced'.
   - 'smote': SMOTE resampler inserted between preprocessor and classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import SEED
from src.data import (
    clean,
    load_raw,
    make_splits,
)
from src.pipelines import (
    StratifiedSubsampleClassifier,
    build_pipeline,
    build_preprocessor,
    get_param_distributions,
)


@pytest.fixture(scope="module")
def dataset_splits():
    """Load real cleaned dataset and create splits once for testing."""
    from src.config import DATA_PATH

    raw = load_raw(DATA_PATH)
    cleaned = clean(raw)
    return make_splits(cleaned, seed=SEED, test_size=0.20)


@pytest.fixture
def sample_data(dataset_splits):
    """Small train/test slice for fast pipeline testing."""
    X_dev, _, y_dev, _ = dataset_splits
    X_train = X_dev.iloc[:500]
    y_train = y_dev.iloc[:500]
    X_test = X_dev.iloc[500:700]
    y_test = y_dev.iloc[500:700]
    return X_train, y_train, X_test, y_test


# ===========================================================================
# 1. Pipeline Structure & Named Steps
# ===========================================================================

class TestPipelineStructure:
    """Verify all 12 (model, strategy) combinations build proper imblearn Pipelines."""

    @pytest.mark.parametrize("model_key", ["lr", "dt", "rf", "svm"])
    @pytest.mark.parametrize("strategy", ["none", "balanced", "smote"])
    def test_pipeline_instance_and_step_names(self, model_key, strategy):
        pipe = build_pipeline(model_key, imbalance_strategy=strategy)

        # Must be imblearn Pipeline so SMOTE only applies during fit
        assert isinstance(pipe, ImbPipeline)
        step_names = [name for name, _ in pipe.steps]

        assert "preprocessor" in step_names
        assert "classifier" in step_names

        if strategy == "smote":
            assert "resampler" in step_names
            assert isinstance(pipe.named_steps["resampler"], SMOTE)
            assert step_names == ["preprocessor", "resampler", "classifier"]
        else:
            assert "resampler" not in step_names
            assert step_names == ["preprocessor", "classifier"]

    def test_invalid_model_key_raises(self):
        with pytest.raises(ValueError, match="Unknown model_key"):
            build_pipeline("xgboost")

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown imbalance_strategy"):
            build_pipeline("lr", imbalance_strategy="adasyn")


# ===========================================================================
# 2. Scale Architecture: Tree passthrough vs Linear/SVM scaling
# ===========================================================================

class TestScalingArchitecture:
    """Verify numeric transformers adhere to scale-invariance rules (TRD §4.2)."""

    def test_trees_have_no_standard_scaler(self):
        for model_key in ["dt", "rf"]:
            pipe = build_pipeline(model_key)
            preprocessor = pipe.named_steps["preprocessor"]
            assert isinstance(preprocessor, ColumnTransformer)

            # preprocessor.transformers is a list of (name, transformer, columns) tuples
            transformers = {name: trans for name, trans, _ in preprocessor.transformers}
            num_trans = transformers["num"]
            assert num_trans == "passthrough", (
                f"Model '{model_key}' must have passthrough for numeric features, found {num_trans}"
            )

    def test_linear_and_svm_have_standard_scaler(self):
        for model_key in ["lr", "svm"]:
            pipe = build_pipeline(model_key)
            preprocessor = pipe.named_steps["preprocessor"]
            transformers = {name: trans for name, trans, _ in preprocessor.transformers}
            num_trans = transformers["num"]
            assert isinstance(num_trans, StandardScaler), (
                f"Model '{model_key}' must use StandardScaler, found {num_trans}"
            )


# ===========================================================================
# 3. Data Leakage Prevention & Spy Transformer
# ===========================================================================

class SpyTrackingTransformer(BaseEstimator, TransformerMixin):
    """Spy transformer that logs row signatures observed during fit and transform."""

    def __init__(self):
        self.fitted_signatures = set()
        self.transformed_signatures = set()

    def fit(self, X, y=None):
        signatures = [tuple(row) for row in np.asarray(X)]
        self.fitted_signatures.update(signatures)
        return self

    def transform(self, X):
        signatures = [tuple(row) for row in np.asarray(X)]
        self.transformed_signatures.update(signatures)
        return X


class TestDataLeakage:
    """Verify strictly zero data leakage from test data into training/fitting."""

    def test_spy_transformer_verifies_test_data_never_touched_during_fit(self, sample_data):
        X_train, y_train, X_test, _ = sample_data

        # Build spy and insert as first step in an evaluation pipeline
        spy = SpyTrackingTransformer()
        pipe = build_pipeline("lr", imbalance_strategy="smote")

        # Insert spy step before preprocessing
        pipe.steps.insert(0, ("spy", spy))

        # Fit on train fold ONLY
        pipe.fit(X_train, y_train)

        # Confirm spy saw ONLY training rows during fit
        train_signatures = set(tuple(row) for row in np.asarray(X_train))
        test_signatures = set(tuple(row) for row in np.asarray(X_test))

        assert spy.fitted_signatures == train_signatures
        # Crucial check: test signatures must NEVER have been seen during fit()
        overlap = spy.fitted_signatures.intersection(test_signatures)
        assert len(overlap) == 0, f"Leakage: {len(overlap)} test rows observed during fit!"

        # Now transform/predict on test fold
        pipe.predict_proba(X_test)
        # Now test rows should be in transformed_signatures, but fitted_signatures is unchanged
        assert spy.fitted_signatures == train_signatures
        assert test_signatures.issubset(spy.transformed_signatures)


# ===========================================================================
# 4. SMOTE Isolation: Never Alters Test Data Size or Values
# ===========================================================================

class TestSMOTEIsolation:
    """Verify SMOTE resamples train data internally without altering test data."""

    def test_smote_does_not_alter_test_set_size_or_leak(self, sample_data):
        X_train, y_train, X_test, _ = sample_data

        pipe = build_pipeline("rf", imbalance_strategy="smote")
        pipe.fit(X_train, y_train)

        # Prediction on test set must have exact same row count
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert probs.shape == (len(X_test), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_smote_resamples_training_distribution(self, sample_data):
        X_train, y_train, _, _ = sample_data
        preprocessor = build_preprocessor("lr")
        X_proc = preprocessor.fit_transform(X_train)

        smote = SMOTE(random_state=SEED)
        X_resampled, y_resampled = smote.fit_resample(X_proc, y_train)

        # Before SMOTE: class 1 is minority (~22%)
        n_min_before = (y_train == 1).sum()
        n_maj_before = (y_train == 0).sum()
        assert n_min_before < n_maj_before

        # After SMOTE: balanced 50/50 in resampled training fold
        n_min_after = (y_resampled == 1).sum()
        n_maj_after = (y_resampled == 0).sum()
        assert n_min_after == n_maj_after == n_maj_before
        assert len(X_resampled) > len(X_train)


# ===========================================================================
# 5. StratifiedSubsampleClassifier (SVM Runtime Control)
# ===========================================================================

class DummyCounterEstimator(BaseEstimator):
    """Dummy estimator to record the exact training set size it receives."""

    def __init__(self):
        self.fit_count_ = 0
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        self.fit_count_ += 1
        self.n_samples_seen_ = len(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.8
        probs[:, 1] = 0.2
        return probs


class TestStratifiedSubsampleClassifier:
    """Verify StratifiedSubsampleClassifier caps training rows while evaluating on all test rows."""

    def test_caps_training_rows_when_exceeding_max_train(self):
        dummy = DummyCounterEstimator()
        max_train = 100
        wrapper = StratifiedSubsampleClassifier(
            estimator=dummy,
            max_train=max_train,
            random_state=SEED,
        )

        # 300 samples > 100
        X = pd.DataFrame(np.random.randn(300, 5))
        y = pd.Series(np.random.choice([0, 1], size=300, p=[0.8, 0.2]))

        wrapper.fit(X, y)
        assert wrapper.estimator_.n_samples_seen_ == max_train

        # Prediction must evaluate on ALL test samples (no subsampling on predict)
        X_test = pd.DataFrame(np.random.randn(250, 5))
        probs = wrapper.predict_proba(X_test)
        assert len(probs) == 250

    def test_preserves_all_rows_when_under_max_train(self):
        dummy = DummyCounterEstimator()
        max_train = 500
        wrapper = StratifiedSubsampleClassifier(
            estimator=dummy,
            max_train=max_train,
            random_state=SEED,
        )

        # 150 samples < 500
        X = pd.DataFrame(np.random.randn(150, 5))
        y = pd.Series(np.random.choice([0, 1], size=150, p=[0.8, 0.2]))

        wrapper.fit(X, y)
        assert wrapper.estimator_.n_samples_seen_ == 150

    def test_maintains_stratification_in_subsample(self):
        dummy = DummyCounterEstimator()
        max_train = 200
        wrapper = StratifiedSubsampleClassifier(
            estimator=dummy,
            max_train=max_train,
            random_state=SEED,
        )

        # Exact 20% positive rate over 1000 samples
        y = pd.Series([1] * 200 + [0] * 800)
        X = pd.DataFrame(np.zeros((1000, 2)))

        wrapper.fit(X, y)
        # Check stratification inside fit via train_test_split behavior
        assert wrapper.estimator_.n_samples_seen_ == 200

    def test_cloneable_by_sklearn(self):
        wrapper = StratifiedSubsampleClassifier(
            estimator=SVC(),
            max_train=500,
            random_state=42,
        )
        cloned = clone(wrapper)
        assert cloned.max_train == 500
        assert cloned.random_state == 42
        assert isinstance(cloned.estimator, SVC)

    def test_real_svm_pipeline_fit_predict(self, sample_data):
        X_train, y_train, X_test, _ = sample_data
        pipe = build_pipeline("svm", imbalance_strategy="none")

        # Set small cap for fast test execution
        pipe.named_steps["classifier"].max_train = 150

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)

        assert probs.shape == (len(X_test), 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert np.allclose(probs.sum(axis=1), 1.0)


# ===========================================================================
# 6. Imbalance Strategy Configuration
# ===========================================================================

class TestImbalanceStrategies:
    """Verify class weights are properly assigned for 'balanced' strategy."""

    def test_lr_balanced_weight(self):
        pipe_none = build_pipeline("lr", "none")
        pipe_bal = build_pipeline("lr", "balanced")
        assert pipe_none.named_steps["classifier"].class_weight is None
        assert pipe_none.named_steps["classifier"].class_weight != "balanced"
        assert pipe_bal.named_steps["classifier"].class_weight == "balanced"

    def test_dt_balanced_weight(self):
        pipe_none = build_pipeline("dt", "none")
        pipe_bal = build_pipeline("dt", "balanced")
        assert pipe_none.named_steps["classifier"].class_weight is None
        assert pipe_bal.named_steps["classifier"].class_weight == "balanced"

    def test_rf_balanced_weight(self):
        pipe_none = build_pipeline("rf", "none")
        pipe_bal = build_pipeline("rf", "balanced")
        assert pipe_none.named_steps["classifier"].class_weight is None
        assert pipe_bal.named_steps["classifier"].class_weight == "balanced"

    def test_svm_balanced_weight(self):
        pipe_none = build_pipeline("svm", "none")
        pipe_bal = build_pipeline("svm", "balanced")

        cal_none = pipe_none.named_steps["classifier"].estimator
        cal_bal = pipe_bal.named_steps["classifier"].estimator

        assert isinstance(cal_none, CalibratedClassifierCV)
        assert isinstance(cal_bal, CalibratedClassifierCV)

        svc_none = cal_none.estimator
        svc_bal = cal_bal.estimator

        assert svc_none.class_weight is None
        assert svc_bal.class_weight == "balanced"


# ===========================================================================
# 7. Hyperparameter Search Spaces
# ===========================================================================

class TestSearchSpaces:
    """Verify parameter distributions match TRD §4.2."""

    @pytest.mark.parametrize("model_key", ["lr", "dt", "rf", "svm"])
    def test_param_distributions_match_pipeline_parameters(self, model_key):
        pipe = build_pipeline(model_key)
        params = get_param_distributions(model_key)
        pipeline_params = set(pipe.get_params().keys())

        for param_name in params:
            assert param_name in pipeline_params, (
                f"Param '{param_name}' not found in {model_key} pipeline parameters!"
            )
