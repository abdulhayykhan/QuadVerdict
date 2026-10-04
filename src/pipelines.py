"""
pipelines.py — Pipeline factories and estimator wrappers for QuadVerdict.

Design rules (TRD §4.2, PRD §7):
1. Preprocessing (ColumnTransformer) and resamplers (SMOTE) MUST live inside
   Pipeline objects — NEVER fit outside a pipeline to prevent data leakage.
2. All pipelines use imblearn.pipeline.Pipeline so SMOTE only applies to
   training folds during fit() and never touches validation/test folds.
3. Estimators and splits strictly derive randomness from config.SEED.
4. StratifiedSubsampleClassifier caps SVM training data to at most SVM_MAX_TRAIN
   rows inside each training fold to keep kernel SVC runtime tractable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED, SVM_MAX_TRAIN
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class StratifiedSubsampleClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper that takes a stratified subsample of training data.

    Enforces runtime bounds for quadratic/cubic algorithms like kernel SVC
    (TRD §4.2) by subsampling at most `max_train` rows during `fit()`.
    Evaluation on test folds uses the full test set without subsampling.

    Parameters
    ----------
    estimator : estimator object, default=None
        The base classifier to fit on the subsampled training data.
    max_train : int, default=SVM_MAX_TRAIN (8000)
        Maximum number of training rows to retain.
    random_state : int, default=SEED (42)
        Seed for deterministic stratified subsampling.
    """

    def __init__(
        self,
        estimator: Any = None,
        max_train: int = SVM_MAX_TRAIN,
        random_state: int = SEED,
    ) -> None:
        self.estimator = estimator
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, X: Any, y: Any, **fit_params: Any) -> StratifiedSubsampleClassifier:
        """Fit the wrapped estimator on a stratified subsample of (X, y).

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like or pd.Series of shape (n_samples,)
            Target labels.
        **fit_params : dict
            Parameters forwarded to the base estimator's fit method.

        Returns
        -------
        self : StratifiedSubsampleClassifier
            Fitted estimator.
        """
        if self.estimator is None:
            raise ValueError("Base estimator must be specified.")

        n_samples = len(X)
        self.estimator_ = clone(self.estimator)

        if n_samples > self.max_train:
            # Deterministic stratified subsampling
            indices = np.arange(n_samples)
            sub_idx, _ = train_test_split(
                indices,
                train_size=self.max_train,
                stratify=y,
                random_state=self.random_state,
            )
            # Support both pandas and numpy indexing
            if hasattr(X, "iloc"):
                X_train = X.iloc[sub_idx]
            else:
                X_train = X[sub_idx]

            if hasattr(y, "iloc"):
                y_train = y.iloc[sub_idx]
            else:
                y_train = y[sub_idx]
        else:
            X_train = X
            y_train = y

        self.estimator_.fit(X_train, y_train, **fit_params)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict classes for X using the fitted base estimator."""
        if not hasattr(self, "estimator_"):
            raise ValueError("This StratifiedSubsampleClassifier instance is not fitted yet.")
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for X using the fitted base estimator."""
        if not hasattr(self, "estimator_"):
            raise ValueError("This StratifiedSubsampleClassifier instance is not fitted yet.")
        return self.estimator_.predict_proba(X)

    def decision_function(self, X: Any) -> np.ndarray:
        """Decision function for X if supported by the base estimator."""
        if not hasattr(self, "estimator_"):
            raise ValueError("This StratifiedSubsampleClassifier instance is not fitted yet.")
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        raise AttributeError(f"{type(self.estimator_).__name__} has no decision_function.")


def build_preprocessor(model_key: str) -> ColumnTransformer:
    """Construct the ColumnTransformer preprocessing step.

    Categorical features (SEX, EDUCATION, MARRIAGE):
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    Numeric features (LIMIT_BAL, AGE, PAY_1..6, BILL_AMT1..6, PAY_AMT1..6):
        StandardScaler() for 'lr' and 'svm'
        'passthrough' for 'dt' and 'rf' (trees are scale-invariant)

    Parameters
    ----------
    model_key : {'lr', 'dt', 'rf', 'svm'}

    Returns
    -------
    ColumnTransformer
    """
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if model_key in ("lr", "svm"):
        num_transformer = StandardScaler()
    elif model_key in ("dt", "rf"):
        num_transformer = "passthrough"
    else:
        raise ValueError(f"Unknown model_key: '{model_key}'. Expected one of: lr, dt, rf, svm.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ("num", num_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(
    model_key: str,
    imbalance_strategy: str = "none",
) -> Pipeline:
    """Build an imblearn Pipeline for the given model and imbalance strategy.

    Parameters
    ----------
    model_key : {'lr', 'dt', 'rf', 'svm'}
        Model identifier.
    imbalance_strategy : {'none', 'balanced', 'smote'}, default='none'
        - 'none': Default estimator without class weight adjustment or resampling.
        - 'balanced': Set class_weight='balanced' on the estimator.
        - 'smote': Insert SMOTE(random_state=SEED) before the estimator.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline containing:
        - ('preprocessor', ColumnTransformer)
        - ('resampler', SMOTE) [only if imbalance_strategy == 'smote']
        - ('classifier', Estimator)
    """
    valid_models = ("lr", "dt", "rf", "svm")
    if model_key not in valid_models:
        raise ValueError(f"Unknown model_key: '{model_key}'. Expected one of: {valid_models}")

    valid_strategies = ("none", "balanced", "smote")
    if imbalance_strategy not in valid_strategies:
        raise ValueError(
            f"Unknown imbalance_strategy: '{imbalance_strategy}'. Expected one of: {valid_strategies}"
        )

    # 1. Preprocessor
    preprocessor = build_preprocessor(model_key)
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]

    # 2. Resampler (SMOTE) if strategy == 'smote'
    if imbalance_strategy == "smote":
        steps.append(("resampler", SMOTE(random_state=SEED)))

    # 3. Classifier with class_weight if strategy == 'balanced'
    class_weight = "balanced" if imbalance_strategy == "balanced" else None

    if model_key == "lr":
        classifier = LogisticRegression(
            solver="saga",
            max_iter=2000,
            class_weight=class_weight,
            random_state=SEED,
        )
    elif model_key == "dt":
        classifier = DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=SEED,
        )
    elif model_key == "rf":
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=SEED,
        )
    elif model_key == "svm":
        # Calibrated RBF SVC with sigmoid calibration over 3 inner folds
        base_svc = SVC(
            kernel="rbf",
            class_weight=class_weight,
            random_state=SEED,
        )
        calibrated_svc = CalibratedClassifierCV(
            estimator=base_svc,
            method="sigmoid",
            cv=3,
        )
        classifier = StratifiedSubsampleClassifier(
            estimator=calibrated_svc,
            max_train=SVM_MAX_TRAIN,
            random_state=SEED,
        )

    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps=steps)
    return pipeline


def get_param_distributions(model_key: str) -> dict[str, Any]:
    """Return the hyperparameter search space for RandomizedSearchCV (TRD §4.2).

    Parameters
    ----------
    model_key : {'lr', 'dt', 'rf', 'svm'}

    Returns
    -------
    dict[str, Any]
        Mapping of pipeline parameter names to candidate values or distributions.
    """
    if model_key == "lr":
        # TRD §4.2: C in loguniform(1e-3, 1e2); penalty in {l1, l2}
        return {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__penalty": ["l1", "l2"],
        }
    elif model_key == "dt":
        # TRD §4.2: max_depth in {3..20, None}; min_samples_leaf in {1..50}; ccp_alpha
        return {
            "classifier__max_depth": [3, 5, 8, 12, 15, 20, None],
            "classifier__min_samples_leaf": list(range(1, 51)),
            "classifier__ccp_alpha": loguniform(1e-6, 1e-2),
        }
    elif model_key == "rf":
        # TRD §4.2: n_estimators {100..500}; max_depth {5..30, None}; max_features; min_samples_leaf
        return {
            "classifier__n_estimators": list(range(100, 501, 50)),
            "classifier__max_depth": [5, 10, 15, 20, 25, 30, None],
            "classifier__max_features": ["sqrt", "log2", 0.3],
            "classifier__min_samples_leaf": list(range(1, 21)),
        }
    elif model_key == "svm":
        # TRD §4.2: C in loguniform(1e-2, 1e2); gamma in loguniform(1e-4, 1e0)
        # Nested through: StratifiedSubsampleClassifier -> CalibratedClassifierCV -> SVC
        return {
            "classifier__estimator__estimator__C": loguniform(1e-2, 1e2),
            "classifier__estimator__estimator__gamma": loguniform(1e-4, 1e0),
        }
    else:
        raise ValueError(f"Unknown model_key: '{model_key}'. Expected one of: lr, dt, rf, svm.")
