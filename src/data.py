"""
data.py — Load, clean, and split the UCI Default of Credit Card Clients dataset.

Public API:
    load_raw(path)  → pd.DataFrame   (raw, unchanged except reading)
    clean(df)       → pd.DataFrame   (quirks fixed, target/columns renamed)
    make_splits(df, seed, test_size) → (X_dev, X_hold, y_dev, y_hold)

Design rules (TRD §4.1):
- NO transformation (scaling, encoding, SMOTE) happens here.
  Those live inside Pipeline objects in pipelines.py.
- The hold-out split is made once with a fixed seed and NEVER touched
  during nested CV or hyperparameter tuning.
- All cleaning logic is in clean(); notebooks import from here — they
  do NOT duplicate this logic.

Dataset quirks (documented in PRD §9 and TRD §4.1):
- Column named PAY_0 (not PAY_1); renamed to PAY_1 for consistency.
- EDUCATION: undocumented values {0, 5, 6} → 4 ("others").
- MARRIAGE: undocumented value 0 → 3 ("others").
- ID column: dropped (row identifier, not a feature).
- Target column: 'default.payment.next.month' → 'default'.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Feature groups used downstream by pipelines.py (single source of truth).
CATEGORICAL_FEATURES: list[str] = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES: list[str] = [
    "LIMIT_BAL",
    "AGE",
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET: str = "default"
ALL_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded. Returns the unmodified DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to UCI_Credit_Card.csv (or any compatible CSV).

    Returns
    -------
    pd.DataFrame
        Raw data with original column names, dtypes as float64/int64.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run `python data/download.py` to fetch it."
        )
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all documented cleaning steps to the raw DataFrame.

    Cleaning steps (in order):
    1. Drop the ID column (row identifier, not a feature).
    2. Rename PAY_0 → PAY_1 (the dataset skips PAY_1 in original naming).
    3. Rename target column to 'default'.
    4. Fix EDUCATION: map {0, 5, 6} → 4 ("others / undocumented").
    5. Fix MARRIAGE: map {0} → 3 ("others / undocumented").
    6. Cast all feature columns to int (they are stored as float64 in the CSV
       due to a NaN-safe read; no actual NaNs exist in this dataset).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: CATEGORICAL_FEATURES + NUMERIC_FEATURES + ['default'].
    """
    df = df.copy()

    # 1. Drop ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # 2. Rename PAY_0 → PAY_1
    if "PAY_0" in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})

    # 3. Rename target
    target_col = "default.payment.next.month"
    if target_col in df.columns:
        df = df.rename(columns={target_col: TARGET})

    # 4. Fix EDUCATION undocumented values {0, 5, 6} → 4 ("others")
    #    Documented categories: 1=graduate school, 2=university, 3=high school, 4=others.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype(int)

    # 5. Fix MARRIAGE undocumented value {0} → 3 ("others")
    #    Documented categories: 1=married, 2=single, 3=others.
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype(int)

    # 6. Cast remaining columns to int (safe: no NaNs in this dataset)
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = df[col].astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)

    # Reorder columns: features first, then target
    df = df[ALL_FEATURES + [TARGET]]

    return df


def make_splits(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/hold-out split.

    The 20 % hold-out set is NEVER used during nested CV or hyperparameter
    tuning. It is touched exactly once as a post-hoc sanity check (TRD §4.9).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean().
    seed : int
        Random seed — must be config.SEED for reproducibility.
    test_size : float
        Fraction of data for the hold-out set. Default 0.20.

    Returns
    -------
    X_dev, X_hold, y_dev, y_hold : DataFrames and Series
        - X_dev  / y_dev  : 80 % development set → fed to nested CV.
        - X_hold / y_hold : 20 % hold-out set → untouched until §4.9.
    """
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    return X_dev, X_hold, y_dev, y_hold


def load_and_prepare(
    path: str | Path,
    seed: int,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience wrapper: load_raw → clean → make_splits.

    Parameters
    ----------
    path : str or Path
        Path to UCI_Credit_Card.csv.
    seed : int
        Random seed — use config.SEED.
    test_size : float
        Hold-out fraction.

    Returns
    -------
    X_dev, X_hold, y_dev, y_hold
    """
    df_raw = load_raw(path)
    df = clean(df_raw)
    return make_splits(df, seed=seed, test_size=test_size)
