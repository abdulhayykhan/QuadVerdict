"""
test_data.py — Full test suite for src/data.py (Phase 1).

Tests:
- load_raw: file-not-found error, basic shape.
- clean: all 5 quirk-handling steps verified.
- make_splits: stratification, disjointness, reproducibility, proportions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean,
    load_and_prepare,
    load_raw,
    make_splits,
)

# ---------------------------------------------------------------------------
# Fixture: path to the real dataset (skipped if not present)
# ---------------------------------------------------------------------------
DATA_PATH = config.DATA_PATH


def _require_data() -> Path:
    """Return DATA_PATH or pytest.skip if it doesn't exist."""
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}. Run `python data/download.py`.")
    return DATA_PATH


# ---------------------------------------------------------------------------
# Fixture: small synthetic DataFrame that mimics the raw CSV structure
# ---------------------------------------------------------------------------
@pytest.fixture()
def raw_mini() -> pd.DataFrame:
    """10-row synthetic raw DataFrame with all the quirks present."""
    n = 10
    rng = np.random.default_rng(0)
    data = {
        "ID": range(1, n + 1),
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": [1, 2, 3, 4, 0, 5, 6, 1, 2, 3],   # 0,5,6 are quirks
        "MARRIAGE": [1, 2, 3, 0, 1, 2, 1, 2, 3, 0],     # 0 is a quirk
        "AGE": rng.integers(20, 70, n),
        "PAY_0": rng.integers(-2, 9, n),                  # should become PAY_1
        "PAY_2": rng.integers(-2, 9, n),
        "PAY_3": rng.integers(-2, 9, n),
        "PAY_4": rng.integers(-2, 9, n),
        "PAY_5": rng.integers(-2, 9, n),
        "PAY_6": rng.integers(-2, 9, n),
        "BILL_AMT1": rng.integers(0, 50_000, n),
        "BILL_AMT2": rng.integers(0, 50_000, n),
        "BILL_AMT3": rng.integers(0, 50_000, n),
        "BILL_AMT4": rng.integers(0, 50_000, n),
        "BILL_AMT5": rng.integers(0, 50_000, n),
        "BILL_AMT6": rng.integers(0, 50_000, n),
        "PAY_AMT1": rng.integers(0, 20_000, n),
        "PAY_AMT2": rng.integers(0, 20_000, n),
        "PAY_AMT3": rng.integers(0, 20_000, n),
        "PAY_AMT4": rng.integers(0, 20_000, n),
        "PAY_AMT5": rng.integers(0, 20_000, n),
        "PAY_AMT6": rng.integers(0, 20_000, n),
        "default.payment.next.month": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests: load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_raw(tmp_path / "nonexistent.csv")

    def test_real_file_shape(self) -> None:
        """Real dataset has 30,000 rows and 25 columns (before cleaning)."""
        path = _require_data()
        df = load_raw(path)
        assert df.shape == (30_000, 25), f"Unexpected shape: {df.shape}"

    def test_real_file_has_pay0_column(self) -> None:
        """Raw file must have PAY_0 (not PAY_1) — that's the quirk."""
        path = _require_data()
        df = load_raw(path)
        assert "PAY_0" in df.columns
        assert "PAY_1" not in df.columns

    def test_real_file_has_id_column(self) -> None:
        path = _require_data()
        df = load_raw(path)
        assert "ID" in df.columns

    def test_real_file_target_original_name(self) -> None:
        path = _require_data()
        df = load_raw(path)
        assert "default.payment.next.month" in df.columns


# ---------------------------------------------------------------------------
# Tests: clean — column renames
# ---------------------------------------------------------------------------

class TestCleanColumnRenames:
    def test_id_column_absent(self, raw_mini: pd.DataFrame) -> None:
        df = clean(raw_mini)
        assert "ID" not in df.columns

    def test_pay0_absent_pay1_present(self, raw_mini: pd.DataFrame) -> None:
        df = clean(raw_mini)
        assert "PAY_0" not in df.columns
        assert "PAY_1" in df.columns

    def test_target_renamed_to_default(self, raw_mini: pd.DataFrame) -> None:
        df = clean(raw_mini)
        assert TARGET in df.columns
        assert "default.payment.next.month" not in df.columns

    def test_all_expected_columns_present(self, raw_mini: pd.DataFrame) -> None:
        df = clean(raw_mini)
        expected = set(ALL_FEATURES + [TARGET])
        assert expected == set(df.columns)


# ---------------------------------------------------------------------------
# Tests: clean — EDUCATION quirk
# ---------------------------------------------------------------------------

class TestCleanEducation:
    def test_education_no_undocumented_values(self, raw_mini: pd.DataFrame) -> None:
        df = clean(raw_mini)
        # After cleaning, only {1, 2, 3, 4} should remain
        invalid = set(df["EDUCATION"].unique()) - {1, 2, 3, 4}
        assert invalid == set(), f"Unexpected EDUCATION values after clean: {invalid}"

    def test_education_zero_mapped_to_four(self) -> None:
        row = _make_single_row(EDUCATION=0)
        df = clean(row)
        assert df["EDUCATION"].iloc[0] == 4

    def test_education_five_mapped_to_four(self) -> None:
        row = _make_single_row(EDUCATION=5)
        df = clean(row)
        assert df["EDUCATION"].iloc[0] == 4

    def test_education_six_mapped_to_four(self) -> None:
        row = _make_single_row(EDUCATION=6)
        df = clean(row)
        assert df["EDUCATION"].iloc[0] == 4

    def test_education_valid_values_unchanged(self) -> None:
        for v in [1, 2, 3, 4]:
            row = _make_single_row(EDUCATION=v)
            df = clean(row)
            assert df["EDUCATION"].iloc[0] == v, f"EDUCATION={v} should not be remapped"

    def test_real_data_education_no_quirk_values(self) -> None:
        path = _require_data()
        df = clean(load_raw(path))
        assert set(df["EDUCATION"].unique()).issubset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# Tests: clean — MARRIAGE quirk
# ---------------------------------------------------------------------------

class TestCleanMarriage:
    def test_marriage_no_zero(self, raw_mini: pd.DataFrame) -> None:
        df = clean(raw_mini)
        assert 0 not in df["MARRIAGE"].values

    def test_marriage_zero_mapped_to_three(self) -> None:
        row = _make_single_row(MARRIAGE=0)
        df = clean(row)
        assert df["MARRIAGE"].iloc[0] == 3

    def test_marriage_valid_values_unchanged(self) -> None:
        for v in [1, 2, 3]:
            row = _make_single_row(MARRIAGE=v)
            df = clean(row)
            assert df["MARRIAGE"].iloc[0] == v

    def test_real_data_marriage_no_zero(self) -> None:
        path = _require_data()
        df = clean(load_raw(path))
        assert 0 not in df["MARRIAGE"].values


# ---------------------------------------------------------------------------
# Tests: clean — real dataset shape and values
# ---------------------------------------------------------------------------

class TestCleanRealData:
    def test_shape_after_clean(self) -> None:
        """30,000 rows, exactly len(ALL_FEATURES) + 1 columns."""
        path = _require_data()
        df = clean(load_raw(path))
        assert df.shape == (30_000, len(ALL_FEATURES) + 1)

    def test_no_nulls(self) -> None:
        path = _require_data()
        df = clean(load_raw(path))
        assert df.isnull().sum().sum() == 0

    def test_target_binary(self) -> None:
        path = _require_data()
        df = clean(load_raw(path))
        assert set(df[TARGET].unique()) == {0, 1}

    def test_positive_rate_approx_22_percent(self) -> None:
        """PRD §9: ~22% positive rate."""
        path = _require_data()
        df = clean(load_raw(path))
        pos_rate = df[TARGET].mean()
        assert 0.20 <= pos_rate <= 0.25, f"Unexpected positive rate: {pos_rate:.4f}"

    def test_feature_groups_cover_all_features(self) -> None:
        assert set(CATEGORICAL_FEATURES + NUMERIC_FEATURES) == set(ALL_FEATURES)
        assert len(CATEGORICAL_FEATURES) == 3  # SEX, EDUCATION, MARRIAGE
        assert len(NUMERIC_FEATURES) == 20     # remaining 20 features


# ---------------------------------------------------------------------------
# Tests: make_splits
# ---------------------------------------------------------------------------

class TestMakeSplits:
    def _get_splits(self) -> tuple:
        path = _require_data()
        df = clean(load_raw(path))
        return make_splits(df, seed=config.SEED, test_size=0.20)

    def test_disjoint_indices(self) -> None:
        X_dev, X_hold, _, _ = self._get_splits()
        overlap = set(X_dev.index) & set(X_hold.index)
        assert overlap == set(), f"Index overlap between dev and hold-out: {len(overlap)} rows"

    def test_total_rows(self) -> None:
        X_dev, X_hold, _, _ = self._get_splits()
        assert len(X_dev) + len(X_hold) == 30_000

    def test_dev_size_approx_80_percent(self) -> None:
        X_dev, _, _, _ = self._get_splits()
        assert 23_500 <= len(X_dev) <= 24_500, f"Dev set size unexpected: {len(X_dev)}"

    def test_hold_size_approx_20_percent(self) -> None:
        _, X_hold, _, _ = self._get_splits()
        assert 5_500 <= len(X_hold) <= 6_500, f"Hold-out size unexpected: {len(X_hold)}"

    def test_stratification_dev(self) -> None:
        """Positive rate in dev set must be within 1 pp of the full dataset rate."""
        path = _require_data()
        df = clean(load_raw(path))
        overall_rate = df[TARGET].mean()
        _, _, y_dev, _ = make_splits(df, seed=config.SEED)
        dev_rate = y_dev.mean()
        assert abs(dev_rate - overall_rate) < 0.01, (
            f"Dev positive rate {dev_rate:.4f} deviates > 1pp from {overall_rate:.4f}"
        )

    def test_stratification_hold(self) -> None:
        """Positive rate in hold-out must be within 1 pp of the full dataset rate."""
        path = _require_data()
        df = clean(load_raw(path))
        overall_rate = df[TARGET].mean()
        _, _, _, y_hold = make_splits(df, seed=config.SEED)
        hold_rate = y_hold.mean()
        assert abs(hold_rate - overall_rate) < 0.01, (
            f"Hold-out positive rate {hold_rate:.4f} deviates > 1pp from {overall_rate:.4f}"
        )

    def test_reproducibility_same_seed(self) -> None:
        """Same seed → identical splits (exact index match)."""
        path = _require_data()
        df = clean(load_raw(path))
        X_dev1, X_hold1, y_dev1, y_hold1 = make_splits(df, seed=42)
        X_dev2, X_hold2, y_dev2, y_hold2 = make_splits(df, seed=42)
        pd.testing.assert_index_equal(X_dev1.index, X_dev2.index)
        pd.testing.assert_index_equal(X_hold1.index, X_hold2.index)

    def test_different_seeds_give_different_splits(self) -> None:
        path = _require_data()
        df = clean(load_raw(path))
        X_dev1, _, _, _ = make_splits(df, seed=42)
        X_dev2, _, _, _ = make_splits(df, seed=99)
        # With 30k rows it's essentially impossible for these to be identical
        assert not X_dev1.index.equals(X_dev2.index)

    def test_only_feature_columns_in_X(self) -> None:
        X_dev, X_hold, _, _ = self._get_splits()
        assert set(X_dev.columns) == set(ALL_FEATURES)
        assert TARGET not in X_dev.columns
        assert TARGET not in X_hold.columns

    def test_y_is_binary_series(self) -> None:
        _, _, y_dev, y_hold = self._get_splits()
        assert set(y_dev.unique()).issubset({0, 1})
        assert set(y_hold.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: load_and_prepare convenience wrapper
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_returns_four_objects(self) -> None:
        path = _require_data()
        result = load_and_prepare(path, seed=config.SEED)
        assert len(result) == 4

    def test_same_as_manual_pipeline(self) -> None:
        path = _require_data()
        X_dev1, X_hold1, y_dev1, y_hold1 = load_and_prepare(path, seed=config.SEED)
        df = clean(load_raw(path))
        X_dev2, X_hold2, y_dev2, y_hold2 = make_splits(df, seed=config.SEED)
        pd.testing.assert_frame_equal(X_dev1.reset_index(drop=True),
                                      X_dev2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_single_row(**overrides: int) -> pd.DataFrame:
    """Build a minimal 1-row raw DataFrame for single-value quirk tests."""
    base = {
        "ID": 1, "LIMIT_BAL": 100_000, "SEX": 1,
        "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 5000, "BILL_AMT2": 5000, "BILL_AMT3": 5000,
        "BILL_AMT4": 5000, "BILL_AMT5": 5000, "BILL_AMT6": 5000,
        "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
        "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
        "default.payment.next.month": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])
