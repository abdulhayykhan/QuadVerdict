"""
download.py — Fetch the UCI Default of Credit Card Clients dataset.

Dataset: UCI Default of Credit Card Clients (Taiwan, 2005)
Source:  Kaggle mirror — uciml/default-of-credit-card-clients-dataset
License: CC0-1.0 (Public Domain) — confirmed on Kaggle dataset page.
         Safe to include in the repo; data/raw/ is NOT gitignored for this project.

Usage:
    python data/download.py

The script is idempotent: if the file already exists it prints a notice and exits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EXPECTED_FILE = RAW_DIR / "UCI_Credit_Card.csv"

# Kaggle dataset identifier (owner/dataset-slug)
KAGGLE_DATASET = "uciml/default-of-credit-card-clients-dataset"

LICENSE_NOTICE = """
Dataset: UCI Default of Credit Card Clients
Source : https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
License: CC0-1.0 (Public Domain)
Original paper: Yeh, I-C., & Lien, C-H. (2009).
  "The comparisons of data mining techniques for the predictive accuracy
   of probability of default of credit card clients."
  Expert Systems with Applications, 36(2), 2473-2480.
"""


def download() -> None:
    if EXPECTED_FILE.exists():
        print(f"[download.py] Dataset already present at {EXPECTED_FILE}")
        print(f"             Size: {EXPECTED_FILE.stat().st_size / 1_048_576:.2f} MB")
        return

    print(LICENSE_NOTICE)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401 — validates credentials on import
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] kaggle credentials not configured: {exc}", file=sys.stderr)
        print("        Set up ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars.")
        sys.exit(1)

    import subprocess

    print(f"[download.py] Downloading {KAGGLE_DATASET} → {RAW_DIR} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(RAW_DIR),
            "--unzip",
        ],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[ERROR] kaggle download failed.", file=sys.stderr)
        sys.exit(result.returncode)

    if not EXPECTED_FILE.exists():
        # The Kaggle CSV might have a slightly different name; try to find it
        candidates = list(RAW_DIR.glob("*.csv"))
        if len(candidates) == 1:
            candidates[0].rename(EXPECTED_FILE)
            print(f"[download.py] Renamed {candidates[0].name} → {EXPECTED_FILE.name}")
        else:
            print(
                f"[ERROR] Expected {EXPECTED_FILE.name} but found: {[c.name for c in candidates]}",
                file=sys.stderr,
            )
            sys.exit(1)

    size_mb = EXPECTED_FILE.stat().st_size / 1_048_576
    print(f"[download.py] Done. {EXPECTED_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download()
