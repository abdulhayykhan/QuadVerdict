"""
run_benchmark.py — QuadVerdict one-command pipeline entry point.

Usage:
    python run_benchmark.py [--quick] [--full] [--resume] [--seed SEED]

Options:
    --quick         Tiny budget (n_iter=3, 2x1 outer, subsampled data). Output is
                    flagged meta.quick=true; the UI shows a non-publication banner.
    --full          Full benchmark run (15 folds, all data, full n_iter budget).
                    Takes several hours; recommended to run with --resume.
    --resume        Skip outer folds already cached in results/_cache/. Useful after
                    an interrupt.
    --seed SEED     Override the random seed (default: config.SEED = 42).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description="QuadVerdict: cost-sensitive benchmark of four classifiers.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny budget for smoke tests. Output is NOT suitable for publication.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full 15-fold nested CV benchmark (takes several hours).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fast dry run: inspect data and print phase markers without fitting.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from cached per-fold results in results/_cache/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (default: config.SEED = 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src import config
    from src.data import clean, load_raw, make_splits
    from src.nested_cv import run_nested_cv, summarize_nested_cv_results
    from src.pipelines import build_pipeline

    seed = args.seed if args.seed is not None else config.SEED
    is_quick = args.quick or (not args.full)  # Default to quick unless --full is explicitly passed

    print("=" * 60)
    print("QuadVerdict Benchmark")
    print("=" * 60)
    if is_quick:
        print("[WARNING] Running in QUICK mode (smoke test budget).")
        if not args.quick and not args.full:
            print("          Pass --full to run the multi-hour 15-fold publication benchmark.")
    else:
        print("[INFO] Running in FULL publication mode (15 folds, full budget).")

    if args.resume:
        print("[INFO] --resume: will skip completed folds in results/_cache/.")
    print(f"[INFO] Seed: {seed} (from {'--seed arg' if args.seed is not None else 'default'})")
    print()

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Phase 1: Load and clean data
    # -----------------------------------------------------------------------
    print("[Phase 1/10] Load and clean data ...")
    t0 = time.perf_counter()
    df_raw = load_raw(config.DATA_PATH)
    df_clean = clean(df_raw)
    X_dev, X_hold, y_dev, y_hold = make_splits(
        df_clean, seed=seed, test_size=config.TEST_SIZE
    )
    print(
        f"             Dev set: {len(X_dev):,} rows | Hold-out set: {len(X_hold):,} rows "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Build pipelines
    # -----------------------------------------------------------------------
    print("[Phase 2/10] Build pipelines ...")
    t0 = time.perf_counter()
    sample_pipes = {m: build_pipeline(m, imbalance_strategy="balanced") for m in config.MODEL_KEYS}
    print(
        f"             Constructed {len(sample_pipes)} pipeline templates: {list(sample_pipes.keys())} "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    # -----------------------------------------------------------------------
    # Phase 3: Nested Cross-Validation
    # -----------------------------------------------------------------------
    print(f"[Phase 3/10] Nested CV ({'quick' if is_quick else '15-fold full'}) ...")
    if args.dry_run:
        print("             Dry run mode: skipped model fitting.")
    else:
        t0 = time.perf_counter()
        cv_output = run_nested_cv(
            X_dev=X_dev,
            y_dev=y_dev,
            models=config.MODEL_KEYS,
            imbalance_strategy="balanced",
            quick=is_quick,
            resume=args.resume,
            seed=seed,
            cache_dir=config.CACHE_DIR,
            n_jobs=1,
        )
        cv_summary = summarize_nested_cv_results(cv_output)
        print(f"             Completed nested CV in {time.perf_counter() - t0:.2f}s")
        print("\n" + cv_summary.to_string(index=False) + "\n")

    # -----------------------------------------------------------------------
    # Phase 4 to 10 stubs (to be wired in subsequent phases)
    # -----------------------------------------------------------------------
    print("[Phase 4/10] Aggregate metrics ... (implemented in Phase 4)")
    print("[Phase 5/10] Significance tests ... (implemented in Phase 4)")
    print("[Phase 6/10] Permutation importance ... (implemented in Phase 5)")
    print("[Phase 7/10] Imbalance experiment + learning curves ... (implemented in Phase 5)")
    print("[Phase 8/10] Timing ... (implemented in Phase 5)")
    print("[Phase 9/10] Hold-out sanity check ... (implemented in Phase 6)")
    print("[Phase 10/10] Export and validate ... (implemented in Phase 6)")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"[DONE] Benchmark run completed in {elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
