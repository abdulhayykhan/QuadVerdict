"""
run_benchmark.py — QuadVerdict one-command pipeline entry point.

Usage:
    python run_benchmark.py [--quick] [--resume] [--seed SEED]

Options:
    --quick         Tiny budget (n_iter=3, 2×1 outer, subsampled data). Output is
                    flagged meta.quick=true; the UI shows a non-publication banner.
    --resume        Skip outer folds already cached in results/_cache/. Useful after
                    an interrupt.
    --seed SEED     Override the random seed (default: 42, from config.SEED).

Phases (to be wired up as each src/ module is implemented):
    1. Load and clean data          (src/data.py)
    2. Build pipelines              (src/pipelines.py)
    3. Nested CV                    (src/nested_cv.py)
    4. Aggregate metrics            (src/metrics.py)
    5. Significance tests           (src/significance.py)
    6. Permutation importance       (src/explain.py)
    7. Imbalance experiment +
       learning curves              (src/experiments.py)
    8. Timing                       (src/timing.py)
    9. Hold-out sanity check        (src/nested_cv.py)
   10. Export and validate          (src/export.py)
"""

import argparse
import sys
import time


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


def main() -> None:
    args = parse_args()

    # Lazy import so the argument parser is always fast even before deps are installed.
    # Each import will be uncommented as the corresponding module is implemented.
    # from src import config  # noqa: F401

    seed = args.seed  # will fall back to config.SEED once config is wired

    print("=" * 60)
    print("QuadVerdict Benchmark")
    print("=" * 60)
    if args.quick:
        print("[WARNING] Running in QUICK mode — NOT for publication.")
    if args.resume:
        print("[INFO] --resume: will skip completed folds in results/_cache/.")
    print(f"[INFO] Seed: {seed if seed is not None else 42} (from {'--seed arg' if seed else 'default'})")
    print()

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Phase stubs — each will be filled in during its respective phase.
    # -----------------------------------------------------------------------

    print("[Phase 1/10] Load and clean data ... (not yet implemented)")
    print("[Phase 2/10] Build pipelines ... (not yet implemented)")
    print("[Phase 3/10] Nested CV ... (not yet implemented)")
    print("[Phase 4/10] Aggregate metrics ... (not yet implemented)")
    print("[Phase 5/10] Significance tests ... (not yet implemented)")
    print("[Phase 6/10] Permutation importance ... (not yet implemented)")
    print("[Phase 7/10] Imbalance experiment + learning curves ... (not yet implemented)")
    print("[Phase 8/10] Timing ... (not yet implemented)")
    print("[Phase 9/10] Hold-out sanity check ... (not yet implemented)")
    print("[Phase 10/10] Export and validate ... (not yet implemented)")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"[DONE] Skeleton run completed in {elapsed:.2f}s (all phases are stubs).")
    print("Run `pytest` to verify the environment is healthy.")


if __name__ == "__main__":
    sys.exit(main())
