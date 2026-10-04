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
    # Phase 4: Aggregate metrics (Threshold tables, ROC/PR curves, Calibration)
    # -----------------------------------------------------------------------
    print("[Phase 4/10] Aggregate metrics ...")
    if args.dry_run:
        print("             Dry run mode: skipped metrics aggregation.")
    else:
        from src.metrics import (
            build_calibration,
            build_roc_pr,
            build_threshold_table,
            extract_repeat_predictions,
        )

        n_splits = cv_output["meta"]["n_splits"]
        n_repeats = cv_output["meta"]["n_repeats"]

        metrics_payload = {}
        for m in config.MODEL_KEYS:
            folds_m = cv_output["models"][m]
            table = build_threshold_table(
                folds_m, n_splits=n_splits, n_repeats=n_repeats, n_steps=config.N_THRESHOLD_STEPS
            )
            y_t0, y_p0 = extract_repeat_predictions(folds_m, repeat_idx=0)
            curves = build_roc_pr(y_t0, y_p0, max_points=200)
            cal_bins = build_calibration(y_t0, y_p0, n_bins=10)

            metrics_payload[m] = {
                "threshold_table": table,
                "curves": curves,
                "calibration": cal_bins,
            }
        print(f"             Aggregated threshold tables (101 steps), ROC/PR curves, and calibration bins for {len(metrics_payload)} models.")

    # -----------------------------------------------------------------------
    # Phase 5: Significance tests
    # -----------------------------------------------------------------------
    print("[Phase 5/10] Significance tests ...")
    if args.dry_run:
        print("             Dry run mode: skipped significance testing.")
    else:
        from src.significance import run_pairwise_significance

        n_samples = cv_output["meta"]["n_samples"]
        n_test = n_samples // cv_output["meta"]["n_splits"]
        n_train = n_samples - n_test

        pairwise_results = run_pairwise_significance(
            cv_output=cv_output,
            n_train=n_train,
            n_test=n_test,
            metric="pr_auc",
        )
        print(f"             Computed {len(pairwise_results)} pairwise comparisons (Nadeau-Bengio + Wilcoxon + Holm):")
        for pair in pairwise_results:
            sig_flag = "SIGNIFICANT" if pair["significant"] else "not significant"
            print(
                f"               {pair['model_a'].upper()} vs {pair['model_b'].upper()}: "
                f"diff = {pair['mean_diff']:+.4f}, t = {pair['t_stat']:.3f}, p_adj = {pair['p_adj_nb']:.4f} ({sig_flag})"
            )

    # -----------------------------------------------------------------------
    # Phase 6: Permutation importance
    # -----------------------------------------------------------------------
    print("[Phase 6/10] Permutation importance ...")
    if args.dry_run:
        print("             Dry run mode: skipped permutation importance.")
    else:
        from src.explain import run_explainability

        importance_payload = run_explainability(
            models=config.MODEL_KEYS,
            cv_output=cv_output,
            X_dev=X_dev,
            y_dev=y_dev,
            n_repeats=2 if is_quick else 5,
            seed=seed,
            top_n=15,
        )
        for m, top_feats in importance_payload.items():
            top3_str = ", ".join(f"{f['feature']} (+{f['importance_mean']:.4f})" for f in top_feats[:3])
            print(f"             {m.upper()} Top 3: {top3_str}")

    # -----------------------------------------------------------------------
    # Phase 7: Imbalance experiment + learning curves
    # -----------------------------------------------------------------------
    print("[Phase 7/10] Imbalance experiment + learning curves ...")
    if args.dry_run:
        print("             Dry run mode: skipped experiments.")
    else:
        from src.experiments import run_imbalance_experiment, run_learning_curves

        imb_results = run_imbalance_experiment(
            X_dev=X_dev,
            y_dev=y_dev,
            models=config.MODEL_KEYS,
            quick=is_quick,
            seed=seed,
        )
        curves_results = run_learning_curves(
            X_dev=X_dev,
            y_dev=y_dev,
            models=config.MODEL_KEYS,
            quick=is_quick,
            seed=seed,
        )
        print(f"             Evaluated {len(imb_results)} models across 3 imbalance strategies (none, balanced, smote).")
        print(f"             Computed learning curves across {len(curves_results)} models.")

    # -----------------------------------------------------------------------
    # Phase 8: Timing & Hardware Telemetry
    # -----------------------------------------------------------------------
    print("[Phase 8/10] Timing ...")
    if args.dry_run:
        print("             Dry run mode: skipped timing benchmarks.")
    else:
        from src.timing import measure_timing_and_size

        timing_payload = measure_timing_and_size(
            models=config.MODEL_KEYS,
            cv_output=cv_output,
            X_dev=X_dev,
            y_dev=y_dev,
            X_hold=X_hold,
            n_inference_runs=5 if is_quick else 10,
            seed=seed,
        )
        hw = timing_payload["hardware"]
        print(f"             Hardware: {hw['cpu_processor']} ({hw['cpu_cores']} cores) on {hw['os_name']} {hw['os_release']}")
        for m, t_info in timing_payload["models"].items():
            print(f"               {m.upper()}: train = {t_info['train_s']:.2f}s | infer = {t_info['infer_ms_per_1k']:.2f} ms/1k | size = {t_info['model_kb']:.1f} KB")

    # -----------------------------------------------------------------------
    # Phase 9: Hold-out sanity check (TRD §4.9)
    # -----------------------------------------------------------------------
    print("[Phase 9/10] Hold-out sanity check ...")
    if args.dry_run:
        print("             Dry run mode: skipped hold-out evaluation.")
    else:
        from src.export import evaluate_holdout_set

        holdout_payload = evaluate_holdout_set(
            models=config.MODEL_KEYS,
            cv_output=cv_output,
            X_dev=X_dev,
            y_dev=y_dev,
            X_hold=X_hold,
            y_hold=y_hold,
            seed=seed,
        )
        for m, h_scores in holdout_payload.items():
            print(f"             {m.upper()} Hold-out: PR-AUC = {h_scores['pr_auc']:.4f}, ROC-AUC = {h_scores['roc_auc']:.4f}")

    # -----------------------------------------------------------------------
    # Phase 10: Export and validate (TRD §5)
    # -----------------------------------------------------------------------
    print("[Phase 10/10] Export and validate ...")
    if args.dry_run:
        print("             Dry run mode: skipped export and schema validation.")
    else:
        from src.export import assemble_results, validate, write

        models_data = {}
        for m in config.MODEL_KEYS:
            models_data[m] = {
                "folds": cv_output["models"][m],
                "thresholds": metrics_payload[m]["threshold_table"],
                "roc": metrics_payload[m]["curves"]["roc"],
                "pr": metrics_payload[m]["curves"]["pr"],
                "calibration": metrics_payload[m]["calibration"],
                "timing": timing_payload["models"][m],
                "importance": importance_payload[m],
                "holdout": holdout_payload[m],
            }

        meta_info = {
            "n": len(X_dev),
            "n_pos": int((y_dev == 1).sum()),
            "n_neg": int((y_dev == 0).sum()),
            "pos_rate": float((y_dev == 1).mean()),
            "seed": seed,
            "outer_splits": cv_output["meta"]["n_splits"],
            "outer_repeats": cv_output["meta"]["n_repeats"],
            "inner_splits": cv_output["meta"]["inner_splits"],
            "n_iter": cv_output["meta"]["n_iter"],
            "quick": is_quick,
        }

        results_payload = assemble_results(
            meta_info=meta_info,
            models_data=models_data,
            imbalance_exp=imb_results,
            significance_data=pairwise_results,
            learning_curves_data=curves_results,
        )

        validate(results_payload, schema_path=config.RESULTS_DIR / "schema.json")
        out_file = config.RESULTS_DIR / "results.json"
        bytes_written = write(results_payload, out_file, minified=True)
        print(f"             Validated against schema.json and exported {bytes_written / 1024.0:.1f} KB to {out_file}")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"[DONE] Benchmark run completed in {elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
