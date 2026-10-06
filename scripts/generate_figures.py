"""
generate_figures.py — Generates publication-grade static figures for QuadVerdict.
Uses the Okabe-Ito colorblind-safe palette:
LR:  #E69F00 (Orange)
DT:  #56B4E9 (Sky Blue)
RF:  #009E73 (Bluish Green)
SVM: #D55E00 (Vermilion)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.edgecolor"] = "#30363d"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e1e4e8"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.6

PALETTE = {
    "lr": "#E69F00",
    "dt": "#56B4E9",
    "rf": "#009E73",
    "svm": "#D55E00",
}
LABELS = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results" / "results.json"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> dict:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def plot_cv_boxplots(data: dict) -> None:
    models = ["lr", "dt", "rf", "svm"]
    pr_scores = [data["models"][m]["cv_scores"]["pr_auc"] for m in models]
    roc_scores = [data["models"][m]["cv_scores"]["roc_auc"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # PR-AUC
    bp0 = axes[0].boxplot(
        pr_scores,
        tick_labels=[LABELS[m] for m in models],
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch, m in zip(bp0["boxes"], models, strict=False):
        patch.set_facecolor(PALETTE[m])
        patch.set_alpha(0.85)
    axes[0].set_title("PR-AUC (15 Outer Folds)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_ylabel("Precision-Recall AUC", fontsize=10)
    axes[0].grid(True, axis="y")
    axes[0].tick_params(axis="x", rotation=15)

    # Annotate RF vs LR significance
    axes[0].annotate(
        "RF vs All: p_adj = 0.0000 ***",
        xy=(3, 0.58),
        xytext=(3, 0.585),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#009E73",
    )

    # ROC-AUC
    bp1 = axes[1].boxplot(
        roc_scores,
        tick_labels=[LABELS[m] for m in models],
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 1.5},
    )
    for patch, m in zip(bp1["boxes"], models, strict=False):
        patch.set_facecolor(PALETTE[m])
        patch.set_alpha(0.85)
    axes[1].set_title("ROC-AUC (15 Outer Folds)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylabel("ROC AUC", fontsize=10)
    axes[1].grid(True, axis="y")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    out_path = OUT_DIR / "cv_boxplots.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_roc_pr_curves(data: dict) -> None:
    models = ["lr", "dt", "rf", "svm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # ROC Curves
    for m in models:
        pts = data["models"][m]["roc"]
        fpr = [p[0] for p in pts]
        tpr = [p[1] for p in pts]
        mean_auc = data["models"][m]["cv_summary"]["roc_auc"]["mean"]
        axes[0].plot(
            fpr,
            tpr,
            label=f"{LABELS[m]} (AUC = {mean_auc:.3f})",
            color=PALETTE[m],
            linewidth=2.0,
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=True)
    axes[0].grid(True)

    # PR Curves
    pos_rate = data["meta"]["pos_rate"]
    for m in models:
        pts = data["models"][m]["pr"]
        rec = [p[0] for p in pts]
        prec = [p[1] for p in pts]
        mean_pr = data["models"][m]["cv_summary"]["pr_auc"]["mean"]
        axes[1].plot(
            rec,
            prec,
            label=f"{LABELS[m]} (PR-AUC = {mean_pr:.3f})",
            color=PALETTE[m],
            linewidth=2.0,
        )
    axes[1].axhline(
        pos_rate,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Chance Baseline ({pos_rate:.1%})",
    )
    axes[1].set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(loc="upper right", fontsize=8.5, frameon=True)
    axes[1].grid(True)

    plt.tight_layout()
    out_path = OUT_DIR / "roc_pr_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_calibration(data: dict) -> None:
    models = ["lr", "dt", "rf", "svm"]
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")

    for m in models:
        cal = data["models"][m]["calibration"]
        pred_p = [b["mean_pred"] for b in cal if b.get("n", 1) > 0]
        true_p = [b["frac_pos"] for b in cal if b.get("n", 1) > 0]
        brier = data["models"][m]["cv_summary"]["brier"]["mean"]
        ax.plot(
            pred_p,
            true_p,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=f"{LABELS[m]} (Brier = {brier:.3f})",
            color=PALETTE[m],
        )

    ax.set_title("Probability Calibration Diagram (10 Bins)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives (Empirical)", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(True)

    plt.tight_layout()
    out_path = OUT_DIR / "calibration_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_cost_curves(data: dict) -> None:
    models = ["lr", "dt", "rf", "svm"]
    cost_ratio = data["meta"].get("default_cost_ratio", 5.0)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for m in models:
        th_table = data["models"][m]["thresholds"]
        t_vals = [r["t"] for r in th_table]
        costs = [(r["fp"] * 1.0 + r["fn"] * cost_ratio) for r in th_table]
        min_idx = int(np.argmin(costs))
        opt_t = t_vals[min_idx]
        min_cost = costs[min_idx]

        ax.plot(t_vals, costs, label=f"{LABELS[m]} (min: {min_cost:,.0f} @ t={opt_t:.2f})", color=PALETTE[m], linewidth=2.0)
        ax.plot(opt_t, min_cost, marker="*", markersize=10, color=PALETTE[m])

    ax.set_title(f"Expected Cost vs Classification Threshold (FN:FP = {int(cost_ratio)}:1)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Classification Decision Threshold (t)", fontsize=10)
    ax.set_ylabel("Total Expected Misclassification Cost", fontsize=10)
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9, frameon=True)

    plt.tight_layout()
    out_path = OUT_DIR / "cost_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_feature_importance(data: dict) -> None:
    models = ["lr", "dt", "rf", "svm"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for idx, m in enumerate(models):
        imp_data = data["models"][m].get("importance", {})
        feats = imp_data.get("permutation", [])[:8]
        names = [f["feature"] for f in feats][::-1]
        means = [f["mean"] for f in feats][::-1]
        stds = [f["std"] for f in feats][::-1]

        y_pos = np.arange(len(names))
        axes[idx].barh(y_pos, means, xerr=stds, align="center", color=PALETTE[m], alpha=0.85, capsize=3)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(names, fontsize=9)
        axes[idx].set_title(f"{LABELS[m]} Top Features", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Mean ΔPR-AUC Permutation Loss", fontsize=9)
        axes[idx].grid(True, axis="x")

    plt.suptitle("Permutation Feature Importance (Top Features across 15 Outer Folds)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = OUT_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_imbalance_experiment(data: dict) -> None:
    records = data.get("imbalance_experiment", [])
    if not records:
        return

    models = ["lr", "dt", "rf", "svm"]
    strategies = ["none", "balanced", "smote"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    colors = {"none": "#6c757d", "balanced": "#009E73", "smote": "#56B4E9"}
    strat_labels = {"none": "None (Vanilla)", "balanced": "Balanced Class Weights", "smote": "SMOTE Oversampling"}

    for idx, strat in enumerate(strategies):
        scores = []
        for m in models:
            match = next((r for r in records if r["model"] == m and r["strategy"] == strat), None)
            scores.append(match["pr_auc_mean"] if match else 0.0)

        offset = (idx - 1) * width
        rects = ax.bar(x + offset, scores, width, label=strat_labels[strat], color=colors[strat], alpha=0.85)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_title("Impact of Imbalance Handling Strategy on PR-AUC", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9.5)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.set_ylim([0.45, 0.60])
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    out_path = OUT_DIR / "imbalance_experiment.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_learning_curves(data: dict) -> None:
    lc_dict = data.get("learning_curves", {})
    if not lc_dict:
        return

    models = ["lr", "dt", "rf", "svm"]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for m in models:
        if m not in lc_dict:
            continue
        m_data = lc_dict[m]
        sizes = m_data["sizes"]
        val_scores = m_data["val"]
        train_scores = m_data["train"]

        ax.plot(sizes, val_scores, marker="o", color=PALETTE[m], label=f"{LABELS[m]} (Val)", linewidth=1.8)
        ax.plot(sizes, train_scores, marker="x", linestyle=":", color=PALETTE[m], alpha=0.5)

    ax.set_title("Learning Curves (Validation PR-AUC vs Training Set Size)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Training Samples", fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    out_path = OUT_DIR / "learning_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print(f"Loading benchmark data from: {RESULTS_FILE}")
    data = load_data()
    print(f"Project: {data['meta']['project']}, n={data['meta']['n']}, models={list(data['models'].keys())}")

    plot_cv_boxplots(data)
    plot_roc_pr_curves(data)
    plot_calibration(data)
    plot_cost_curves(data)
    plot_feature_importance(data)
    plot_imbalance_experiment(data)
    plot_learning_curves(data)
    print("All figures successfully generated in results/figures/")


if __name__ == "__main__":
    main()
