"""
analysis.py — Paper Figures + Bootstrap Significance + Result Tables
=====================================================================
Usage:
    python analysis.py                # Generate all plots and tables
    python analysis.py --skip-boot    # Skip bootstrap (fast mode for plot iteration)

Methodology:
  - Results tables: mean ± std across 3 seeds
  - Confusion matrix + bootstrap: majority vote across 3 seeds
    (ties broken by probability averaging)
  - Paired bootstrap: N=10,000 on majority-voted ensemble predictions

Reads saved results from run_experiments.py. No GPU needed.
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)

from config import *


# ═══════════════════════════════════════════════════════════════════════
# LOAD RESULTS
# ═══════════════════════════════════════════════════════════════════════

def load_ablation_results(configs, name_fn):
    """Load test metrics, histories, and predictions for an ablation group."""
    results = {}
    for cfg in configs:
        seeds_data = {}
        for seed in SEEDS:
            run_name = name_fn(cfg, seed)
            run_dir = os.path.join(RESULTS_DIR, run_name)
            metrics_path = os.path.join(run_dir, "test_metrics.json")
            history_path = os.path.join(run_dir, "history.json")
            preds_path = os.path.join(run_dir, "test_predictions.npz")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            history = []
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history = json.load(f)
            preds_data = None
            if os.path.exists(preds_path):
                preds_data = dict(np.load(preds_path))
            seeds_data[seed] = {"metrics": metrics, "history": history, "preds": preds_data}
        if seeds_data:
            results[cfg] = seeds_data
    return results


def load_depth_results():
    return load_ablation_results(
        DEPTH_ABLATION, lambda d, s: f"depth_ablation/layers_{d}_seed_{s}")


def load_fusion_results():
    return load_ablation_results(
        FUSION_MODES, lambda m, s: f"fusion_ablation/{m}_seed_{s}")


# ═══════════════════════════════════════════════════════════════════════
# MAJORITY VOTE ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════

def majority_vote_ensemble(seed_data):
    """
    Combine predictions across seeds using majority vote.
    Ties broken by averaging probabilities and taking argmax.

    Args:
        seed_data: dict of {seed: {"preds": {"preds": array, "labels": array, "probs": array}}}
    Returns:
        ensemble_preds, labels, ensemble_probs
    """
    seeds_with_preds = [sd for sd in seed_data.values() if sd["preds"] is not None]
    if not seeds_with_preds:
        return None, None, None

    all_preds = np.stack([sd["preds"]["preds"] for sd in seeds_with_preds], axis=0)  # (n_seeds, n_samples)
    all_probs = np.stack([sd["preds"]["probs"] for sd in seeds_with_preds], axis=0)  # (n_seeds, n_samples, n_classes)
    labels = seeds_with_preds[0]["preds"]["labels"]
    n_seeds, n_samples = all_preds.shape

    ensemble_preds = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        votes = all_preds[:, i]
        classes, counts = np.unique(votes, return_counts=True)
        max_count = counts.max()
        winners = classes[counts == max_count]

        if len(winners) == 1:
            # Clear majority
            ensemble_preds[i] = winners[0]
        else:
            # Tie: break by averaging probabilities
            avg_probs = all_probs[:, i, :].mean(axis=0)
            ensemble_preds[i] = avg_probs.argmax()

    # Ensemble probs = averaged across seeds (for AUC computation)
    ensemble_probs = all_probs.mean(axis=0)

    return ensemble_preds, labels, ensemble_probs


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: TRAINING CURVES (depth ablation)
# ═══════════════════════════════════════════════════════════════════════

def plot_training_curves(results, save_path):
    """Train/val loss curves for each depth, seeds as mean ± std shading."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, depth in enumerate(DEPTH_ABLATION):
        ax = axes[idx]
        seed_data = results.get(depth, {})
        if not seed_data:
            ax.set_title(f"Depth {depth} — no data")
            continue

        max_ep = max(len(sd["history"]) for sd in seed_data.values())
        train_losses = np.full((len(seed_data), max_ep), np.nan)
        val_losses = np.full((len(seed_data), max_ep), np.nan)

        for si, (seed, sd) in enumerate(seed_data.items()):
            for ei, entry in enumerate(sd["history"]):
                train_losses[si, ei] = entry["train"]["loss"]
                val_losses[si, ei] = entry["val"]["loss"]

        epochs = np.arange(1, max_ep + 1)
        for losses, color, label in [(train_losses, "#4A90D9", "Train"),
                                      (val_losses, "#E74C3C", "Val")]:
            mean = np.nanmean(losses, axis=0)
            std = np.nanstd(losses, axis=0)
            ax.plot(epochs, mean, color=color, label=label, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(f"Depth = {depth} layer{'s' if depth > 1 else ''}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves — Depth Ablation (mean ± std over 3 seeds)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Training curves → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 & 3: ABLATION COMPARISON BAR CHARTS
# ═══════════════════════════════════════════════════════════════════════

def plot_ablation_bars(results, config_labels, title, save_path):
    """Bar chart: balanced accuracy + AUC with error bars (mean ± std across seeds)."""
    configs = list(config_labels.keys())
    labels = list(config_labels.values())
    bacc_means, bacc_stds, auc_means, auc_stds = [], [], [], []

    for cfg in configs:
        seed_data = results.get(cfg, {})
        baccs = [sd["metrics"]["bal_acc"] for sd in seed_data.values()]
        aucs = [sd["metrics"]["auc"] for sd in seed_data.values()]
        bacc_means.append(np.mean(baccs) if baccs else 0)
        bacc_stds.append(np.std(baccs) if baccs else 0)
        auc_means.append(np.mean(aucs) if aucs else 0)
        auc_stds.append(np.std(aucs) if aucs else 0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, bacc_means, width, yerr=bacc_stds, label="Bal. Accuracy",
           color="#4A90D9", capsize=5, alpha=0.85)
    ax.bar(x + width / 2, auc_means, width, yerr=auc_stds, label="AUC-ROC",
           color="#E74C3C", capsize=5, alpha=0.85)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(len(labels)):
        ax.text(x[i] - width / 2, bacc_means[i] + bacc_stds[i] + 0.02,
                f"{bacc_means[i]:.3f}", ha="center", fontsize=9)
        ax.text(x[i] + width / 2, auc_means[i] + auc_stds[i] + 0.02,
                f"{auc_means[i]:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Ablation bars → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: CONFUSION MATRIX (majority vote across seeds)
# ═══════════════════════════════════════════════════════════════════════

def plot_best_confusion_matrix(results, save_path):
    """Confusion matrix from best depth config, majority vote across seeds."""
    best_depth, best_auc = DEPTH_ABLATION[0], 0
    for depth in DEPTH_ABLATION:
        seed_data = results.get(depth, {})
        aucs = [sd["metrics"]["auc"] for sd in seed_data.values()]
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_depth = np.mean(aucs), depth

    seed_data = results.get(best_depth, {})
    if not seed_data:
        print("  [plot] No data for confusion matrix")
        return

    preds, labels, _ = majority_vote_ensemble(seed_data)
    if preds is None:
        print("  [plot] No prediction data for confusion matrix")
        return

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Counts", fontsize=13)
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Normalized", fontsize=13)
    fig.suptitle(f"Confusion Matrix — Best Model (depth={best_depth}, majority vote over 3 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Confusion matrix → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: PER-CLASS F1 COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def plot_per_class_f1(results, config_labels, title, save_path):
    """Grouped bar chart: F1 per class for each config."""
    configs = list(config_labels.keys())
    labels = list(config_labels.values())
    n_classes = len(CLASS_NAMES)
    n_configs = len(configs)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_classes)
    width = 0.8 / n_configs
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12"]

    for ci, cfg in enumerate(configs):
        seed_data = results.get(cfg, {})
        f1_means, f1_stds = [], []
        for cls in CLASS_NAMES:
            f1s = [sd["metrics"]["per_class"][cls]["f1"] for sd in seed_data.values()
                   if "per_class" in sd["metrics"] and cls in sd["metrics"]["per_class"]]
            f1_means.append(np.mean(f1s) if f1s else 0)
            f1_stds.append(np.std(f1s) if f1s else 0)
        offset = (ci - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, f1_means, width, yerr=f1_stds, label=labels[ci],
               color=colors[ci % len(colors)], capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Per-class F1 → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: MODALITY OCCLUSION
# ═══════════════════════════════════════════════════════════════════════

def plot_modality_occlusion(save_path):
    occ_path = os.path.join(RESULTS_DIR, "evaluation", "modality_occlusion.json")
    if not os.path.exists(occ_path):
        print("  [plot] No modality occlusion data found")
        return
    with open(occ_path) as f:
        data = json.load(f)

    conditions = ["full", "audio_zeroed", "video_zeroed"]
    labels = ["Full Model", "Audio Zeroed\n(Video Only)", "Video Zeroed\n(Audio Only)"]
    baccs = [data[c]["bal_acc"] for c in conditions]
    aucs = [data[c]["auc"] for c in conditions]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baccs, width, label="Bal. Accuracy", color="#4A90D9", alpha=0.85)
    ax.bar(x + width / 2, aucs, width, label="AUC-ROC", color="#E74C3C", alpha=0.85)
    for i in range(len(labels)):
        ax.text(x[i] - width / 2, baccs[i] + 0.02, f"{baccs[i]:.3f}", ha="center", fontsize=10)
        ax.text(x[i] + width / 2, aucs[i] + 0.02, f"{aucs[i]:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Modality Occlusion Study", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Modality occlusion → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# PAIRED BOOTSTRAP SIGNIFICANCE (MAJORITY VOTE ENSEMBLE)
# ═══════════════════════════════════════════════════════════════════════

def paired_bootstrap_test(preds_a, labels_a, preds_b, labels_b,
                          metric_fn=balanced_accuracy_score, n_boot=BOOTSTRAP_N):
    """
    Paired bootstrap test on majority-voted ensemble predictions.
    Tests whether model A is significantly better than model B.

    Returns: dict with delta, p_value, ci_lower, ci_upper
    """
    n = len(preds_a)
    observed_delta = metric_fn(labels_a, preds_a) - metric_fn(labels_b, preds_b)

    deltas = np.zeros(n_boot)
    rng = np.random.RandomState(42)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        score_a = metric_fn(labels_a[idx], preds_a[idx])
        score_b = metric_fn(labels_b[idx], preds_b[idx])
        deltas[i] = score_a - score_b

    p_value = np.mean(deltas <= 0)
    alpha = 1 - BOOTSTRAP_CI
    ci_lo = np.percentile(deltas, 100 * alpha / 2)
    ci_hi = np.percentile(deltas, 100 * (1 - alpha / 2))

    return {"delta": float(observed_delta), "p_value": float(p_value),
            "ci_lower": float(ci_lo), "ci_upper": float(ci_hi)}


def run_significance_tests(depth_results, fusion_results, save_path):
    """
    Run paired bootstrap on majority-voted ensemble predictions.
    All 3 seeds contribute to each ensemble via majority vote.
    """
    tests = []

    print("\n  ── Depth comparisons (majority vote ensemble) ──")
    depth_ensembles = {}
    for depth in DEPTH_ABLATION:
        if depth in depth_results:
            preds, labels, probs = majority_vote_ensemble(depth_results[depth])
            if preds is not None:
                depth_ensembles[depth] = {"preds": preds, "labels": labels, "probs": probs}

    for i in range(len(DEPTH_ABLATION) - 1):
        d_a, d_b = DEPTH_ABLATION[i + 1], DEPTH_ABLATION[i]
        if d_a not in depth_ensembles or d_b not in depth_ensembles:
            continue
        result = paired_bootstrap_test(
            depth_ensembles[d_a]["preds"], depth_ensembles[d_a]["labels"],
            depth_ensembles[d_b]["preds"], depth_ensembles[d_b]["labels"],
        )
        result["comparison"] = f"depth_{d_a}_vs_{d_b}"
        tests.append(result)
        sig = "✓ YES" if result["p_value"] < 0.05 else "✗ no"
        print(f"    {d_a} layers vs {d_b} layers: "
              f"Δ={result['delta']:+.4f}  p={result['p_value']:.4f}  "
              f"CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  sig={sig}")

    print("\n  ── Fusion comparisons (majority vote ensemble) ──")
    fusion_ensembles = {}
    for mode in FUSION_MODES:
        if mode in fusion_results:
            preds, labels, probs = majority_vote_ensemble(fusion_results[mode])
            if preds is not None:
                fusion_ensembles[mode] = {"preds": preds, "labels": labels, "probs": probs}

    # Compare cross_modal vs each other mode
    if "cross_modal" in fusion_ensembles:
        for mode in ["concat", "video_only", "audio_only"]:
            if mode not in fusion_ensembles:
                continue
            result = paired_bootstrap_test(
                fusion_ensembles["cross_modal"]["preds"],
                fusion_ensembles["cross_modal"]["labels"],
                fusion_ensembles[mode]["preds"],
                fusion_ensembles[mode]["labels"],
            )
            result["comparison"] = f"cross_modal_vs_{mode}"
            tests.append(result)
            sig = "✓ YES" if result["p_value"] < 0.05 else "✗ no"
            print(f"    cross_modal vs {mode}: "
                  f"Δ={result['delta']:+.4f}  p={result['p_value']:.4f}  "
                  f"CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  sig={sig}")

    with open(save_path, "w") as f:
        json.dump(tests, f, indent=2)
    print(f"\n  [bootstrap] Results → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════

def print_results_table(results, config_labels, title):
    """Print formatted results table: mean ± std across seeds."""
    print(f"\n{'='*85}")
    print(f"  {title}")
    print(f"{'='*85}")
    print(f"  {'Config':<20s} {'Bal.Acc':>14s} {'AUC':>14s} {'ECE':>10s}")
    print(f"  {'-'*58}")
    for cfg, label in config_labels.items():
        seed_data = results.get(cfg, {})
        if not seed_data:
            print(f"  {label:<20s} {'—':>14s} {'—':>14s} {'—':>10s}")
            continue
        baccs = [sd["metrics"]["bal_acc"] for sd in seed_data.values()]
        aucs = [sd["metrics"]["auc"] for sd in seed_data.values()]
        eces = [sd["metrics"].get("ece", 0) for sd in seed_data.values()]
        print(f"  {label:<20s} "
              f"{np.mean(baccs):.4f}±{np.std(baccs):.4f} "
              f"{np.mean(aucs):.4f}±{np.std(aucs):.4f} "
              f"{np.mean(eces):.4f}")

    # Also print majority vote ensemble results
    print(f"\n  Majority vote ensemble:")
    for cfg, label in config_labels.items():
        seed_data = results.get(cfg, {})
        if not seed_data:
            continue
        preds, labels, probs = majority_vote_ensemble(seed_data)
        if preds is None:
            continue
        bacc = balanced_accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0
        print(f"    {label:<20s} bacc={bacc:.4f}  auc={auc:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-boot", action="store_true", help="Skip bootstrap tests")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[analysis] Loading results...")
    depth_results = load_depth_results()
    fusion_results = load_fusion_results()

    if not depth_results and not fusion_results:
        print("[analysis] No results found. Run run_experiments.py first.")
        return

    depth_labels = {d: f"{d} layer{'s' if d > 1 else ''}" for d in DEPTH_ABLATION}
    fusion_labels = {m: m.replace("_", " ").title() for m in FUSION_MODES}

    # Print tables
    if depth_results:
        print_results_table(depth_results, depth_labels, "DEPTH ABLATION RESULTS")
    if fusion_results:
        print_results_table(fusion_results, fusion_labels, "FUSION ABLATION RESULTS")

    # Generate plots
    print("[analysis] Generating plots...")
    if depth_results:
        plot_training_curves(depth_results, os.path.join(PLOTS_DIR, "F2_training_curves.png"))
        plot_ablation_bars(depth_results, depth_labels, "Depth Ablation",
                           os.path.join(PLOTS_DIR, "F3_depth_ablation.png"))
        plot_best_confusion_matrix(depth_results, os.path.join(PLOTS_DIR, "F5_confusion_matrix.png"))
        plot_per_class_f1(depth_results, depth_labels, "Per-Class F1 — Depth Ablation",
                          os.path.join(PLOTS_DIR, "F6a_per_class_f1_depth.png"))

    if fusion_results:
        plot_ablation_bars(fusion_results, fusion_labels, "Fusion Ablation",
                           os.path.join(PLOTS_DIR, "F4_fusion_ablation.png"))
        plot_per_class_f1(fusion_results, fusion_labels, "Per-Class F1 — Fusion Ablation",
                          os.path.join(PLOTS_DIR, "F6b_per_class_f1_fusion.png"))

    plot_modality_occlusion(os.path.join(PLOTS_DIR, "F7_modality_occlusion.png"))

    # Bootstrap significance
    if not args.skip_boot and (depth_results or fusion_results):
        print("\n[analysis] Running paired bootstrap significance tests (majority vote ensemble)...")
        run_significance_tests(depth_results, fusion_results,
                               os.path.join(RESULTS_DIR, "bootstrap_significance.json"))
    elif args.skip_boot:
        print("\n[analysis] Skipping bootstrap (--skip-boot)")

    print(f"\n{'='*60}")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print(f"  All results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
