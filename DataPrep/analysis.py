"""
analysis.py — Paper Figures + Bootstrap Significance + Result Tables
=====================================================================
Usage:
    python analysis.py                # Generate all plots and tables
    python analysis.py --skip-boot    # Skip bootstrap (fast mode for plot iteration)

Reads saved results from run_experiments.py. No GPU needed.
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import *


# ═══════════════════════════════════════════════════════════════════════
# LOAD RESULTS
# ═══════════════════════════════════════════════════════════════════════

def load_ablation_results(ablation_type, configs, name_fn):
    """Load test metrics and histories for an ablation group."""
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
                preds_data = np.load(preds_path)
            seeds_data[seed] = {"metrics": metrics, "history": history, "preds": preds_data}
        results[cfg] = seeds_data
    return results


def load_depth_results():
    return load_ablation_results(
        "depth", DEPTH_ABLATION,
        lambda d, s: f"depth_ablation/layers_{d}_seed_{s}"
    )


def load_fusion_results():
    return load_ablation_results(
        "fusion", FUSION_MODES,
        lambda m, s: f"fusion_ablation/{m}_seed_{s}"
    )


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

        # Collect per-epoch train/val loss across seeds
        max_ep = max(len(sd["history"]) for sd in seed_data.values())
        train_losses = np.full((len(SEEDS), max_ep), np.nan)
        val_losses = np.full((len(SEEDS), max_ep), np.nan)

        for si, seed in enumerate(SEEDS):
            if seed not in seed_data:
                continue
            hist = seed_data[seed]["history"]
            for ei, entry in enumerate(hist):
                train_losses[si, ei] = entry["train"]["loss"]
                val_losses[si, ei] = entry["val"]["loss"]

        epochs = np.arange(1, max_ep + 1)
        for losses, color, label in [(train_losses, "#4A90D9", "Train"),
                                      (val_losses, "#E74C3C", "Val")]:
            mean = np.nanmean(losses, axis=0)
            std = np.nanstd(losses, axis=0)
            ax.plot(epochs, mean, color=color, label=label, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(f"Depth = {depth} layers", fontsize=13, fontweight="bold")
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
    """Bar chart: balanced accuracy + AUC with error bars."""
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

    # Add value labels
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
# PLOT 4: CONFUSION MATRIX (best model, majority seed vote)
# ═══════════════════════════════════════════════════════════════════════

def plot_best_confusion_matrix(results, save_path):
    """Confusion matrix from the best depth config, majority vote across seeds."""
    # Find best depth by mean AUC
    best_depth, best_auc = DEPTH_ABLATION[0], 0
    for depth in DEPTH_ABLATION:
        seed_data = results.get(depth, {})
        aucs = [sd["metrics"]["auc"] for sd in seed_data.values()]
        if aucs and np.mean(aucs) > best_auc:
            best_auc = np.mean(aucs)
            best_depth = depth

    # Majority vote: for each test sample, pick the most-predicted class across seeds
    seed_data = results[best_depth]
    all_preds = []
    labels = None
    for seed, sd in seed_data.items():
        if sd["preds"] is not None:
            all_preds.append(sd["preds"]["preds"])
            if labels is None:
                labels = sd["preds"]["labels"]

    if not all_preds:
        print("  [plot] No prediction data for confusion matrix")
        return

    preds_stack = np.stack(all_preds, axis=0)  # (n_seeds, n_samples)
    from scipy.stats import mode
    majority_preds = mode(preds_stack, axis=0, keepdims=False).mode

    cm = confusion_matrix(labels, majority_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Counts", fontsize=13)
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Normalized", fontsize=13)
    fig.suptitle(f"Confusion Matrix — Best Model (depth={best_depth}, majority vote)", fontsize=14)
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
# BOOTSTRAP SIGNIFICANCE TEST
# ═══════════════════════════════════════════════════════════════════════

def paired_bootstrap_test(preds_a, labels_a, preds_b, labels_b, metric_fn, n_boot=BOOTSTRAP_N):
    """
    Paired bootstrap: test whether metric(A) > metric(B) significantly.
    Returns: delta, p_value, ci_lower, ci_upper
    """
    assert len(preds_a) == len(preds_b) == len(labels_a)
    n = len(preds_a)
    observed_delta = metric_fn(labels_a, preds_a) - metric_fn(labels_b, preds_b)

    deltas = np.zeros(n_boot)
    rng = np.random.RandomState(42)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        score_a = metric_fn(labels_a[idx], preds_a[idx])
        score_b = metric_fn(labels_b[idx], preds_b[idx])
        deltas[i] = score_a - score_b

    p_value = np.mean(deltas <= 0)  # proportion where B >= A
    alpha = 1 - BOOTSTRAP_CI
    ci_lo = np.percentile(deltas, 100 * alpha / 2)
    ci_hi = np.percentile(deltas, 100 * (1 - alpha / 2))

    return {"delta": observed_delta, "p_value": p_value, "ci_lower": ci_lo, "ci_upper": ci_hi}


def run_significance_tests(depth_results, fusion_results, save_path):
    """Run bootstrap tests between adjacent configs."""
    tests = []

    # Depth: compare adjacent (1v2, 2v3, 3v4)
    for i in range(len(DEPTH_ABLATION) - 1):
        d_a, d_b = DEPTH_ABLATION[i + 1], DEPTH_ABLATION[i]
        # Use first seed's predictions for bootstrap (largest sample)
        seed = SEEDS[0]
        data_a = depth_results.get(d_a, {}).get(seed, {}).get("preds")
        data_b = depth_results.get(d_b, {}).get(seed, {}).get("preds")
        if data_a is None or data_b is None:
            continue
        result = paired_bootstrap_test(
            data_a["preds"], data_a["labels"],
            data_b["preds"], data_b["labels"],
            balanced_accuracy_score,
        )
        result["comparison"] = f"depth_{d_a}_vs_{d_b}"
        tests.append(result)
        sig = "YES" if result["p_value"] < 0.05 else "no"
        print(f"  Depth {d_a} vs {d_b}: Δ={result['delta']:.4f} "
              f"p={result['p_value']:.4f} CI=[{result['ci_lower']:.4f},{result['ci_upper']:.4f}] "
              f"sig={sig}")

    # Fusion: cross_modal vs each other
    for mode in ["concat", "video_only", "audio_only"]:
        seed = SEEDS[0]
        data_a = fusion_results.get("cross_modal", {}).get(seed, {}).get("preds")
        data_b = fusion_results.get(mode, {}).get(seed, {}).get("preds")
        if data_a is None or data_b is None:
            continue
        result = paired_bootstrap_test(
            data_a["preds"], data_a["labels"],
            data_b["preds"], data_b["labels"],
            balanced_accuracy_score,
        )
        result["comparison"] = f"cross_modal_vs_{mode}"
        tests.append(result)
        sig = "YES" if result["p_value"] < 0.05 else "no"
        print(f"  cross_modal vs {mode}: Δ={result['delta']:.4f} "
              f"p={result['p_value']:.4f} sig={sig}")

    with open(save_path, "w") as f:
        json.dump(tests, f, indent=2)
    print(f"  [bootstrap] Results → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def print_results_table(results, config_labels, title):
    """Print a formatted results table to console."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Config':<20s} {'Bal.Acc':>12s} {'AUC':>12s} {'ECE':>10s}")
    print(f"  {'-'*54}")
    for cfg, label in config_labels.items():
        seed_data = results.get(cfg, {})
        baccs = [sd["metrics"]["bal_acc"] for sd in seed_data.values()]
        aucs = [sd["metrics"]["auc"] for sd in seed_data.values()]
        eces = [sd["metrics"].get("ece", 0) for sd in seed_data.values()]
        if baccs:
            print(f"  {label:<20s} {np.mean(baccs):.4f}±{np.std(baccs):.4f} "
                  f"{np.mean(aucs):.4f}±{np.std(aucs):.4f} "
                  f"{np.mean(eces):.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-boot", action="store_true", help="Skip bootstrap tests")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load results
    print("[analysis] Loading results...")
    depth_results = load_depth_results()
    fusion_results = load_fusion_results()

    depth_labels = {d: f"{d} layer{'s' if d > 1 else ''}" for d in DEPTH_ABLATION}
    fusion_labels = {m: m.replace("_", " ").title() for m in FUSION_MODES}

    # Print tables
    print_results_table(depth_results, depth_labels, "DEPTH ABLATION RESULTS")
    print_results_table(fusion_results, fusion_labels, "FUSION ABLATION RESULTS")

    # Generate all plots
    print("\n[analysis] Generating plots...")
    plot_training_curves(depth_results, os.path.join(PLOTS_DIR, "F2_training_curves.png"))
    plot_ablation_bars(depth_results, depth_labels, "Depth Ablation",
                       os.path.join(PLOTS_DIR, "F3_depth_ablation.png"))
    plot_ablation_bars(fusion_results, fusion_labels, "Fusion Ablation",
                       os.path.join(PLOTS_DIR, "F4_fusion_ablation.png"))
    plot_best_confusion_matrix(depth_results, os.path.join(PLOTS_DIR, "F5_confusion_matrix.png"))
    plot_per_class_f1(depth_results, depth_labels, "Per-Class F1 — Depth Ablation",
                      os.path.join(PLOTS_DIR, "F6a_per_class_f1_depth.png"))
    plot_per_class_f1(fusion_results, fusion_labels, "Per-Class F1 — Fusion Ablation",
                      os.path.join(PLOTS_DIR, "F6b_per_class_f1_fusion.png"))
    plot_modality_occlusion(os.path.join(PLOTS_DIR, "F7_modality_occlusion.png"))

    # Bootstrap significance
    if not args.skip_boot:
        print("\n[analysis] Running bootstrap significance tests...")
        from sklearn.metrics import balanced_accuracy_score
        run_significance_tests(depth_results, fusion_results,
                               os.path.join(RESULTS_DIR, "bootstrap_significance.json"))
    else:
        print("\n[analysis] Skipping bootstrap (--skip-boot)")

    print(f"\n✓ All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
