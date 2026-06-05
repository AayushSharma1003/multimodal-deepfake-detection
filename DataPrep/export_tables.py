"""
export_tables.py — Generate results tables from per-run test_metrics.json files.
Saves CSV + Markdown tables for: depth ablation, fusion ablation, bootstrap significance.
Run from the DataPrep directory.
"""

import json
import os
import csv
import numpy as np

RESULTS_ROOT = r"C:\Users\Komal-Sch\Desktop\LAV-DF-preprocessed-20k\results"
OUTPUT_DIR = r"C:\Users\Komal-Sch\Desktop\LAV-DF-preprocessed-20k\results\tables"
SEEDS = [7, 42, 123]


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def mean_std(values):
    return np.mean(values), np.std(values)


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def collect_depth_ablation():
    rows = []
    for depth in [1, 2, 3, 4]:
        baccs, aucs, eces = [], [], []
        for seed in SEEDS:
            p = os.path.join(RESULTS_ROOT, "depth_ablation", f"layers_{depth}_seed_{seed}", "test_metrics.json")
            if not os.path.exists(p):
                print(f"  [MISSING] {p}")
                continue
            m = load_metrics(p)
            baccs.append(m.get("balanced_accuracy", m.get("test_balanced_accuracy", 0)))
            aucs.append(m.get("auc_macro", m.get("test_auc", 0)))
            eces.append(m.get("ece", m.get("test_ece", 0)))
        if baccs:
            ba_m, ba_s = mean_std(baccs)
            auc_m, auc_s = mean_std(aucs)
            ece_m, ece_s = mean_std(eces)
            rows.append({
                "Config": f"{depth} layer{'s' if depth > 1 else ''}",
                "Bal.Acc (mean±std)": fmt(ba_m, ba_s),
                "AUC (mean±std)": fmt(auc_m, auc_s),
                "ECE (mean)": f"{ece_m:.4f}",
                "Bal.Acc_mean": ba_m,
                "AUC_mean": auc_m,
                "seed_baccs": baccs,
                "seed_aucs": aucs,
            })
    return rows


def collect_fusion_ablation():
    rows = []
    for mode in ["video_only", "audio_only", "concat", "cross_modal"]:
        baccs, aucs, eces = [], [], []
        for seed in SEEDS:
            p = os.path.join(RESULTS_ROOT, "fusion_ablation", f"{mode}_seed_{seed}", "test_metrics.json")
            if not os.path.exists(p):
                print(f"  [MISSING] {p}")
                continue
            m = load_metrics(p)
            baccs.append(m.get("balanced_accuracy", m.get("test_balanced_accuracy", 0)))
            aucs.append(m.get("auc_macro", m.get("test_auc", 0)))
            eces.append(m.get("ece", m.get("test_ece", 0)))
        if baccs:
            ba_m, ba_s = mean_std(baccs)
            auc_m, auc_s = mean_std(aucs)
            ece_m, ece_s = mean_std(eces)
            label = mode.replace("_", " ").title()
            rows.append({
                "Config": label,
                "Bal.Acc (mean±std)": fmt(ba_m, ba_s),
                "AUC (mean±std)": fmt(auc_m, auc_s),
                "ECE (mean)": f"{ece_m:.4f}",
                "Bal.Acc_mean": ba_m,
                "AUC_mean": auc_m,
                "seed_baccs": baccs,
                "seed_aucs": aucs,
            })
    return rows


def load_bootstrap():
    p = os.path.join(RESULTS_ROOT, "bootstrap_significance.json")
    if not os.path.exists(p):
        print(f"  [MISSING] {p}")
        return []
    with open(p, "r") as f:
        data = json.load(f)
    rows = []
    # Handle both possible formats (list of dicts or dict of dicts)
    if isinstance(data, list):
        for entry in data:
            rows.append({
                "Comparison": entry.get("comparison", entry.get("name", "")),
                "Delta (BAcc)": f"{entry.get('delta', entry.get('delta_bacc', 0)):+.4f}",
                "p-value": f"{entry.get('p_value', entry.get('p', 0)):.4f}",
                "95% CI": f"[{entry.get('ci_lower', entry.get('ci', [0,0])[0]):.4f}, {entry.get('ci_upper', entry.get('ci', [0,0])[1]):.4f}]",
                "Significant": "Yes" if entry.get("significant", False) else "No",
            })
    elif isinstance(data, dict):
        for key, entry in data.items():
            if isinstance(entry, dict):
                ci = entry.get("ci", entry.get("confidence_interval", [0, 0]))
                if isinstance(ci, list) and len(ci) == 2:
                    ci_lo, ci_hi = ci
                else:
                    ci_lo = entry.get("ci_lower", 0)
                    ci_hi = entry.get("ci_upper", 0)
                rows.append({
                    "Comparison": key,
                    "Delta (BAcc)": f"{entry.get('delta', entry.get('delta_bacc', 0)):+.4f}",
                    "p-value": f"{entry.get('p_value', entry.get('p', 0)):.4f}",
                    "95% CI": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                    "Significant": "Yes" if entry.get("significant", False) else "No",
                })
    return rows


def save_csv(rows, columns, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {filepath}")


def save_markdown(rows, columns, filepath, title):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        # Header
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        # Rows
        for row in rows:
            vals = [str(row.get(c, "")) for c in columns]
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")
    print(f"  [MD]  {filepath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Depth Ablation ---
    print("\n=== Depth Ablation ===")
    depth_rows = collect_depth_ablation()
    cols = ["Config", "Bal.Acc (mean±std)", "AUC (mean±std)", "ECE (mean)"]
    save_csv(depth_rows, cols, os.path.join(OUTPUT_DIR, "depth_ablation_results.csv"))
    save_markdown(depth_rows, cols, os.path.join(OUTPUT_DIR, "depth_ablation_results.md"),
                  "Depth Ablation Results (mean ± std across 3 seeds)")

    # --- Fusion Ablation ---
    print("\n=== Fusion Ablation ===")
    fusion_rows = collect_fusion_ablation()
    save_csv(fusion_rows, cols, os.path.join(OUTPUT_DIR, "fusion_ablation_results.csv"))
    save_markdown(fusion_rows, cols, os.path.join(OUTPUT_DIR, "fusion_ablation_results.md"),
                  "Fusion Ablation Results (mean ± std across 3 seeds)")

    # --- Bootstrap Significance ---
    print("\n=== Bootstrap Significance ===")
    boot_rows = load_bootstrap()
    boot_cols = ["Comparison", "Delta (BAcc)", "p-value", "95% CI", "Significant"]
    if boot_rows:
        save_csv(boot_rows, boot_cols, os.path.join(OUTPUT_DIR, "bootstrap_significance_table.csv"))
        save_markdown(boot_rows, boot_cols, os.path.join(OUTPUT_DIR, "bootstrap_significance_table.md"),
                      "Bootstrap Paired Significance Tests (N=10,000, majority vote ensemble)")
    else:
        print("  [WARN] Could not parse bootstrap_significance.json — check format manually")

    # --- Combined summary markdown ---
    print("\n=== Combined Summary ===")
    summary_path = os.path.join(OUTPUT_DIR, "all_results_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Multimodal Deepfake Detection — Full Results Summary\n\n")
        f.write("**Best config:** 1-layer transformer, cross-modal fusion\n")
        f.write("**Ensemble BAcc:** 0.7096 | **Ensemble AUC:** 0.9035\n\n")
        f.write("---\n\n")

        # Depth table
        f.write("## Table 1: Depth Ablation (fusion_mode = cross_modal)\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in depth_rows:
            vals = [str(row.get(c, "")) for c in cols]
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")

        # Fusion table
        f.write("## Table 2: Fusion Ablation (depth = 1 layer)\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in fusion_rows:
            vals = [str(row.get(c, "")) for c in cols]
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")

        # Bootstrap table
        if boot_rows:
            f.write("## Table 3: Bootstrap Significance Tests\n\n")
            f.write("Paired bootstrap (N=10,000) on majority-voted ensemble predictions.\n\n")
            f.write("| " + " | ".join(boot_cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(boot_cols)) + " |\n")
            for row in boot_rows:
                vals = [str(row.get(c, "")) for c in boot_cols]
                f.write("| " + " | ".join(vals) + " |\n")
            f.write("\n")

    print(f"  [MD]  {summary_path}")
    print(f"\nAll tables saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()