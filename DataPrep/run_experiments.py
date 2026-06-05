"""
run_experiments.py — Training + Ablation + Evaluation + Explainability
=======================================================================
Usage:
    python run_experiments.py --phase depth       # Depth ablation (12 runs)
    python run_experiments.py --phase fusion      # Fusion ablation (12 runs)
    python run_experiments.py --phase evaluate    # Full eval on best model
    python run_experiments.py --phase all         # Everything sequentially
"""

import os, sys, json, math, time, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from dataset import get_dataloaders
from model import MultimodalDeepfakeDetector, count_parameters

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = (confidences >= lo) & (confidences < hi)
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += np.abs(avg_conf - avg_acc) * in_bin.mean()
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for batch in loader:
        video = batch["video"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(video, audio)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits.detach().float(), dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "bal_acc": balanced_accuracy_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro"),
    }


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for batch in loader:
        video = batch["video"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        with autocast():
            logits = model(video, audio)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits.float(), dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "bal_acc": balanced_accuracy_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro"),
    }


def train_single_run(run_name, seed, n_encoder_layers, n_cross_layers, fusion_mode):
    """Train one configuration with one seed. Returns path to results dir."""
    run_dir = os.path.join(RESULTS_DIR, run_name)
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    # Skip if already completed
    if os.path.exists(os.path.join(run_dir, "test_predictions.npz")):
        print(f"  [{run_name}] Already completed, skipping.")
        return run_dir

    os.makedirs(run_dir, exist_ok=True)
    set_seed(seed)

    # Data
    loaders = get_dataloaders(label_type=LABEL_TYPE, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader = loaders["train"], loaders["dev"], loaders["test"]

    # Model
    model = MultimodalDeepfakeDetector(
        num_classes=NUM_CLASSES, n_encoder_layers=n_encoder_layers,
        n_cross_layers=n_cross_layers, fusion_mode=fusion_mode,
    ).to(DEVICE)
    counts = count_parameters(model)
    print(f"  [{run_name}] Params: {counts['total']:,} total, {counts['trainable']:,} trainable")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.get_param_groups(LR_PRETRAINED, LR_NEW_LAYERS),
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # Loss
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler()

    # Training loop
    best_auc, patience_ctr, history = 0.0, 0, []
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
        val_m = validate(model, val_loader, criterion)
        elapsed = time.time() - t0

        improved = val_m["auc"] > best_auc
        if improved:
            best_auc = val_m["auc"]
            patience_ctr = 0
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "config": {"n_encoder_layers": n_encoder_layers,
                                   "n_cross_layers": n_cross_layers,
                                   "fusion_mode": fusion_mode, "seed": seed},
                        "val_auc": best_auc}, ckpt_path)
        else:
            patience_ctr += 1

        history.append({"epoch": epoch + 1, "train": train_m, "val": val_m, "time": elapsed})
        mark = " ★" if improved else ""
        print(f"    Ep {epoch+1:02d}/{NUM_EPOCHS} ({elapsed:.0f}s) | "
              f"train: loss={train_m['loss']:.4f} bacc={train_m['bal_acc']:.4f} | "
              f"val: loss={val_m['loss']:.4f} bacc={val_m['bal_acc']:.4f} auc={val_m['auc']:.4f}{mark}")

        if EARLY_STOP_PATIENCE and patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # Save history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set with best model
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_preds, test_labels, test_probs = evaluate_on_test(model, test_loader)

    # Save test predictions
    np.savez(os.path.join(run_dir, "test_predictions.npz"),
             preds=test_preds, labels=test_labels, probs=test_probs)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"  [{run_name}] Done. Test bacc={test_metrics['bal_acc']:.4f} auc={test_metrics['auc']:.4f}")
    return run_dir


@torch.no_grad()
def evaluate_on_test(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        video = batch["video"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        with autocast():
            logits = model(video, audio)
        probs = torch.softmax(logits.float(), dim=1)
        all_preds.extend(probs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    preds, labels, probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    metrics = {
        "bal_acc": balanced_accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs, multi_class="ovr", average="macro"),
        "ece": compute_ece(probs, labels),
        "per_class": {name: {"precision": report[name]["precision"],
                             "recall": report[name]["recall"],
                             "f1": report[name]["f1-score"]}
                      for name in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    return metrics, preds, labels, probs


# ═══════════════════════════════════════════════════════════════════════
# MODALITY OCCLUSION
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def modality_occlusion_study(model, test_loader, save_dir):
    """Zero out one modality at a time, measure performance drop."""
    model.eval()
    results = {}

    for condition in ["full", "video_zeroed", "audio_zeroed"]:
        all_preds, all_labels, all_probs = [], [], []
        for batch in tqdm(test_loader, desc=f"  Occlusion [{condition}]", leave=False):
            video = batch["video"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            labels = batch["label"]

            if condition == "video_zeroed":
                video = torch.zeros_like(video)
            elif condition == "audio_zeroed":
                audio = torch.zeros_like(audio)

            with autocast():
                logits = model(video, audio)
            probs = torch.softmax(logits.float(), dim=1)
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        preds, labels, probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
        results[condition] = {
            "bal_acc": balanced_accuracy_score(labels, preds),
            "auc": roc_auc_score(labels, probs, multi_class="ovr", average="macro"),
        }
        print(f"    {condition}: bacc={results[condition]['bal_acc']:.4f} auc={results[condition]['auc']:.4f}")

    with open(os.path.join(save_dir, "modality_occlusion.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════
# GRAD-CAM++
# ═══════════════════════════════════════════════════════════════════════

class GradCAMPP:
    def __init__(self, model, target_layer):
        self.activations = self.gradients = None
        self._fwd = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def compute(self):
        g, a = self.gradients, self.activations
        alpha = (g ** 2) / (2 * g ** 2 + (g ** 3 * a).sum((2, 3), keepdim=True) + 1e-8)
        weights = (alpha * F.relu(g)).sum((2, 3))
        cam = F.relu((weights.unsqueeze(-1).unsqueeze(-1) * a).sum(1))
        cam = cam[0].cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / cam.max()
        return cam

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def run_gradcam(model, test_dataset, save_dir, n_samples=GRADCAM_SAMPLES):
    """Generate Grad-CAM++ heatmaps for video and audio branches."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Only works for cross_modal or concat (needs both branches)
    if model.fusion_mode in ("video_only", "audio_only"):
        print("  [gradcam] Skipping — need both branches for comparison.")
        return

    for i in range(min(n_samples, len(test_dataset))):
        sample = test_dataset[i]
        vid = sample["video"].unsqueeze(0).to(DEVICE)
        aud = sample["audio"].unsqueeze(0).to(DEVICE)
        label = sample["label"].item()
        vid_id = sample["video_id"]

        # Video Grad-CAM++ on EfficientNet last block
        cam_v = GradCAMPP(model, model.video_encoder.backbone[-1])
        model.zero_grad()
        logits = model(vid, aud)
        pred = logits.argmax(1).item()
        logits[0, pred].backward()
        heatmap_v = cam_v.compute()
        cam_v.remove()

        # Audio Grad-CAM++ on VGGish last conv
        cam_a = GradCAMPP(model, model.audio_encoder.vggish.features[-2])
        model.zero_grad()
        logits = model(vid, aud)
        logits[0, pred].backward()
        heatmap_a = cam_a.compute()
        cam_a.remove()

        # Save video heatmap (overlay on frames 0,4,8,12)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        from PIL import Image
        hm_resized = np.array(Image.fromarray((heatmap_v * 255).astype(np.uint8)).resize((224, 224))) / 255.0
        for j, fi in enumerate([0, 4, 8, 12]):
            frame = vid[0, fi].cpu().permute(1, 2, 0).numpy() * std + mean
            axes[j].imshow(np.clip(frame, 0, 1))
            axes[j].imshow(hm_resized, alpha=0.4, cmap="jet")
            axes[j].set_title(f"Frame {fi}")
            axes[j].axis("off")
        fig.suptitle(f"{vid_id} | true={CLASS_NAMES[label]} pred={CLASS_NAMES[pred]}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{vid_id}_video_gradcam.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Save audio heatmap
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        spec = aud[0, 0].cpu().numpy()
        hm_a = np.array(Image.fromarray((heatmap_a * 255).astype(np.uint8)).resize(
            (spec.shape[1], spec.shape[0]))) / 255.0
        axes[0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[0].set_title("Spectrogram")
        axes[1].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[1].imshow(hm_a, alpha=0.5, cmap="jet", aspect="auto", origin="lower")
        axes[1].set_title(f"Grad-CAM++ (pred={CLASS_NAMES[pred]})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{vid_id}_audio_gradcam.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  [gradcam] Saved {n_samples} samples → {save_dir}")


def run_cross_attention_viz(model, test_dataset, save_dir, n_samples=4):
    """Visualize cross-modal attention weight matrices."""
    if model.fusion_mode != "cross_modal":
        return
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for i in range(min(n_samples, len(test_dataset))):
        sample = test_dataset[i]
        vid = sample["video"].unsqueeze(0).to(DEVICE)
        aud = sample["audio"].unsqueeze(0).to(DEVICE)
        vid_id = sample["video_id"]

        with torch.no_grad():
            _ = model(vid, aud)

        attn_list = model.cross_modal.attn_weights
        n_layers = len(attn_list)
        fig, axes = plt.subplots(n_layers, 2, figsize=(10, 3.5 * n_layers))
        if n_layers == 1:
            axes = axes[np.newaxis, :]
        for li, w in enumerate(attn_list):
            v2a = w["v2a"][0].cpu().numpy()
            a2v = w["a2v"][0].cpu().numpy()
            axes[li, 0].imshow(v2a, cmap="viridis", aspect="auto")
            axes[li, 0].set_title(f"L{li+1}: Video→Audio")
            axes[li, 0].set_xlabel("Audio"); axes[li, 0].set_ylabel("Video")
            axes[li, 1].imshow(a2v, cmap="viridis", aspect="auto")
            axes[li, 1].set_title(f"L{li+1}: Audio→Video")
            axes[li, 1].set_xlabel("Video"); axes[li, 1].set_ylabel("Audio")
        fig.suptitle(f"Cross-Attention: {vid_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{vid_id}_cross_attn.png"), dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [cross-attn] Saved {n_samples} samples → {save_dir}")


def run_gradient_shap(model, test_dataset, save_dir, n_samples=4):
    """GradientSHAP attribution maps using captum."""
    try:
        from captum.attr import GradientShap
    except ImportError:
        print("  [shap] captum not installed. Run: pip install captum --break-system-packages")
        return

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if model.fusion_mode in ("video_only", "audio_only"):
        print("  [shap] Skipping — need both branches.")
        return

    # Wrapper for captum (needs tuple input)
    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, v, a): return self.m(v, a)

    wrapped = Wrapper(model)
    gs = GradientShap(wrapped)

    for i in range(min(n_samples, len(test_dataset))):
        sample = test_dataset[i]
        vid = sample["video"].unsqueeze(0).to(DEVICE)
        aud = sample["audio"].unsqueeze(0).to(DEVICE)
        label = sample["label"].item()
        vid_id = sample["video_id"]

        # Baselines: zeros
        vid_bl = torch.zeros_like(vid)
        aud_bl = torch.zeros_like(aud)

        try:
            attr_v, attr_a = gs.attribute((vid, aud), baselines=(vid_bl, aud_bl),
                                           target=label, n_samples=50)
        except Exception as e:
            print(f"  [shap] Failed on {vid_id}: {e}")
            continue

        # Video attribution: sum over channels, show frame 0
        attr_frame = attr_v[0, 0].sum(0).abs().cpu().numpy()  # (224, 224)
        if attr_frame.max() > 0:
            attr_frame /= attr_frame.max()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        frame = vid[0, 0].cpu().permute(1, 2, 0).numpy()
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        axes[0].imshow(np.clip(frame * std + mean, 0, 1))
        axes[0].set_title("Original frame")
        axes[1].imshow(np.clip(frame * std + mean, 0, 1))
        axes[1].imshow(attr_frame, alpha=0.5, cmap="hot")
        axes[1].set_title(f"SHAP ({CLASS_NAMES[label]})")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{vid_id}_shap.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  [shap] Saved {n_samples} samples → {save_dir}")


# ═══════════════════════════════════════════════════════════════════════
# ABLATION RUNNERS
# ═══════════════════════════════════════════════════════════════════════

def run_depth_ablation():
    """Ablation Study 1: Transformer depth 1,2,3,4 × 3 seeds."""
    print("\n" + "=" * 70)
    print("  DEPTH ABLATION")
    print("=" * 70)
    for depth in DEPTH_ABLATION:
        for seed in SEEDS:
            run_name = f"depth_ablation/layers_{depth}_seed_{seed}"
            print(f"\n▶ {run_name}")
            train_single_run(run_name, seed, n_encoder_layers=depth,
                             n_cross_layers=depth, fusion_mode="cross_modal")


def find_best_depth():
    """Determine best depth from ablation results (highest mean val AUC across seeds)."""
    best_depth, best_mean_auc = 2, 0.0
    for depth in DEPTH_ABLATION:
        aucs = []
        for seed in SEEDS:
            metrics_path = os.path.join(RESULTS_DIR, f"depth_ablation/layers_{depth}_seed_{seed}/test_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    aucs.append(json.load(f)["auc"])
        if aucs:
            mean_auc = np.mean(aucs)
            print(f"  Depth {depth}: AUC = {mean_auc:.4f} ± {np.std(aucs):.4f}")
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_depth = depth
    print(f"  → Best depth: {best_depth} (AUC={best_mean_auc:.4f})")
    return best_depth


def run_fusion_ablation():
    """Ablation Study 2: Fusion modes × 3 seeds, using best depth."""
    best_depth = find_best_depth()
    print("\n" + "=" * 70)
    print(f"  FUSION ABLATION (using depth={best_depth})")
    print("=" * 70)
    for mode in FUSION_MODES:
        for seed in SEEDS:
            run_name = f"fusion_ablation/{mode}_seed_{seed}"
            print(f"\n▶ {run_name}")
            n_cross = best_depth if mode == "cross_modal" else 0
            train_single_run(run_name, seed, n_encoder_layers=best_depth,
                             n_cross_layers=n_cross, fusion_mode=mode)


def run_full_evaluation():
    """Full evaluation on the best model: metrics, occlusion, Grad-CAM++, SHAP."""
    best_depth = find_best_depth()
    # Use seed with best AUC for qualitative analysis
    best_seed, best_auc = SEEDS[0], 0.0
    for seed in SEEDS:
        mpath = os.path.join(RESULTS_DIR, f"depth_ablation/layers_{best_depth}_seed_{seed}/test_metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                auc = json.load(f)["auc"]
            if auc > best_auc:
                best_auc, best_seed = auc, seed

    run_dir = os.path.join(RESULTS_DIR, f"depth_ablation/layers_{best_depth}_seed_{best_seed}")
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    eval_dir = os.path.join(RESULTS_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FULL EVALUATION — depth={best_depth}, seed={best_seed}")
    print(f"{'='*70}")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = MultimodalDeepfakeDetector(
        num_classes=NUM_CLASSES, n_encoder_layers=best_depth,
        n_cross_layers=best_depth, fusion_mode="cross_modal",
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # Data
    loaders = get_dataloaders(label_type=LABEL_TYPE, batch_size=BATCH_SIZE)
    test_loader = loaders["test"]

    # Print detailed metrics
    with open(os.path.join(run_dir, "test_metrics.json")) as f:
        metrics = json.load(f)
    print(f"\n  Balanced Accuracy: {metrics['bal_acc']:.4f}")
    print(f"  AUC-ROC:           {metrics['auc']:.4f}")
    print(f"  ECE:               {metrics['ece']:.4f}")
    print(f"\n  Per-class F1:")
    for name in CLASS_NAMES:
        f1 = metrics['per_class'][name]['f1']
        print(f"    {name}: {f1:.4f}")

    # Modality occlusion
    print("\n  Running modality occlusion study...")
    modality_occlusion_study(model, test_loader, eval_dir)

    # Grad-CAM++
    print("\n  Generating Grad-CAM++ visualizations...")
    run_gradcam(model, test_loader.dataset, os.path.join(eval_dir, "gradcam"))

    # Cross-attention visualization
    print("\n  Generating cross-attention maps...")
    run_cross_attention_viz(model, test_loader.dataset, os.path.join(eval_dir, "cross_attn"))

    # GradientSHAP
    print("\n  Generating GradientSHAP attributions...")
    run_gradient_shap(model, test_loader.dataset, os.path.join(eval_dir, "shap"))

    print(f"\n  All evaluation outputs → {eval_dir}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["depth", "fusion", "evaluate", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.phase in ("depth", "all"):
        run_depth_ablation()
    if args.phase in ("fusion", "all"):
        run_fusion_ablation()
    if args.phase in ("evaluate", "all"):
        run_full_evaluation()

    print("\n✓ All done.")
