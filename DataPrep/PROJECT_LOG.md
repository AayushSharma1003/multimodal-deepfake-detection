## [2026-06-05] Phase 2 Complete — All Experiments & Analysis Done

### Status
- **Phase 1 (Preprocessing):** ✅ Complete
- **Phase 2 (Training + Ablations + Evaluation + Analysis):** ✅ Complete
- **Phase 3 (Paper Writing + DFDC Cross-Dataset):** 🔲 Starting

### Experiments Completed

**Experiment 1 — Depth Ablation (12 runs)**
Varied transformer depth (1–4 layers), fusion_mode=cross_modal, 3 seeds each (7, 42, 123).

| Layers | Bal.Acc (mean±std) | AUC (mean±std) | ECE | Ensemble BAcc | Ensemble AUC |
|--------|--------------------|----------------|------|---------------|--------------|
| 1 | 0.6998±0.0022 | 0.8941±0.0026 | 0.1539 | **0.7096** | **0.9035** |
| 2 | 0.6661±0.0194 | 0.8711±0.0114 | 0.2043 | 0.6804 | 0.8847 |
| 3 | 0.5995±0.0626 | 0.8302±0.0406 | 0.1199 | 0.6538 | 0.8691 |
| 4 | 0.5369±0.0407 | 0.7852±0.0244 | 0.0401 | 0.5551 | 0.8183 |

**Finding:** Performance degrades monotonically with depth. 1-layer is optimal. Deeper models overfit on 20K samples — variance increases from 0.0022 (1L) to 0.0407 (4L).

**Experiment 2 — Fusion Ablation (12 runs)**
Used best depth (1 layer), compared 4 fusion strategies, 3 seeds each.

| Mode | Bal.Acc (mean±std) | AUC (mean±std) | ECE | Ensemble BAcc | Ensemble AUC |
|------|--------------------|----------------|------|---------------|--------------|
| Video Only | 0.6742±0.0005 | 0.8802±0.0011 | 0.1499 | 0.6846 | 0.8887 |
| Audio Only | 0.3201±0.0066 | 0.5873±0.0009 | 0.0374 | 0.3200 | 0.5887 |
| Concat | 0.6878±0.0038 | 0.8879±0.0012 | 0.1756 | 0.6982 | 0.8966 |
| Cross Modal | 0.6998±0.0022 | 0.8941±0.0026 | 0.1539 | **0.7096** | **0.9035** |

**Finding:** Cross-modal significantly beats all alternatives (bootstrap p<0.05). Audio alone is near-random (~32% on 4-class) but contributes meaningfully through cross-modal fusion (+2.5% over video-only).

**Experiment 3 — Full Evaluation (complete)**
Best model config (1-layer cross-modal): test metrics, modality occlusion, Grad-CAM++, cross-modal attention visualization, GradientSHAP attribution maps.

**Experiment 4 — Analysis (complete)**
All paper figures (F2–F7) generated. Bootstrap significance tests (paired, majority vote ensemble, N=10,000) completed.

### Bootstrap Significance Tests (majority vote ensemble)

**Depth comparisons (is deeper better?):**
- 2L vs 1L: Δ=-0.0293, p=1.000, CI=[-0.0417, -0.0167] — NOT significant (deeper is worse)
- 3L vs 2L: Δ=-0.0266, p=1.000, CI=[-0.0395, -0.0134] — NOT significant (deeper is worse)
- 4L vs 3L: Δ=-0.0987, p=1.000, CI=[-0.1147, -0.0834] — NOT significant (deeper is worse)

**Fusion comparisons (is cross_modal better?):**
- cross_modal vs concat: Δ=+0.0115, p=0.0247, CI=[0.0000, 0.0230] — ✅ SIGNIFICANT
- cross_modal vs video_only: Δ=+0.0251, p=0.000, CI=[0.0130, 0.0372] — ✅ SIGNIFICANT
- cross_modal vs audio_only: Δ=+0.3896, p=0.000, CI=[0.3734, 0.4060] — ✅ SIGNIFICANT

### Best Model Summary
- **Config:** 1-layer transformer, cross_modal fusion, EfficientNet-B0 + VGGish
- **Parameters:** 13,283,072 (at 2-layer; 1-layer is fewer)
- **Test Balanced Accuracy:** 70.96% (majority vote ensemble)
- **Test AUC-ROC:** 0.9035 (majority vote ensemble)

### Generated Artifacts
**Plots (in `plots/`):**
- F2_training_curves.png — training/validation loss and AUC curves
- F3_depth_ablation.png — depth ablation bar chart
- F4_fusion_ablation.png — fusion ablation bar chart
- F5_confusion_matrix.png — 4-class confusion matrix (majority vote)
- F6a_per_class_f1_depth.png — per-class F1 across depths
- F6b_per_class_f1_fusion.png — per-class F1 across fusion modes
- F7_modality_occlusion.png — modality occlusion study results

**Results (in `results/`):**
- `depth_ablation/layers_{1,2,3,4}_seed_{7,42,123}/` — model checkpoints, history, test metrics, predictions
- `fusion_ablation/{video_only,audio_only,concat,cross_modal}_seed_{7,42,123}/` — same structure
- `evaluation/` — full evaluation outputs (Grad-CAM++, SHAP, attention weights, occlusion)
- `bootstrap_significance.json` — all p-values and CIs

### Decisions
- **Best depth = 1 layer:** Counter-intuitive but clean result. Pretrained backbones already provide strong features; minimal fusion head avoids overfitting on 20K samples.
- **Cross-modal attention validated:** Statistically significant improvement over concat (p=0.0247) confirms that learned cross-modal correspondences outperform naive feature aggregation.

### Next Steps
- [ ] Update GitHub repo with all results, plots, and code; remove stale 5K preprocessing files
- [ ] Interpret plots and confusion matrix in detail
- [ ] Cross-dataset evaluation on DFDC (binary collapse at inference)
- [ ] Paper writing (venue TBD — IEEE or ACL format)
- [ ] Error analysis on failure cases
- [ ] Comparison table with published baselines on LAV-DF