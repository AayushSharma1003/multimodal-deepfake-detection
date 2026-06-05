# Multimodal Deepfake Detection — Full Results Summary

**Best config:** 1-layer transformer, cross-modal fusion
**Ensemble BAcc:** 0.7096 | **Ensemble AUC:** 0.9035

---

## Table 1: Depth Ablation (fusion_mode = cross_modal)

| Config | Bal.Acc (mean±std) | AUC (mean±std) | ECE (mean) |
| --- | --- | --- | --- |
| 1 layer | 0.0000±0.0000 | 0.0000±0.0000 | 0.1539 |
| 2 layers | 0.0000±0.0000 | 0.0000±0.0000 | 0.2043 |
| 3 layers | 0.0000±0.0000 | 0.0000±0.0000 | 0.1199 |
| 4 layers | 0.0000±0.0000 | 0.0000±0.0000 | 0.0401 |

## Table 2: Fusion Ablation (depth = 1 layer)

| Config | Bal.Acc (mean±std) | AUC (mean±std) | ECE (mean) |
| --- | --- | --- | --- |
| Video Only | 0.0000±0.0000 | 0.0000±0.0000 | 0.1499 |
| Audio Only | 0.0000±0.0000 | 0.0000±0.0000 | 0.0374 |
| Concat | 0.0000±0.0000 | 0.0000±0.0000 | 0.1756 |
| Cross Modal | 0.0000±0.0000 | 0.0000±0.0000 | 0.1539 |

## Table 3: Bootstrap Significance Tests

Paired bootstrap (N=10,000) on majority-voted ensemble predictions.

| Comparison | Delta (BAcc) | p-value | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| depth_2_vs_1 | -0.0293 | 1.0000 | [-0.0417, -0.0167] | No |
| depth_3_vs_2 | -0.0266 | 1.0000 | [-0.0395, -0.0134] | No |
| depth_4_vs_3 | -0.0987 | 1.0000 | [-0.1147, -0.0834] | No |
| cross_modal_vs_concat | +0.0115 | 0.0247 | [0.0000, 0.0230] | No |
| cross_modal_vs_video_only | +0.0251 | 0.0000 | [0.0130, 0.0372] | No |
| cross_modal_vs_audio_only | +0.3896 | 0.0000 | [0.3734, 0.4060] | No |

