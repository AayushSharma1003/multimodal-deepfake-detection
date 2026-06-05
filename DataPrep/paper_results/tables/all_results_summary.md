# Multimodal Deepfake Detection — Full Results Summary

**Best config:** 1-layer transformer, cross-modal fusion
**Ensemble BAcc:** 0.7096 | **Ensemble AUC:** 0.9035

---

## Table 1: Depth Ablation (fusion_mode = cross_modal)

| Config | Bal.Acc (mean±std) | AUC (mean±std) | ECE (mean) |
| --- | --- | --- | --- |
| 1 layer | 0.6998±0.0022 | 0.8941±0.0026 | 0.1539 |
| 2 layers | 0.6661±0.0194 | 0.8717±0.0114 | 0.2043 |
| 3 layers | 0.5995±0.0626 | 0.8302±0.0406 | 0.1199 |
| 4 layers | 0.5369±0.0407 | 0.7852±0.0244 | 0.0401 |

## Table 2: Fusion Ablation (depth = 1 layer)

| Config | Bal.Acc (mean±std) | AUC (mean±std) | ECE (mean) |
| --- | --- | --- | --- |
| Video Only | 0.6742±0.0005 | 0.8802±0.0011 | 0.1499 |
| Audio Only | 0.3201±0.0066 | 0.5873±0.0009 | 0.0374 |
| Concat | 0.6878±0.0038 | 0.8870±0.0012 | 0.1756 |
| Cross Modal | 0.6998±0.0022 | 0.8941±0.0026 | 0.1539 |

## Table 3: Bootstrap Significance Tests

Paired bootstrap (N=10,000) on majority-voted ensemble predictions.

| Comparison | Delta (BAcc) | p-value | 95% CI | Significant |
| --- | --- | --- | --- | --- |
| depth_1_vs_2 | +0.0293 | 0.0000 | [0.0167, 0.0417] | Yes |
| depth_2_vs_3 | +0.0266 | 0.0000 | [0.0134, 0.0395] | Yes |
| depth_3_vs_4 | +0.0987 | 0.0000 | [0.0834, 0.1147] | Yes |
| cross_modal_vs_concat | +0.0115 | 0.0247 | [0.0000, 0.0230] | Yes |
| cross_modal_vs_video_only | +0.0251 | 0.0000 | [0.0130, 0.0372] | Yes |
| cross_modal_vs_audio_only | +0.3896 | 0.0000 | [0.3734, 0.4060] | Yes |

