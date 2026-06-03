"""
model.py — Multimodal Deepfake Detection Architecture
======================================================
Configurable for ablation studies:
  - n_encoder_layers: per-branch transformer depth (1, 2, 3, 4)
  - n_cross_layers: cross-modal transformer depth
  - fusion_mode: "cross_modal" | "concat" | "video_only" | "audio_only"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import D_MODEL, N_HEADS, DIM_FEEDFORWARD, DROPOUT


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class VGGish(nn.Module):
    """VGGish CNN: (B,1,128,T) → (B,T',128). T' = T//16 after 4x MaxPool."""
    PRETRAINED_URL = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"

    def __init__(self, pretrained=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.temporal_proj = nn.Linear(512, 128)
        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        try:
            state = torch.hub.load_state_dict_from_url(self.PRETRAINED_URL, progress=True, map_location="cpu")
            self.load_state_dict(state, strict=False)
            print(f"[VGGish] Loaded pretrained weights")
        except Exception as e:
            print(f"[VGGish] Pretrained load failed: {e}. Training from scratch.")

    def forward(self, x):
        x = self.features(x)       # (B, 512, H', W')
        x = x.mean(dim=2)          # pool over frequency → (B, 512, T')
        x = x.permute(0, 2, 1)     # (B, T', 512)
        return self.temporal_proj(x)  # (B, T', 128)


class VideoEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dim_ff=DIM_FEEDFORWARD,
                 n_layers=2, dropout=DROPOUT):
        super().__init__()
        enet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone = enet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.backbone(x.reshape(B * T, C, H, W))
        x = self.pool(x).flatten(1).view(B, T, -1)
        return self.transformer(self.pos_enc(self.proj(x)))


class AudioEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dim_ff=DIM_FEEDFORWARD,
                 n_layers=2, dropout=DROPOUT, pretrained=True):
        super().__init__()
        self.vggish = VGGish(pretrained=pretrained)
        self.proj = nn.Linear(128, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        return self.transformer(self.pos_enc(self.proj(self.vggish(x))))


class CrossModalAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.v2a_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.v2a_norm1 = nn.LayerNorm(d_model)
        self.v2a_ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.v2a_norm2 = nn.LayerNorm(d_model)
        self.a2v_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.a2v_norm1 = nn.LayerNorm(d_model)
        self.a2v_ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.a2v_norm2 = nn.LayerNorm(d_model)

    def forward(self, video, audio):
        v_out, v2a_w = self.v2a_attn(video, audio, audio)
        video = self.v2a_norm2(self.v2a_norm1(video + v_out) + self.v2a_ffn(self.v2a_norm1(video + v_out)))
        a_out, a2v_w = self.a2v_attn(audio, video, video)
        audio = self.a2v_norm2(self.a2v_norm1(audio + a_out) + self.a2v_ffn(self.a2v_norm1(audio + a_out)))
        return video, audio, v2a_w, a2v_w


class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dim_ff=DIM_FEEDFORWARD,
                 n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])
        self.attn_weights = []

    def forward(self, video, audio):
        self.attn_weights = []
        for layer in self.layers:
            video, audio, v2a_w, a2v_w = layer(video, audio)
            self.attn_weights.append({"v2a": v2a_w.detach(), "a2v": a2v_w.detach()})
        return video, audio


class MultimodalDeepfakeDetector(nn.Module):
    """
    fusion_mode: "cross_modal" | "concat" | "video_only" | "audio_only"
    """
    def __init__(self, num_classes=4, d_model=D_MODEL,
                 n_encoder_layers=2, n_cross_layers=2, fusion_mode="cross_modal"):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.d_model = d_model

        if fusion_mode != "audio_only":
            self.video_encoder = VideoEncoder(d_model=d_model, n_layers=n_encoder_layers)
        if fusion_mode != "video_only":
            self.audio_encoder = AudioEncoder(d_model=d_model, n_layers=n_encoder_layers)
        if fusion_mode == "cross_modal":
            self.cross_modal = CrossModalTransformer(d_model=d_model, n_layers=n_cross_layers)

        cls_in = d_model if fusion_mode in ("video_only", "audio_only") else d_model * 2
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, d_model), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, video, audio):
        if self.fusion_mode == "video_only":
            return self.classifier(self.video_encoder(video).mean(1))
        if self.fusion_mode == "audio_only":
            return self.classifier(self.audio_encoder(audio).mean(1))

        v, a = self.video_encoder(video), self.audio_encoder(audio)
        if self.fusion_mode == "cross_modal":
            v, a = self.cross_modal(v, a)
        return self.classifier(torch.cat([v.mean(1), a.mean(1)], dim=1))

    def get_param_groups(self, lr_pretrained, lr_new):
        pretrained, new = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "video_encoder.backbone" in name or "audio_encoder.vggish.features" in name:
                pretrained.append(p)
            else:
                new.append(p)
        return [{"params": pretrained, "lr": lr_pretrained},
                {"params": new, "lr": lr_new}]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    for mode in ["cross_modal", "concat", "video_only", "audio_only"]:
        m = MultimodalDeepfakeDetector(4, n_encoder_layers=2, n_cross_layers=2, fusion_mode=mode)
        c = count_parameters(m)
        print(f"{mode:15s} | total: {c['total']:>10,} | trainable: {c['trainable']:>10,}")
