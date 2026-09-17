"""Small from-scratch temporal CNN for Depth_Color / Thermal HAR (<100MB)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FrameEncoder(nn.Module):
    """Tiny CNN: 112x112 RGB -> 256-d embedding. ~1.2M params."""

    def __init__(self, in_ch: int = 3, emb: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, 32, s=2),   # 56
            ConvBNAct(32, 32),
            ConvBNAct(32, 64, s=2),      # 28
            ConvBNAct(64, 64),
            ConvBNAct(64, 128, s=2),     # 14
            ConvBNAct(128, 128),
            ConvBNAct(128, 256, s=2),    # 7
            ConvBNAct(256, emb),
            nn.AdaptiveAvgPool2d(1),
        )
        self.emb = emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        return self.net(x).flatten(1)


class TemporalCNNHAR(nn.Module):
    """2D CNN per-frame + bidirectional GRU temporal head."""

    def __init__(
        self,
        num_classes: int = 40,
        in_ch: int = 3,
        emb: int = 256,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = FrameEncoder(in_ch=in_ch, emb=emb)
        self.gru = nn.GRU(
            emb,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if gru_layers == 1 else dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        feats = self.encoder(x.reshape(b * t, c, h, w)).view(b, t, -1)
        out, _ = self.gru(feats)
        # attention-ish: mean of last 2 steps + global mean
        pooled = 0.5 * (out.mean(dim=1) + out[:, -1])
        return self.head(self.drop(pooled))


def build_model(num_classes: int = 40, in_ch: int = 3) -> TemporalCNNHAR:
    return TemporalCNNHAR(num_classes=num_classes, in_ch=in_ch)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    buf = []
    for p in model.parameters():
        buf.append(p.detach().cpu().nbytes)
    for b in model.buffers():
        buf.append(b.detach().cpu().nbytes)
    return sum(buf) / (1024 * 1024)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(2, 8, 3, 112, 112)
    y = m(x)
    print("out", y.shape, "params", count_parameters(m), "size_mb", round(model_size_mb(m), 2))
