"""Compact R(2+1)D-style 3D CNN from scratch for HAR (<100MB)."""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv2Plus1D(nn.Module):
    def __init__(self, cin, cout, spatial_k=3, temporal_k=3, stride=(1, 1, 1), padding=None):
        super().__init__()
        st, sh, sw = stride if isinstance(stride, tuple) else (stride, stride, stride)
        if padding is None:
            pt, ph, pw = temporal_k // 2, spatial_k // 2, spatial_k // 2
        else:
            pt, ph, pw = padding
        mid = max(cout // 2, cin // 2, 16)
        self.spatial = nn.Sequential(
            nn.Conv3d(cin, mid, kernel_size=(1, spatial_k, spatial_k), stride=(1, sh, sw), padding=(0, ph, pw), bias=False),
            nn.BatchNorm3d(mid),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(mid, cout, kernel_size=(temporal_k, 1, 1), stride=(st, 1, 1), padding=(pt, 0, 0), bias=False),
            nn.BatchNorm3d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.temporal(self.spatial(x))


class CompactR2Plus1D(nn.Module):
    """Small R(2+1)D: input (B,C,T,H,W) -> logits. ~5-15MB."""

    def __init__(self, num_classes: int = 40, in_ch: int = 3, base: int = 32, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(base),
            nn.SiLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            Conv2Plus1D(base, base, stride=(1, 1, 1)),
            Conv2Plus1D(base, base * 2, stride=(1, 2, 2)),
        )
        self.layer2 = nn.Sequential(
            Conv2Plus1D(base * 2, base * 2, stride=(1, 1, 1)),
            Conv2Plus1D(base * 2, base * 4, stride=(2, 2, 2)),
        )
        self.layer3 = nn.Sequential(
            Conv2Plus1D(base * 4, base * 4, stride=(1, 1, 1)),
            Conv2Plus1D(base * 4, base * 8, stride=(2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(base * 8, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B,T,C,H,W) or (B,C,T,H,W)
        if x.dim() == 5 and x.shape[2] in (1, 3, 6) and x.shape[1] > 4:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.head(self.drop(x))


def build_model_r2p1d(num_classes: int = 40, in_ch: int = 3, base: int = 32) -> CompactR2Plus1D:
    return CompactR2Plus1D(num_classes=num_classes, in_ch=in_ch, base=base)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    n = sum(p.detach().cpu().nbytes for p in model.parameters())
    n += sum(b.detach().cpu().nbytes for b in model.buffers())
    return n / (1024 * 1024)


if __name__ == "__main__":
    m = build_model_r2p1d()
    x = torch.randn(2, 16, 3, 112, 112)
    y = m(x)
    print(y.shape, count_parameters(m), round(model_size_mb(m), 2))
