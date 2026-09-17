"""Tiny temporal CNN for Depth_Color + IR (no ImageNet backbone). Target <<10MB."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTempCNN(nn.Module):
    """Per-frame Conv2d stem -> temporal Conv1d -> logits.

    Input: (B, C, T, H, W) with C=2 (depth gray + IR), T=K frames.
    """

    def __init__(self, in_ch: int = 2, num_classes: int = 40, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, stride=2, padding=1),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(w * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(w * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        feat = w * 4
        self.temporal = nn.Sequential(
            nn.Conv1d(feat, feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat, feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(feat),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        f = self.stem(x).flatten(1)  # (B*T, feat)
        f = f.view(b, t, -1).transpose(1, 2)  # (B, feat, T)
        f = self.temporal(f).flatten(1)
        return self.head(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ckpt_mb(model: nn.Module) -> float:
    # rough fp32
    return count_parameters(model) * 4 / (1024 * 1024)


if __name__ == "__main__":
    m = TinyTempCNN()
    x = torch.randn(2, 2, 8, 48, 64)
    y = m(x)
    print(y.shape, count_parameters(m), f"{ckpt_mb(m):.2f} MB")
