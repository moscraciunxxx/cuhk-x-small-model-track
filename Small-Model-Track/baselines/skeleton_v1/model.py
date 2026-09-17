"""Small skeleton sequence classifiers for CUHK-X (CNN / GRU / Transformer).

Target: few MB on disk, no large pretrained weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dataset import FEAT_DIM, NUM_JOINTS, COORDS


class Conv1DClassifier(nn.Module):
    """Temporal Conv1D over (B, T, F) sequences."""

    def __init__(
        self,
        num_classes: int = 40,
        feat_dim: int = FEAT_DIM,
        channels: tuple = (64, 128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_ch = feat_dim
        for ch in channels:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_ch = ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.backbone(x)
        return self.head(h)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        feat_dim: int = FEAT_DIM,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, h = self.gru(x)
        # use last layer hidden (concat directions if bi)
        if self.gru.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_last = h[-1]
        return self.head(h_last)


class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        feat_dim: int = FEAT_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 128,
    ):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        b, t, _ = x.shape
        h = self.proj(x) + self.pos[:, :t, :]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))
        return self.head(h)


def build_model(name: str = "conv1d", num_classes: int = 40, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("conv1d", "cnn", "conv"):
        return Conv1DClassifier(num_classes=num_classes, **kwargs)
    if name in ("gru", "rnn"):
        return GRUClassifier(num_classes=num_classes, **kwargs)
    if name in ("transformer", "xfmr", "tiny_transformer"):
        return TinyTransformerClassifier(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
