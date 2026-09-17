"""Stronger small models for CUHK-X skeleton / skeleton+IMU v2.

Targets: few MB on disk, CNN/RNN/Transformer only, no large pretrained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SKEL_DIM, IMU_DIM


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + residual, inplace=True)


class TemporalConvBranch(nn.Module):
    """Deeper Conv1D with residual blocks -> pooled vector."""

    def __init__(
        self,
        in_dim: int,
        channels: tuple = (64, 128, 256),
        n_res: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        layers = []
        ch_in = in_dim
        for ch in channels:
            layers += [
                nn.Conv1d(ch_in, ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            ch_in = ch
            for _ in range(n_res):
                layers.append(ResBlock1D(ch, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        h = self.net(x.transpose(1, 2))
        return self.pool(h).flatten(1)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):
        # h: (B, T, D)
        w = torch.softmax(self.score(h), dim=1)  # (B, T, 1)
        return (h * w).sum(dim=1)


class DeepConv1D(nn.Module):
    """Skeleton-only deeper residual Conv1D."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        channels: tuple = (96, 192, 256),
        n_res: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branch = TemporalConvBranch(skel_dim, channels, n_res, dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.branch.out_dim, self.branch.out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.branch.out_dim // 2, num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        return self.head(self.branch(x_skel))


class GRUAttn(nn.Module):
    """BiGRU + attention pooling (skeleton-only)."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        hidden: int = 160,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(skel_dim)
        self.gru = nn.GRU(
            skel_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        d = hidden * 2
        self.attn = AttentionPool(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        h = self.input_norm(x_skel)
        out, _ = self.gru(h)
        return self.head(self.attn(out))


class MidFuseNet(nn.Module):
    """Skeleton Conv branch + IMU Conv branch, mid-level concat fuse."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        imu_dim: int = IMU_DIM,
        skel_channels: tuple = (96, 192, 256),
        imu_channels: tuple = (64, 128, 128),
        n_res: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.skel = TemporalConvBranch(skel_dim, skel_channels, n_res, dropout)
        self.imu = TemporalConvBranch(imu_dim, imu_channels, n_res, dropout)
        fuse_dim = self.skel.out_dim + self.imu.out_dim
        self.imu_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.imu.out_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fuse_dim // 2, num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        hs = self.skel(x_skel)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), self.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                # imu_flag: (B,) or (B,1) — gate IMU features when missing
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                gate = self.imu_gate(flag)
                hi = hi * gate * flag
        return self.head(torch.cat([hs, hi], dim=-1))


class TinyTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        d_model: int = 160,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 320,
        dropout: float = 0.2,
        max_len: int = 128,
    ):
        super().__init__()
        self.proj = nn.Linear(skel_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.attn = AttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        b, t, _ = x_skel.shape
        h = self.proj(x_skel) + self.pos[:, :t, :]
        h = self.encoder(h)
        h = self.norm(self.attn(h))
        return self.head(h)




class CompactConv(nn.Module):
    """v1-sized Conv1D with one residual stage — less overfit than DeepConv."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        channels: tuple = (64, 128, 256),
        dropout: float = 0.35,
    ):
        super().__init__()
        layers = []
        ch_in = skel_dim
        for i, ch in enumerate(channels):
            layers += [
                nn.Conv1d(ch_in, ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ]
            if i >= 1:
                layers.append(ResBlock1D(ch, dropout=dropout * 0.5))
            layers += [nn.MaxPool1d(2), nn.Dropout(dropout)]
            ch_in = ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        h = self.backbone(x_skel.transpose(1, 2))
        return self.head(h)


class CompactMidFuse(nn.Module):
    """Compact skel Conv + small IMU Conv fuse (keeps param count modest)."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        imu_dim: int = IMU_DIM,
        skel_channels: tuple = (64, 128, 256),
        imu_channels: tuple = (32, 64, 64),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.skel = TemporalConvBranch(skel_dim, skel_channels, n_res=1, dropout=dropout)
        self.imu = TemporalConvBranch(imu_dim, imu_channels, n_res=0, dropout=dropout)
        fuse = self.skel.out_dim + self.imu.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fuse, num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        hs = self.skel(x_skel)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), self.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                hi = hi * imu_flag.view(-1, 1).to(hi.dtype)
        return self.head(torch.cat([hs, hi], dim=-1))



class MidFuseWide(MidFuseNet):
    """Slightly wider MidFuse (~3.3M params, still <<100MB)."""

    def __init__(self, num_classes: int = 40, **kwargs):
        kwargs.setdefault("skel_channels", (128, 256, 320))
        kwargs.setdefault("imu_channels", (96, 160, 192))
        kwargs.setdefault("n_res", 1)
        kwargs.setdefault("dropout", 0.3)
        super().__init__(num_classes=num_classes, **kwargs)

def build_model(name: str = "midfuse", num_classes: int = 40, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("midfuse", "fuse", "skeleton_imu"):
        return MidFuseNet(num_classes=num_classes, **kwargs)
    if name in ("midfuse_wide", "wide", "midfuse_v3"):
        return MidFuseWide(num_classes=num_classes, **kwargs)
    if name in ("compact_fuse", "compact_midfuse"):
        return CompactMidFuse(num_classes=num_classes, **kwargs)
    if name in ("compact", "compact_conv"):
        return CompactConv(num_classes=num_classes, **kwargs)
    if name in ("deepconv", "conv1d", "cnn"):
        return DeepConv1D(num_classes=num_classes, **kwargs)
    if name in ("gru", "gru_attn", "gruattn"):
        return GRUAttn(num_classes=num_classes, **kwargs)
    if name in ("transformer", "xfmr"):
        return TinyTransformer(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

