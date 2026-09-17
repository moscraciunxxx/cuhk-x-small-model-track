"""v4 models: BoneMidFuse, TripleFuse (skel+bone+imu), TinyThermalCNN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SKEL_DIM, IMU_DIM
from bones import bone_dim
from model_base import (
    TemporalConvBranch,
    MidFuseNet,
    DeepConv1D,
    GRUAttn,
    TinyTransformer,
    CompactConv,
    CompactMidFuse,
    count_parameters,
)


class BoneMidFuse(nn.Module):
    """Bone-feature Conv + IMU Conv mid-fuse (no raw joints)."""

    def __init__(
        self,
        num_classes: int = 40,
        bone_in: int = None,
        imu_dim: int = IMU_DIM,
        bone_channels: tuple = (96, 192, 256),
        imu_channels: tuple = (64, 128, 128),
        n_res: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if bone_in is None:
            bone_in = bone_dim()
        self.bone = TemporalConvBranch(bone_in, bone_channels, n_res, dropout)
        self.imu = TemporalConvBranch(imu_dim, imu_channels, n_res, dropout)
        fuse_dim = self.bone.out_dim + self.imu.out_dim
        self.imu_gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True),
            nn.Linear(16, self.imu.out_dim), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fuse_dim // 2, num_classes),
        )

    def forward(self, x_bone, x_imu=None, imu_flag=None):
        hb = self.bone(x_bone)
        if x_imu is None:
            hi = torch.zeros(hb.size(0), self.imu.out_dim, device=hb.device, dtype=hb.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                hi = hi * self.imu_gate(flag) * flag
        return self.head(torch.cat([hb, hi], dim=-1))


class TripleFuse(nn.Module):
    """Raw skel + bone + IMU mid-fuse."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_dim: int = SKEL_DIM,
        bone_in: int = None,
        imu_dim: int = IMU_DIM,
        skel_channels: tuple = (64, 128, 192),
        bone_channels: tuple = (64, 128, 192),
        imu_channels: tuple = (48, 96, 96),
        n_res: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if bone_in is None:
            bone_in = bone_dim()
        self.skel = TemporalConvBranch(skel_dim, skel_channels, n_res, dropout)
        self.bone = TemporalConvBranch(bone_in, bone_channels, n_res, dropout)
        self.imu = TemporalConvBranch(imu_dim, imu_channels, n_res, dropout)
        fuse_dim = self.skel.out_dim + self.bone.out_dim + self.imu.out_dim
        self.imu_gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True),
            nn.Linear(16, self.imu.out_dim), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fuse_dim // 2, num_classes),
        )

    def forward(self, x_skel, x_bone, x_imu=None, imu_flag=None):
        hs = self.skel(x_skel)
        hb = self.bone(x_bone)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), self.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                hi = hi * self.imu_gate(flag) * flag
        return self.head(torch.cat([hs, hb, hi], dim=-1))


class TinyThermalCNN(nn.Module):
    """Tiny frame CNN: sample N frames -> logits. ~<<10MB."""

    def __init__(self, num_classes: int = 40, in_ch: int = 1, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temp = nn.Sequential(
            nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        # x: (B, N, H, W) or (B, N, 1, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(2)
        B, N, C, H, W = x.shape
        h = self.backbone(x.reshape(B * N, C, H, W))  # B*N,96,1,1
        h = h.view(B, N, 96).transpose(1, 2)  # B,96,N
        h = self.temp(h)  # B,96,1
        return self.head(h)


class GatedThermalFuse(nn.Module):
    """MidFuse logits prior + thermal residual when available."""

    def __init__(self, num_classes: int = 40, dropout: float = 0.3):
        super().__init__()
        self.thermal = TinyThermalCNN(num_classes=num_classes, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, midfuse_logits, frames, has_thermal):
        # frames: (B,N,H,W); has_thermal: (B,)
        th = self.thermal(frames)
        a = torch.sigmoid(self.alpha)
        flag = has_thermal.view(-1, 1).to(th.dtype)
        return midfuse_logits + a * flag * th


def build_model(name: str = "bonemidfuse", num_classes: int = 40, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("bonemidfuse", "bone_midfuse", "bone"):
        return BoneMidFuse(num_classes=num_classes, **kwargs)
    if name in ("triple", "triplefuse", "skel_bone_imu"):
        return TripleFuse(num_classes=num_classes, **kwargs)
    if name in ("midfuse", "fuse", "skeleton_imu"):
        return MidFuseNet(num_classes=num_classes, **kwargs)
    if name in ("deepconv", "conv1d", "cnn"):
        return DeepConv1D(num_classes=num_classes, **kwargs)
    if name in ("gru", "gru_attn"):
        return GRUAttn(num_classes=num_classes, **kwargs)
    if name in ("transformer", "xfmr"):
        return TinyTransformer(num_classes=num_classes, **kwargs)
    if name in ("thermal", "tiny_thermal"):
        return TinyThermalCNN(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")


__all__ = [
    "BoneMidFuse", "TripleFuse", "TinyThermalCNN", "GatedThermalFuse",
    "build_model", "count_parameters", "MidFuseNet",
]
