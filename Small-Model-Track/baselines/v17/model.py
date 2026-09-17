"""MidFusePlus v15: exact MidFuseNet (v2b) + skeleton velocity; modest params.

Broken v13/v14 MidFusePlus used MultiScaleIMUEncoder (~3.2M) and lagged v2b
(~0.30 fold0 vs ~0.45). This version keeps proven TemporalConvBranch IMU and
only adds temporal velocity on skeleton (51 -> 102 in_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SKEL_DIM = 51
IMU_DIM = 30


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
    def __init__(self, in_dim, channels=(96, 192, 256), n_res=1, dropout=0.3):
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
        return self.pool(self.net(x.transpose(1, 2))).flatten(1)


def skel_with_velocity(x_skel: torch.Tensor) -> torch.Tensor:
    """(B,T,51) -> (B,T,102) concat position + first-difference velocity."""
    v = torch.zeros_like(x_skel)
    v[:, 1:] = x_skel[:, 1:] - x_skel[:, :-1]
    return torch.cat([x_skel, v], dim=-1)


class MidFusePlus(nn.Module):
    """v2b MidFuseNet + skeleton velocity channels."""

    def __init__(self, num_classes=40, dropout=0.3, use_velocity=True):
        super().__init__()
        self.use_velocity = use_velocity
        skel_in = SKEL_DIM * 2 if use_velocity else SKEL_DIM
        self.skel = TemporalConvBranch(skel_in, (96, 192, 256), n_res=1, dropout=dropout)
        self.imu = TemporalConvBranch(IMU_DIM, (64, 128, 128), n_res=1, dropout=dropout)
        fuse = self.skel.out_dim + self.imu.out_dim
        self.imu_gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, self.imu.out_dim), nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(fuse),
            nn.Dropout(dropout),
            nn.Linear(fuse, fuse // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fuse // 2, num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        xs = skel_with_velocity(x_skel) if self.use_velocity else x_skel
        hs = self.skel(xs)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), self.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                hi = hi * self.imu_gate(flag) * flag
        return self.head(torch.cat([hs, hi], dim=-1))


def build_model(name="midfuse_plus", num_classes=40, **kwargs):
    n = name.lower()
    if n in ("midfuse_plus", "mfp", "midfuse_vel", "plus"):
        return MidFusePlus(num_classes=num_classes, **kwargs)
    if n in ("midfuse_nov", "midfuse_base"):
        return MidFusePlus(num_classes=num_classes, use_velocity=False, **kwargs)
    raise ValueError(name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = MidFusePlus()
    print("params", count_parameters(m))
    print(m(torch.randn(2, 64, 51), torch.randn(2, 64, 30), torch.ones(2)).shape)
    m0 = MidFusePlus(use_velocity=False)
    print("nov_params", count_parameters(m0))
