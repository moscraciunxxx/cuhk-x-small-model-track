"""MidFusePlus lite: exact v2b skel (n_res=1) + multi-scale IMU only."""
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


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x).unsqueeze(-1)


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


class MultiScaleIMUEncoder(nn.Module):
    def __init__(self, in_dim=30, stem=40, channels=(64, 128, 128), dropout=0.3):
        super().__init__()
        self.stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_dim, stem, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(stem),
                    nn.ReLU(inplace=True),
                )
                for k in (3, 5, 7)
            ]
        )
        layers = [
            nn.Conv1d(stem * 3, channels[0], kernel_size=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            SE1D(channels[0]),
            nn.Dropout(dropout),
        ]
        ch_in = channels[0]
        for i, ch in enumerate(channels):
            if i == 0:
                layers += [ResBlock1D(ch, dropout=dropout), nn.MaxPool1d(2), nn.Dropout(dropout)]
                continue
            layers += [
                nn.Conv1d(ch_in, ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
                ResBlock1D(ch, dropout=dropout),
            ]
            ch_in = ch
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        xt = x.transpose(1, 2)
        h = torch.cat([s(xt) for s in self.stems], dim=1)
        return self.pool(self.net(h)).flatten(1)


class MidFusePlus(nn.Module):
    def __init__(self, num_classes=40, dropout=0.3):
        super().__init__()
        self.skel = TemporalConvBranch(51, (96, 192, 256), n_res=1, dropout=dropout)
        self.imu = MultiScaleIMUEncoder(dropout=dropout)
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
        hs = self.skel(x_skel)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), self.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                hi = hi * self.imu_gate(flag) * flag
        return self.head(torch.cat([hs, hi], dim=-1))


MidFuseWide = MidFusePlus


def build_model(name="midfuse_plus", num_classes=40, **kwargs):
    if name.lower() in ("midfuse_wide", "mfw", "wide", "midfuse_plus", "mfp"):
        return MidFusePlus(num_classes=num_classes, **kwargs)
    raise ValueError(name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = MidFusePlus()
    print("params", count_parameters(m))
    print(m(torch.randn(2, 64, 51), torch.randn(2, 64, 30), torch.ones(2)).shape)
