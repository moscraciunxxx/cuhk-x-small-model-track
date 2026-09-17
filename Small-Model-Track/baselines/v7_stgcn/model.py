"""Small ST-GCN (+ optional IMU mid-fuse) for CUHK-X Small Model Track v7."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import NUM_JOINTS, get_A_tensor

IMU_DIM = 30  # 5 devices x 6


class SpatialGraphConv(nn.Module):
    """Graph convolution over V joints with K hop kernels: out = sum_k A_k x W_k."""

    def __init__(self, in_channels: int, out_channels: int, max_hop: int = 1):
        super().__init__()
        self.max_hop = max_hop
        self.K = max_hop + 1
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)
        self.register_buffer("A", get_A_tensor(max_hop=max_hop), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        y = self.conv(x)  # (N, out*K, T, V)
        y = y.view(N, self.K, -1, T, V)  # (N, K, Cout, T, V)
        A = self.A  # (K, V, V)
        out = torch.einsum("kvw,nkctw->nctv", A, y)
        return out


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 9,
        stride: int = 1,
        max_hop: int = 1,
        dropout: float = 0.3,
        residual: bool = True,
    ):
        super().__init__()
        assert temporal_kernel % 2 == 1
        pad = (temporal_kernel - 1) // 2
        self.gcn = SpatialGraphConv(in_channels, out_channels, max_hop=max_hop)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(pad, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


def joints_with_velocity(x: torch.Tensor) -> torch.Tensor:
    """(N,T,51) or (N,T,17,3) -> (N, 6, T, 17) = xyz + frame delta."""
    if x.dim() == 3:
        n, t, _ = x.shape
        j = x.view(n, t, NUM_JOINTS, 3)
    else:
        j = x
        n, t = j.shape[0], j.shape[1]
    # (N, C=3, T, V)
    pos = j.permute(0, 3, 1, 2).contiguous()
    vel = torch.zeros_like(pos)
    vel[:, :, 1:, :] = pos[:, :, 1:, :] - pos[:, :, :-1, :]
    return torch.cat([pos, vel], dim=1)  # (N, 6, T, V)


class SmallSTGCN(nn.Module):
    """Compact ST-GCN on 17 joints. Input (N,T,51) or (N,T,17,3). Uses xyz+vel (C=6)."""

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 6,
        channels: tuple = (64, 64, 128, 128, 256),
        max_hop: int = 1,
        dropout: float = 0.35,
        temporal_kernel: int = 9,
        use_velocity: bool = True,
    ):
        super().__init__()
        self.use_velocity = use_velocity
        c_in = 6 if use_velocity else 3
        self.data_bn = nn.BatchNorm1d(c_in * NUM_JOINTS)
        layers = []
        ch_in = c_in
        # temporal downsample at blocks 1 and 3
        down = {1, 3}
        for i, ch in enumerate(channels):
            stride = 2 if i in down else 1
            layers.append(
                STGCNBlock(
                    ch_in,
                    ch,
                    temporal_kernel=temporal_kernel,
                    stride=stride,
                    max_hop=max_hop,
                    dropout=dropout,
                    residual=True,
                )
            )
            ch_in = ch
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self.out_dim = channels[-1]

    def _to_nctv(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_velocity:
            return joints_with_velocity(x)
        if x.dim() == 3:
            n, t, f = x.shape
            x = x.view(n, t, NUM_JOINTS, 3).permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def encode(self, x_skel: torch.Tensor) -> torch.Tensor:
        x = self._to_nctv(x_skel)
        n, c, t, v = x.shape
        x = x.view(n, c * v, t)
        x = self.data_bn(x)
        x = x.view(n, c, t, v)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).view(n, -1)
        return x

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        return self.head(self.encode(x_skel))


class TemporalConvBranch(nn.Module):
    """Lightweight Conv1D branch for IMU (reuse MidFuse idea)."""

    def __init__(
        self,
        in_dim: int,
        channels: tuple = (64, 128, 192),
        dropout: float = 0.3,
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
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.pool(h).flatten(1)


class STGCNMidFuse(nn.Module):
    """ST-GCN skeleton encoder + IMU Conv branch, mid-level concat."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_channels: tuple = (64, 64, 128, 128, 256),
        imu_channels: tuple = (64, 128, 192),
        imu_dim: int = IMU_DIM,
        max_hop: int = 1,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.skel = SmallSTGCN(
            num_classes=num_classes,
            channels=skel_channels,
            max_hop=max_hop,
            dropout=dropout,
            use_velocity=True,
        )
        self.imu = TemporalConvBranch(imu_dim, imu_channels, dropout=dropout)
        fuse = self.skel.out_dim + self.imu.out_dim
        self.imu_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.imu.out_dim),
            nn.Sigmoid(),
        )
        self.fuse_head = nn.Sequential(
            nn.LayerNorm(fuse),
            nn.Dropout(dropout),
            nn.Linear(fuse, fuse // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fuse // 2, num_classes),
        )

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        hs = self.skel.encode(x_skel)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), self.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                gate = self.imu_gate(flag)
                hi = hi * gate * flag
        return self.fuse_head(torch.cat([hs, hi], dim=-1))


def build_model(name: str = "stgcn_fuse", num_classes: int = 40, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("stgcn", "st-gcn", "skeleton"):
        return SmallSTGCN(num_classes=num_classes, **kwargs)
    if name in ("stgcn_fuse", "stgcn_midfuse", "fuse", "midfuse"):
        return STGCNMidFuse(num_classes=num_classes, **kwargs)
    if name in ("stgcn_2s", "2s", "twostream"):
        from model_2s import TwoStreamSTGCNFuse
        return TwoStreamSTGCNFuse(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ("stgcn", "stgcn_fuse"):
        m = build_model(name)
        n = count_parameters(m)
        x = torch.randn(2, 64, 51)
        xi = torch.randn(2, 64, 30)
        flag = torch.ones(2)
        y = m(x, xi, flag)
        print(name, "params", n, "out", tuple(y.shape), "MB~", n * 4 / 1e6)

