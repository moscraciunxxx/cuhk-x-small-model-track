"""Multi-scale temporal + bone-stream ST-GCN fuse (v13 complementary base)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

V7 = Path(__file__).resolve().parent.parent / "v7_stgcn"
sys.path.insert(0, str(V7))

def _load_v7_model():
    spec = importlib.util.spec_from_file_location("v7_model_ms", V7 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_v7_2s():
    spec = importlib.util.spec_from_file_location("v7_model_2s_ms", V7 / "model_2s.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_v7 = _load_v7_model()
_v72 = _load_v7_2s()
IMU_DIM = _v7.IMU_DIM
TemporalConvBranch = _v7.TemporalConvBranch
joints_with_velocity = _v7.joints_with_velocity
count_parameters = _v7.count_parameters
joints_to_bones = _v72.joints_to_bones

from graph import get_A_tensor  # noqa: E402


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_hop: int = 1):
        super().__init__()
        self.K = max_hop + 1
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)
        self.register_buffer("A", get_A_tensor(max_hop=max_hop), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        y = self.conv(x).view(N, self.K, -1, T, V)
        return torch.einsum("kvw,nkctw->nctv", self.A, y)


class MSTemporal(nn.Module):
    def __init__(self, channels: int, stride: int = 1, dropout: float = 0.3):
        super().__init__()
        branches = []
        for k in (3, 5, 9):
            pad = (k - 1) // 2
            branches.append(
                nn.Conv2d(channels, channels, kernel_size=(k, 1), stride=(stride, 1), padding=(pad, 0), bias=False)
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.bn(x))
        y = torch.cat([b(h) for b in self.branches], dim=1)
        return self.fuse(y)


class MSSTGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, max_hop=1, dropout=0.3, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, max_hop=max_hop)
        self.tcn = MSTemporal(out_channels, stride=stride, dropout=dropout)
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


class MSSmallSTGCN(nn.Module):
    def __init__(self, in_channels=6, channels=(64, 64, 128, 128), max_hop=1, dropout=0.35):
        super().__init__()
        layers = []
        ch_in = in_channels
        for i, ch in enumerate(channels):
            stride = 2 if i in (1, 3) else 1
            layers.append(MSSTGCNBlock(ch_in, ch, stride=stride, max_hop=max_hop, dropout=dropout))
            ch_in = ch
        self.blocks = nn.Sequential(*layers)
        self.out_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def encode(self, x_skel):
        x = joints_with_velocity(x_skel)
        h = self.blocks(x)
        return self.pool(h).flatten(1)


class MSTwoStreamSTGCNFuse(nn.Module):
    def __init__(self, num_classes=40, skel_channels=(64, 64, 128, 128), imu_channels=(64, 128, 128), dropout=0.35):
        super().__init__()
        self.joint = MSSmallSTGCN(channels=skel_channels, dropout=dropout)
        self.bone = MSSmallSTGCN(channels=skel_channels, dropout=dropout)
        self.imu = TemporalConvBranch(IMU_DIM, imu_channels, dropout=dropout)
        fuse = self.joint.out_dim + self.bone.out_dim + self.imu.out_dim
        self.imu_gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, self.imu.out_dim), nn.Sigmoid()
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
        hj = self.joint.encode(x_skel)
        hb = self.bone.encode(joints_to_bones(x_skel))
        if x_imu is None:
            hi = torch.zeros(hj.size(0), self.imu.out_dim, device=hj.device, dtype=hj.dtype)
        else:
            hi = self.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                hi = hi * self.imu_gate(flag) * flag
        return self.fuse_head(torch.cat([hj, hb, hi], dim=-1))


def build_model(name="ms_stgcn_2s", num_classes=40, **kwargs):
    if name.lower() in ("ms_stgcn_2s", "ms2s", "stgcn_ms_2s"):
        return MSTwoStreamSTGCNFuse(num_classes=num_classes, **kwargs)
    raise ValueError(name)


if __name__ == "__main__":
    m = build_model()
    print("params", count_parameters(m))
    print(m(torch.randn(2, 64, 51), torch.randn(2, 64, 30), torch.ones(2)).shape)
