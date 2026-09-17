"""Two-stream (joint + bone) ST-GCN MidFuse for v7."""
from __future__ import annotations

import torch
import torch.nn as nn

from graph import COCO17_EDGES, NUM_JOINTS
from model import IMU_DIM, SmallSTGCN, TemporalConvBranch, joints_with_velocity


# Map each joint to a bone vector target: bone[j] = joint[j] - joint[parent]
# Using a spanning tree from nose/shoulders/hips.
BONE_PARENTS = [
    0,  # 0 nose -> self (zero bone) handled specially
    0,  # 1 L_eye <- nose
    0,  # 2 R_eye <- nose
    1,  # 3 L_ear <- L_eye
    2,  # 4 R_ear <- R_eye
    0,  # 5 L_sho <- nose
    0,  # 6 R_sho <- nose
    5,  # 7 L_elb <- L_sho
    6,  # 8 R_elb <- R_sho
    7,  # 9 L_wri <- L_elb
    8,  # 10 R_wri <- R_elb
    5,  # 11 L_hip <- L_sho
    6,  # 12 R_hip <- R_sho
    11, # 13 L_kne <- L_hip
    12, # 14 R_kne <- R_hip
    13, # 15 L_ank <- L_kne
    14, # 16 R_ank <- R_kne
]


def joints_to_bones(x: torch.Tensor) -> torch.Tensor:
    """(N,T,51) -> (N,T,51) bone vectors aligned to child joints."""
    n, t, _ = x.shape
    j = x.view(n, t, NUM_JOINTS, 3)
    b = torch.zeros_like(j)
    for child, parent in enumerate(BONE_PARENTS):
        if child == 0:
            continue
        b[:, :, child, :] = j[:, :, child, :] - j[:, :, parent, :]
    # root: mid-shoulder - mid-hip as torso bone stored at joint 0
    mid_sh = 0.5 * (j[:, :, 5, :] + j[:, :, 6, :])
    mid_hp = 0.5 * (j[:, :, 11, :] + j[:, :, 12, :])
    b[:, :, 0, :] = mid_sh - mid_hp
    return b.reshape(n, t, NUM_JOINTS * 3)


class TwoStreamSTGCNFuse(nn.Module):
    """Joint ST-GCN + Bone ST-GCN + IMU Conv, mid fuse."""

    def __init__(
        self,
        num_classes: int = 40,
        skel_channels: tuple = (64, 64, 128, 128),
        imu_channels: tuple = (64, 128, 128),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.joint = SmallSTGCN(
            num_classes=num_classes,
            channels=skel_channels,
            dropout=dropout,
            use_velocity=True,
        )
        self.bone = SmallSTGCN(
            num_classes=num_classes,
            channels=skel_channels,
            dropout=dropout,
            use_velocity=True,
        )
        self.imu = TemporalConvBranch(IMU_DIM, imu_channels, dropout=dropout)
        fuse = self.joint.out_dim + self.bone.out_dim + self.imu.out_dim
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


def build_2s(num_classes: int = 40, **kwargs):
    return TwoStreamSTGCNFuse(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    from model import count_parameters

    m = build_2s()
    print("params", count_parameters(m))
    y = m(torch.randn(2, 64, 51), torch.randn(2, 64, 30), torch.ones(2))
    print(y.shape)
