"""COCO / body-17 skeleton graph for ST-GCN.

Joint order verified from training JSON keypoints (17x3) and prior
baselines/v4/bones.py COCO-like topology used across this track:

  0 nose, 1 L_eye, 2 R_eye, 3 L_ear, 4 R_ear,
  5 L_shoulder, 6 R_shoulder, 7 L_elbow, 8 R_elbow,
  9 L_wrist, 10 R_wrist, 11 L_hip, 12 R_hip,
  13 L_knee, 14 R_knee, 15 L_ankle, 16 R_ankle
"""
from __future__ import annotations

import numpy as np
import torch

NUM_JOINTS = 17

# Undirected bone edges (i, j) with i < j preferred but order free
COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (0, 5), (0, 6),  # nose-shoulders
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

JOINT_NAMES = [
    "nose",
    "L_eye",
    "R_eye",
    "L_ear",
    "R_ear",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hip",
    "R_hip",
    "L_knee",
    "R_knee",
    "L_ankle",
    "R_ankle",
]


def adjacency_matrix(
    num_joints: int = NUM_JOINTS,
    edges=COCO17_EDGES,
    self_loops: bool = True,
) -> np.ndarray:
    """Binary adjacency A (V,V)."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        np.fill_diagonal(A, 1.0)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}."""
    A = A.astype(np.float32)
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-6), -0.5)
    D = np.diag(deg_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


def build_hop_adjacencies(
    num_joints: int = NUM_JOINTS,
    edges=COCO17_EDGES,
    max_hop: int = 1,
) -> np.ndarray:
    """Return stack of hop-partitioned normalized adjacencies (K, V, V).

    K = max_hop + 1 partitions: hop0 = identity, hop1 = 1-hop neighbors, ...
    Matching classic ST-GCN spatial configuration (simplified).
    """
    A_bin = adjacency_matrix(num_joints, edges, self_loops=False)
    # all-pairs shortest hop via repeated multiply
    hop = np.full((num_joints, num_joints), fill_value=num_joints + 1, dtype=np.int32)
    np.fill_diagonal(hop, 0)
    reach = np.eye(num_joints, dtype=np.float32)
    for d in range(1, max_hop + 1):
        reach = ((reach @ A_bin) > 0).astype(np.float32)
        mask = (hop > d) & (reach > 0)
        hop[mask] = d
    As = []
    for d in range(max_hop + 1):
        Ad = (hop == d).astype(np.float32)
        As.append(normalize_adjacency(Ad))
    return np.stack(As, axis=0)  # (K, V, V)


def get_A_tensor(max_hop: int = 1, device=None, dtype=torch.float32) -> torch.Tensor:
    A = build_hop_adjacencies(max_hop=max_hop)
    t = torch.tensor(A, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


if __name__ == "__main__":
    A = build_hop_adjacencies(max_hop=1)
    print("A shape", A.shape, "sum hops", A.sum(axis=(1, 2)))
    print("edges", len(COCO17_EDGES))
