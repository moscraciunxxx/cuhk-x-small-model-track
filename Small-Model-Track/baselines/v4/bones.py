"""Bone / angle features from 17-kp skeleton (COCO-like)."""
from __future__ import annotations

import numpy as np

# COCO-17 topology (matches common OpenPose/MMPose 17kp HAR layouts)
BONE_PAIRS = [
    (5, 6),    # shoulders
    (11, 12),  # hips
    (5, 7), (7, 9),     # L arm
    (6, 8), (8, 10),    # R arm
    (5, 11), (6, 12),   # torso sides
    (11, 13), (13, 15), # L leg
    (12, 14), (14, 16), # R leg
    (0, 5), (0, 6),     # nose-shoulder
    (0, 1), (0, 2),     # nose-eyes (stable head)
    (1, 3), (2, 4),     # eyes-ears
]

# Adjacent bone triples for joint angles (i-j-k)
ANGLE_TRIPLES = [
    (5, 7, 9),   # L elbow
    (6, 8, 10),  # R elbow
    (11, 13, 15),  # L knee
    (12, 14, 16),  # R knee
    (7, 5, 11),  # L shoulder
    (8, 6, 12),  # R shoulder
    (5, 11, 13),  # L hip
    (6, 12, 14),  # R hip
    (5, 6, 12),  # torso
    (6, 5, 11),
]


def _joints(x: np.ndarray) -> np.ndarray:
    """(T, 51) -> (T, 17, 3)"""
    t = x.shape[0]
    return x.reshape(t, 17, 3)


def bone_features(x: np.ndarray) -> np.ndarray:
    """Compute bone vectors, lengths, velocities, angles.

    x: (T, 51) float
    returns: (T, F) float32
    """
    j = _joints(x)  # T,17,3
    T = j.shape[0]
    vecs = []
    lens = []
    for a, b in BONE_PAIRS:
        v = j[:, b] - j[:, a]  # T,3
        vecs.append(v)
        lens.append(np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    V = np.concatenate(vecs, axis=1)  # T, n_bones*3
    L = np.concatenate(lens, axis=1)  # T, n_bones
    # unit vectors
    U = V / (L.repeat(3, axis=1))
    # velocities (forward diff, pad first)
    dV = np.zeros_like(V)
    dV[1:] = V[1:] - V[:-1]
    dL = np.zeros_like(L)
    dL[1:] = L[1:] - L[:-1]
    # angles: cos of joint angle
    angs = []
    for i, jnt, k in ANGLE_TRIPLES:
        v1 = j[:, i] - j[:, jnt]
        v2 = j[:, k] - j[:, jnt]
        n1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-6
        n2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-6
        cos = np.sum(v1 * v2, axis=1, keepdims=True) / (n1 * n2)
        cos = np.clip(cos, -1.0, 1.0)
        angs.append(cos)
    A = np.concatenate(angs, axis=1)  # T, n_angles
    dA = np.zeros_like(A)
    dA[1:] = A[1:] - A[:-1]
    # mid-shoulder / mid-hip relative (torso orientation)
    mid_sh = 0.5 * (j[:, 5] + j[:, 6])
    mid_hp = 0.5 * (j[:, 11] + j[:, 12])
    torso = mid_sh - mid_hp  # T,3
    torso_len = np.linalg.norm(torso, axis=1, keepdims=True) + 1e-6
    torso_u = torso / torso_len
    feat = np.concatenate([U, L, dV, dL, A, dA, torso_u, torso_len], axis=1)
    return feat.astype(np.float32)


def bone_dim() -> int:
    n_bones = len(BONE_PAIRS)
    n_ang = len(ANGLE_TRIPLES)
    # U(n*3) + L(n) + dV(n*3) + dL(n) + A(na) + dA(na) + torso_u(3) + torso_len(1)
    return n_bones * 3 + n_bones + n_bones * 3 + n_bones + n_ang + n_ang + 3 + 1


def precompute_bone_cache(X_skel: np.ndarray) -> np.ndarray:
    """X_skel (N,T,51) -> (N,T,F)"""
    out = np.zeros((X_skel.shape[0], X_skel.shape[1], bone_dim()), dtype=np.float32)
    for i in range(len(X_skel)):
        out[i] = bone_features(X_skel[i])
    return out


if __name__ == "__main__":
    print("bone_dim", bone_dim())
    x = np.random.randn(64, 51).astype(np.float32)
    f = bone_features(x)
    print(f.shape)
