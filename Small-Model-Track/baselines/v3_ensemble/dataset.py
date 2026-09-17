"""Skeleton + IMU dataset for CUHK-X Small Model Track v2.

Skeleton: reuse T x 51 cache from v1.
IMU: 5 devices x (acc+gyro=6) = 30 feats, resampled to same T.
Align train trials by (action_name, user_id, trial).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_JOINTS = 17
COORDS = 3
SKEL_DIM = NUM_JOINTS * COORDS  # 51
IMU_DEVICES = ("WTLA", "WTRA", "WTC", "WTLL", "WTRL")  # fixed order
IMU_PER_DEV = 6  # acc xyz + gyro xyz
IMU_DIM = len(IMU_DEVICES) * IMU_PER_DEV  # 30
DEFAULT_T = 64
DEFAULT_HOLD_OUT_USERS = (8, 9, 24)

ACTION_DIR_RE = re.compile(r"^(\d+)_")
USER_RE = re.compile(r"^user(\d+)$", re.IGNORECASE)

# Chinese column names (UTF-8-BOM CSVs)
COL_TIME = "时间"
COL_DEV = "设备名称"
COL_AX, COL_AY, COL_AZ = "加速度X(g)", "加速度Y(g)", "加速度Z(g)"
COL_GX, COL_GY, COL_GZ = "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"
IMU_FEAT_COLS = [COL_AX, COL_AY, COL_AZ, COL_GX, COL_GY, COL_GZ]


def parse_action_id(folder_name: str) -> int:
    m = ACTION_DIR_RE.match(folder_name)
    if not m:
        raise ValueError(f"Cannot parse action id from {folder_name!r}")
    return int(m.group(1))


def parse_user_id(folder_name: str) -> int:
    m = USER_RE.match(folder_name)
    if not m:
        raise ValueError(f"Cannot parse user id from {folder_name!r}")
    return int(m.group(1))


def device_key(name: str) -> Optional[str]:
    s = str(name).split("(")[0].strip().upper()
    for d in IMU_DEVICES:
        if s.startswith(d) or d in s:
            return d
    return None


def resample_1d(seq: np.ndarray, T: int) -> np.ndarray:
    """seq (T_raw, F) -> (T, F) linear interp."""
    t_raw = seq.shape[0]
    if t_raw == T:
        return seq.astype(np.float32, copy=False)
    if t_raw == 0:
        return np.zeros((T, seq.shape[1]), dtype=np.float32)
    if t_raw == 1:
        return np.repeat(seq, T, axis=0).astype(np.float32)
    src = np.linspace(0, t_raw - 1, num=T, dtype=np.float64)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, t_raw - 1)
    w = (src - i0).astype(np.float32)[:, None]
    return ((1.0 - w) * seq[i0] + w * seq[i1]).astype(np.float32)


def load_imu_trial(trial_dir: Path, T: int = DEFAULT_T) -> np.ndarray:
    """Load up+down CSVs -> (T, 30) float32. Missing devices -> zeros."""
    trial_dir = Path(trial_dir)
    per_dev: Dict[str, List[np.ndarray]] = {d: [] for d in IMU_DEVICES}
    for csv_path in list(trial_dir.glob("up*.csv")) + list(trial_dir.glob("down*.csv")):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            continue
        if COL_DEV not in df.columns:
            continue
        if any(c not in df.columns for c in IMU_FEAT_COLS):
            continue
        devs = df[COL_DEV].astype(str).map(device_key)
        feats = df[IMU_FEAT_COLS].to_numpy(dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        for d in IMU_DEVICES:
            mask = (devs == d).to_numpy()
            if mask.any():
                per_dev[d].append(feats[mask])

    blocks = []
    for d in IMU_DEVICES:
        arrs = per_dev[d]
        if not arrs:
            blocks.append(np.zeros((T, IMU_PER_DEV), dtype=np.float32))
            continue
        seq = np.concatenate(arrs, axis=0)
        seq = resample_1d(seq, T)
        mu = seq.mean(axis=0, keepdims=True)
        sd = seq.std(axis=0, keepdims=True) + 1e-6
        seq = (seq - mu) / sd
        blocks.append(seq.astype(np.float32))
    return np.concatenate(blocks, axis=1)


def discover_imu_trials(imu_root: Path) -> Dict[Tuple[str, int, str], str]:
    """Map (action_name, user_id, trial) -> trial_dir path."""
    imu_root = Path(imu_root)
    out: Dict[Tuple[str, int, str], str] = {}
    if not imu_root.is_dir():
        return out
    for action_dir in sorted(imu_root.iterdir()):
        if not action_dir.is_dir():
            continue
        try:
            parse_action_id(action_dir.name)
        except ValueError:
            continue
        for user_dir in sorted(action_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            try:
                uid = parse_user_id(user_dir.name)
            except ValueError:
                continue
            for trial_dir in sorted(user_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                if list(trial_dir.glob("*.csv")):
                    out[(action_dir.name, uid, trial_dir.name)] = str(trial_dir)
    return out


def discover_test_imu(test_root: Path) -> Dict[str, Optional[str]]:
    """clip_id -> IMU trial dir (folder containing csvs) or None."""
    test_root = Path(test_root)
    out: Dict[str, Optional[str]] = {}
    for clip_dir in sorted(test_root.glob("SM_test_*")):
        if not clip_dir.is_dir():
            continue
        imu_dir = clip_dir / "IMU"
        if imu_dir.is_dir() and list(imu_dir.glob("*.csv")):
            out[clip_dir.name] = str(imu_dir)
        else:
            out[clip_dir.name] = None
    return out


def load_skel_train_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    # prefer skel_train.npz (v2) else train.npz
    for name in ("skel_train.npz", "train.npz"):
        p = cache_dir / name
        if p.exists():
            data = np.load(p, allow_pickle=False)
            break
    else:
        raise FileNotFoundError(f"No skeleton train cache in {cache_dir}")
    X = data["X"]
    y = data["y"]
    users = data["users"]
    meta_path = cache_dir / "train_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return X, y, users, meta


def load_skel_test_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    for name in ("skel_test.npz", "test.npz"):
        p = cache_dir / name
        if p.exists():
            data = np.load(p, allow_pickle=True)
            break
    else:
        raise FileNotFoundError(f"No skeleton test cache in {cache_dir}")
    X = data["X"]
    paths = [str(p) for p in data["paths"].tolist()]
    return X, paths


class CachedDualDataset(Dataset):
    """Cached skeleton + optional IMU, with optional training augment."""

    def __init__(
        self,
        X_skel: np.ndarray,
        X_imu: np.ndarray,
        y: np.ndarray,
        users: np.ndarray,
        indices=None,
        has_imu: Optional[np.ndarray] = None,
        augment: bool = False,
        seed: int = 42,
    ):
        if indices is None:
            indices = np.arange(len(y))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.X_skel = X_skel
        self.X_imu = X_imu
        self.y = y
        self.users = users
        self.has_imu = has_imu if has_imu is not None else np.ones(len(y), dtype=np.bool_)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        xs = np.asarray(self.X_skel[idx], dtype=np.float32).copy()
        xi = np.asarray(self.X_imu[idx], dtype=np.float32).copy()
        if self.augment:
            # gaussian noise
            if self.rng.rand() < 0.5:
                xs += self.rng.randn(*xs.shape).astype(np.float32) * 0.02
            if self.rng.rand() < 0.5:
                xi += self.rng.randn(*xi.shape).astype(np.float32) * 0.02
            # temporal shift (circular)
            if self.rng.rand() < 0.5:
                shift = self.rng.randint(-4, 5)
                xs = np.roll(xs, shift, axis=0)
                xi = np.roll(xi, shift, axis=0)
            # time mask
            if self.rng.rand() < 0.3:
                t0 = self.rng.randint(0, xs.shape[0])
                w = self.rng.randint(1, max(2, xs.shape[0] // 8))
                xs[t0 : t0 + w] = 0
                xi[t0 : t0 + w] = 0
        y = int(self.y[idx])
        user = int(self.users[idx])
        imu_flag = float(self.has_imu[idx])
        return (
            torch.from_numpy(xs),
            torch.from_numpy(xi),
            y,
            user,
            imu_flag,
        )

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.y[self.indices], dtype=np.int64)

    @property
    def groups(self) -> np.ndarray:
        return np.asarray(self.users[self.indices], dtype=np.int64)


class CachedSkelDataset(Dataset):
    def __init__(self, X, y, users, indices=None, augment: bool = False, seed: int = 42):
        if indices is None:
            indices = np.arange(len(y))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.X = X
        self.y = y
        self.users = users
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], dtype=np.float32).copy()
        if self.augment:
            if self.rng.rand() < 0.5:
                x += self.rng.randn(*x.shape).astype(np.float32) * 0.02
            if self.rng.rand() < 0.5:
                x = np.roll(x, self.rng.randint(-4, 5), axis=0)
        return torch.from_numpy(x), int(self.y[idx]), int(self.users[idx])

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.y[self.indices], dtype=np.int64)
