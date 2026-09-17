"""Skeleton HAR dataset for CUHK-X Small Model Track.

Loads per-frame skeleton JSON sequences into fixed T x (17*3) tensors.
Labels from action folder id; GroupKFold-ready via user ids.
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
COORDS = 3  # x, y, z
FEAT_DIM = NUM_JOINTS * COORDS  # 51
DEFAULT_T = 64
DEFAULT_HOLD_OUT_USERS = (8, 9, 24)

ACTION_DIR_RE = re.compile(r"^(\d+)_")
USER_RE = re.compile(r"^user(\d+)$", re.IGNORECASE)


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


def load_frame_keypoints(path: Path) -> np.ndarray:
    """Load one frame JSON -> (17, 3) float32. Empty / missing person -> zeros."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = np.zeros((NUM_JOINTS, COORDS), dtype=np.float32)
    if not data:
        return out
    person = data[0]
    kps = person.get("keypoints")
    if kps is None:
        return out
    arr = np.asarray(kps, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < COORDS:
        return out
    n = min(NUM_JOINTS, arr.shape[0])
    out[:n] = arr[:n, :COORDS]
    return out


def frame_sort_key(p: Path) -> Tuple:
    """Sort frames by trailing frame index in filename when present."""
    stem = p.stem
    m = re.search(r"_(\d+)$", stem)
    if m:
        return (int(m.group(1)), stem)
    return (0, stem)


def load_sequence(pred_dir: Path) -> np.ndarray:
    """Load all JSON frames in predictions/ -> (T_raw, 17, 3)."""
    files = sorted(pred_dir.glob("*.json"), key=frame_sort_key)
    if not files:
        return np.zeros((1, NUM_JOINTS, COORDS), dtype=np.float32)
    frames = [load_frame_keypoints(fp) for fp in files]
    return np.stack(frames, axis=0)


def resample_sequence(seq: np.ndarray, T: int) -> np.ndarray:
    """Resample / pad / crop to fixed length T. seq: (T_raw, J, C) -> (T, J, C)."""
    t_raw = seq.shape[0]
    if t_raw == T:
        return seq.astype(np.float32, copy=False)
    if t_raw == 1:
        return np.repeat(seq, T, axis=0).astype(np.float32)
    # Linear interpolate along time for each joint/coord
    src_idx = np.linspace(0, t_raw - 1, num=T, dtype=np.float64)
    i0 = np.floor(src_idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, t_raw - 1)
    w = (src_idx - i0).astype(np.float32)
    out = (1.0 - w)[:, None, None] * seq[i0] + w[:, None, None] * seq[i1]
    return out.astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Center on root (joint 0) and scale by mean limb length proxy (std of coords)."""
    root = seq[:, 0:1, :]  # (T, 1, 3)
    centered = seq - root
    # Keep root xyz as relative zeros; scale by per-clip std of non-root
    flat = centered.reshape(-1)
    scale = np.std(flat) + 1e-6
    return (centered / scale).astype(np.float32)


def discover_train_samples(
    skeleton_root: Path,
) -> List[Dict]:
    """Walk train skeleton tree; one sample per trial folder with predictions/."""
    samples: List[Dict] = []
    skeleton_root = Path(skeleton_root)
    for action_dir in sorted(skeleton_root.iterdir()):
        if not action_dir.is_dir():
            continue
        try:
            label = parse_action_id(action_dir.name)
        except ValueError:
            continue
        for user_dir in sorted(action_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            try:
                user_id = parse_user_id(user_dir.name)
            except ValueError:
                continue
            for trial_dir in sorted(user_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                pred = trial_dir / "predictions"
                if not pred.is_dir():
                    continue
                n_json = sum(1 for _ in pred.glob("*.json"))
                if n_json == 0:
                    continue
                samples.append(
                    {
                        "pred_dir": str(pred),
                        "label": label,
                        "user_id": user_id,
                        "action_name": action_dir.name,
                        "trial": trial_dir.name,
                        "n_frames": n_json,
                    }
                )
    return samples


def discover_test_clips(test_root: Path) -> List[Dict]:
    """Discover test clips with Skeleton/predictions under SM_test_XXXX."""
    test_root = Path(test_root)
    clips: List[Dict] = []
    for clip_dir in sorted(test_root.glob("SM_test_*")):
        if not clip_dir.is_dir():
            continue
        pred = clip_dir / "Skeleton" / "predictions"
        if not pred.is_dir():
            # fallback: any nested predictions with json
            cands = list(clip_dir.rglob("predictions"))
            pred = None
            for c in cands:
                if any(c.glob("*.json")):
                    pred = c
                    break
            if pred is None:
                clips.append(
                    {
                        "clip_id": clip_dir.name,
                        "pred_dir": None,
                        "path": f"small_model_track_test/{clip_dir.name}/",
                    }
                )
                continue
        clips.append(
            {
                "clip_id": clip_dir.name,
                "pred_dir": str(pred),
                "path": f"small_model_track_test/{clip_dir.name}/",
            }
        )
    return clips


def load_class_mapping(csv_path: Path) -> Dict[int, str]:
    df = pd.read_csv(csv_path)
    return {int(r.action_id): str(r.action_name) for _, r in df.iterrows()}


class SkeletonSequenceDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict],
        T: int = DEFAULT_T,
        normalize: bool = True,
        cache: bool = False,
    ):
        self.samples = list(samples)
        self.T = T
        self.normalize = normalize
        self.cache = cache
        self._cache: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> np.ndarray:
        return np.array([s["label"] for s in self.samples], dtype=np.int64)

    @property
    def groups(self) -> np.ndarray:
        return np.array([s["user_id"] for s in self.samples], dtype=np.int64)

    def _load_tensor(self, idx: int) -> torch.Tensor:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        s = self.samples[idx]
        seq = load_sequence(Path(s["pred_dir"]))  # (T_raw, 17, 3)
        seq = resample_sequence(seq, self.T)
        if self.normalize:
            seq = normalize_sequence(seq)
        feat = seq.reshape(self.T, FEAT_DIM)  # (T, 51)
        tensor = torch.from_numpy(feat)
        if self.cache:
            self._cache[idx] = tensor
        return tensor

    def __getitem__(self, idx: int):
        x = self._load_tensor(idx)
        y = int(self.samples[idx]["label"])
        user = int(self.samples[idx]["user_id"])
        return x, y, user


def split_by_users(
    samples: Sequence[Dict],
    hold_out_users: Sequence[int] = DEFAULT_HOLD_OUT_USERS,
) -> Tuple[List[Dict], List[Dict]]:
    hold = set(int(u) for u in hold_out_users)
    train = [s for s in samples if s["user_id"] not in hold]
    val = [s for s in samples if s["user_id"] in hold]
    return train, val


def group_kfold_indices(
    samples: Sequence[Dict],
    n_splits: int = 3,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) with GroupKFold by user."""
    from sklearn.model_selection import GroupKFold

    groups = np.array([s["user_id"] for s in samples])
    y = np.array([s["label"] for s in samples])
    X = np.arange(len(samples))
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold has no shuffle; order groups for reproducibility
    folds = list(gkf.split(X, y, groups))
    return folds

def load_train_cache(cache_dir: Path):
    """Load precomputed train.npz + meta. Returns (X, y, users, meta_list)."""
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / "train.npz", allow_pickle=False)
    X = data["X"]
    y = data["y"]
    users = data["users"]
    meta_path = cache_dir / "train_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = [
            {"label": int(y[i]), "user_id": int(users[i]), "pred_dir": None}
            for i in range(len(y))
        ]
    return X, y, users, meta


def load_test_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    data = np.load(cache_dir / "test.npz", allow_pickle=True)
    X = data["X"]
    paths = [str(p) for p in data["paths"].tolist()]
    return X, paths


class CachedSkeletonDataset(Dataset):
    """In-memory / memmap-backed dataset from train.npz indices."""

    def __init__(self, X: np.ndarray, y: np.ndarray, users: np.ndarray, indices=None):
        if indices is None:
            indices = np.arange(len(y))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.X = X
        self.y = y
        self.users = users

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        y = int(self.y[idx])
        user = int(self.users[idx])
        return x, y, user

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.y[self.indices], dtype=np.int64)

    @property
    def groups(self) -> np.ndarray:
        return np.asarray(self.users[self.indices], dtype=np.int64)
