"""Dataset + clip index for Depth_Color / Thermal temporal CNN."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

TRACK_ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
TRAIN_ROOT = TRACK_ROOT / "Training" / "data" / "HAR" / "data"
TEST_ROOT = TRACK_ROOT / "Testing" / "data" / "small_model_track_test"
DEFAULT_HOLD_OUT_USERS = (8, 9, 24)
NUM_CLASSES = 40


def list_frame_paths(clip_dir: Path, modality: str) -> list[Path]:
    if modality.lower() in ("depth_color", "depth"):
        frames = sorted(clip_dir.glob("*.png")) + sorted(clip_dir.glob("*.jpg"))
        # prefer png for Depth_Color
        frames = sorted(set(frames), key=lambda p: p.name)
    else:
        frames = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("frame_*.jpg"))
        frames = sorted(set(frames), key=lambda p: p.name)
    return frames


def sample_indices(n: int, t: int, train: bool, rng: random.Random | None = None) -> np.ndarray:
    if n <= 0:
        return np.zeros(t, dtype=np.int64)
    if n == 1:
        return np.zeros(t, dtype=np.int64)
    if train and rng is not None and n > t:
        # random contiguous-ish jittered uniform
        start = rng.random() * 0.15
        end = 1.0 - rng.random() * 0.15
        if end <= start + 0.2:
            start, end = 0.0, 1.0
        xs = np.linspace(start, end, t)
        xs = xs + (rng.random() - 0.5) * (0.5 / t)
        xs = np.clip(xs, 0.0, 1.0)
        idx = (xs * (n - 1)).astype(np.int64)
    else:
        idx = np.linspace(0, max(n - 1, 0), t).astype(np.int64)
    return idx


def load_clip_frames(
    frame_paths: list[Path],
    t: int,
    size: int,
    train: bool,
    in_ch: int = 3,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Return (T,C,H,W) float32 in [0,1]."""
    n = len(frame_paths)
    if n == 0:
        return np.zeros((t, in_ch, size, size), dtype=np.float32)
    idxs = sample_indices(n, t, train=train, rng=rng)
    # augment geometry once per clip
    do_flip = bool(train and rng is not None and rng.random() < 0.5)
    # random resized crop params
    if train and rng is not None:
        scale = 0.75 + 0.25 * rng.random()
        crop = int(size / scale)
        # we'll crop on original then resize
    else:
        crop = None

    out = np.zeros((t, in_ch, size, size), dtype=np.float32)
    for i, fi in enumerate(idxs):
        p = frame_paths[int(fi)]
        try:
            img = Image.open(p)
            if in_ch == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
        except Exception:
            continue
        w, h = img.size
        if train and rng is not None and crop is not None and min(w, h) > 16:
            side = min(w, h)
            # random square crop then resize
            cw = max(int(side * (0.7 + 0.3 * rng.random())), size)
            ch = cw
            left = rng.randint(0, max(w - cw, 0))
            top = rng.randint(0, max(h - ch, 0))
            img = img.crop((left, top, left + cw, top + ch))
        # center crop square for val
        if not train:
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.BILINEAR)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if in_ch == 1:
            if arr.ndim == 2:
                arr = arr[None, ...]
            else:
                arr = arr.mean(axis=2, keepdims=False)[None, ...]
        else:
            arr = arr.transpose(2, 0, 1)  # C,H,W
            if train and rng is not None:
                # mild brightness/contrast
                b = 0.85 + 0.3 * rng.random()
                c = 0.85 + 0.3 * rng.random()
                arr = np.clip((arr - 0.5) * c + 0.5 * b, 0.0, 1.0)
        out[i] = arr[:in_ch]
    return out


def discover_train_clips(modality: str = "Depth_Color") -> list[dict[str, Any]]:
    root = TRAIN_ROOT / modality
    clips: list[dict[str, Any]] = []
    if not root.exists():
        raise FileNotFoundError(root)
    for act in sorted(root.iterdir()):
        if not act.is_dir():
            continue
        try:
            aid = int(act.name.split("_")[0])
        except ValueError:
            continue
        for user in sorted(act.iterdir()):
            if not user.is_dir() or not user.name.startswith("user"):
                continue
            uid = int(user.name.replace("user", ""))
            for trial in sorted(user.iterdir()):
                if not trial.is_dir():
                    continue
                frames = list_frame_paths(trial, modality)
                if len(frames) == 0:
                    continue
                clips.append(
                    {
                        "clip_dir": str(trial),
                        "label": aid,
                        "user_id": uid,
                        "action_name": act.name,
                        "trial": trial.name,
                        "n_frames": len(frames),
                        "modality": modality,
                    }
                )
    return clips


def discover_test_clips(modality: str = "Depth_Color") -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for s in sorted(TEST_ROOT.glob("SM_test_*")):
        d = s / modality
        frames = list_frame_paths(d, modality) if d.exists() else []
        rel = f"small_model_track_test/{s.name}/"
        clips.append(
            {
                "sample_id": s.name,
                "path": rel,
                "clip_dir": str(d),
                "n_frames": len(frames),
                "empty": len(frames) == 0,
                "modality": modality,
            }
        )
    return clips


class LiveClipDataset(Dataset):
    def __init__(
        self,
        clips: list[dict[str, Any]],
        t: int = 8,
        size: int = 112,
        train: bool = False,
        in_ch: int = 3,
        seed: int = 0,
    ):
        self.clips = clips
        self.t = t
        self.size = size
        self.train = train
        self.in_ch = in_ch
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        meta = self.clips[idx]
        rng = random.Random(self.seed + idx * 10007 + (random.randint(0, 10**6) if self.train else 0))
        frames = list_frame_paths(Path(meta["clip_dir"]), meta.get("modality", "Depth_Color"))
        x = load_clip_frames(frames, self.t, self.size, train=self.train, in_ch=self.in_ch, rng=rng)
        x = torch.from_numpy(x)  # T,C,H,W
        y = int(meta.get("label", -1))
        uid = int(meta.get("user_id", -1))
        return x, y, uid, idx


class CachedClipDataset(Dataset):
    """Reads pre-resized uint8 cache (N,T,H,W,C) + labels/users."""

    def __init__(
        self,
        cache_x: np.ndarray,
        labels: np.ndarray,
        users: np.ndarray,
        indices: np.ndarray | list[int],
        train: bool = False,
        seed: int = 0,
    ):
        self.x = cache_x
        self.labels = labels
        self.users = users
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        arr = self.x[idx]  # T,H,W,C uint8
        t, h, w, c = arr.shape
        if self.train:
            rng = random.Random(self.seed + idx * 13 + random.randint(0, 10**9))
            # temporal crop/jitter: pick T' contiguous-ish subset if T>=8 keep all with shuffle offsets
            if rng.random() < 0.5 and t >= 4:
                # reverse temporal sometimes
                if rng.random() < 0.15:
                    arr = arr[::-1].copy()
            do_flip = rng.random() < 0.5
            # spatial: random crop 87.5% then resize back via indexing
            if rng.random() < 0.7:
                ch = int(h * (0.8 + 0.2 * rng.random()))
                cw = int(w * (0.8 + 0.2 * rng.random()))
                top = rng.randint(0, max(h - ch, 0))
                left = rng.randint(0, max(w - cw, 0))
                crop = arr[:, top : top + ch, left : left + cw, :]
                # nearest resize back
                ys = (np.linspace(0, crop.shape[1] - 1, h)).astype(np.int64)
                xs = (np.linspace(0, crop.shape[2] - 1, w)).astype(np.int64)
                arr = crop[:, ys][:, :, xs]
            if do_flip:
                arr = arr[:, :, ::-1, :].copy()
            x = arr.astype(np.float32) / 255.0
            b = 0.85 + 0.3 * rng.random()
            contrast = 0.85 + 0.3 * rng.random()
            x = np.clip((x - 0.5) * contrast + 0.5 * b, 0.0, 1.0)
        else:
            x = arr.astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))  # T,C,H,W
        y = int(self.labels[idx])
        uid = int(self.users[idx])
        return x, y, uid, idx


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
