"""Build resized frame cache for fast training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    discover_test_clips,
    discover_train_clips,
    list_frame_paths,
    sample_indices,
    save_json,
)

ROOT = Path(__file__).resolve().parent


def resize_clip(frame_paths, t: int, size: int, in_ch: int) -> np.ndarray:
    n = len(frame_paths)
    out = np.zeros((t, size, size, in_ch), dtype=np.uint8)
    if n == 0:
        return out
    idxs = sample_indices(n, t, train=False)
    for i, fi in enumerate(idxs):
        try:
            img = Image.open(frame_paths[int(fi)])
            img = img.convert("L" if in_ch == 1 else "RGB")
        except Exception:
            continue
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if in_ch == 1:
            if arr.ndim == 3:
                arr = arr.mean(axis=2).astype(np.uint8)
            out[i, :, :, 0] = arr
        else:
            out[i] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color", choices=["Depth_Color", "Thermal"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "cache" / args.modality.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_clips = discover_train_clips(args.modality)
    test_clips = discover_test_clips(args.modality)
    print(f"train clips={len(train_clips)} test={len(test_clips)} modality={args.modality}")

    N = len(train_clips)
    x_path = out_dir / f"train_x_t{args.t}_s{args.size}.npy"
    y_path = out_dir / "train_y.npy"
    u_path = out_dir / "train_users.npy"
    meta_path = out_dir / "train_meta.json"

    if x_path.exists() and y_path.exists():
        print("train cache exists:", x_path)
    else:
        X = np.memmap(x_path, dtype=np.uint8, mode="w+", shape=(N, args.t, args.size, args.size, args.in_ch))
        y = np.zeros(N, dtype=np.int64)
        users = np.zeros(N, dtype=np.int64)
        for i, c in enumerate(tqdm(train_clips, desc="cache-train")):
            frames = list_frame_paths(Path(c["clip_dir"]), args.modality)
            X[i] = resize_clip(frames, args.t, args.size, args.in_ch)
            y[i] = c["label"]
            users[i] = c["user_id"]
            if i % 200 == 0:
                X.flush()
        X.flush()
        np.save(y_path, y)
        np.save(u_path, users)
        save_json(meta_path, train_clips)
        print("saved", x_path, X.shape)

    # test cache
    Nt = len(test_clips)
    tx_path = out_dir / f"test_x_t{args.t}_s{args.size}.npy"
    tempty = out_dir / "test_empty.json"
    tmeta = out_dir / "test_meta.json"
    if tx_path.exists():
        print("test cache exists:", tx_path)
    else:
        X = np.memmap(tx_path, dtype=np.uint8, mode="w+", shape=(Nt, args.t, args.size, args.size, args.in_ch))
        empty = []
        for i, c in enumerate(tqdm(test_clips, desc="cache-test")):
            frames = list_frame_paths(Path(c["clip_dir"]), args.modality) if not c["empty"] else []
            X[i] = resize_clip(frames, args.t, args.size, args.in_ch)
            if c["empty"] or len(frames) == 0:
                empty.append(c["sample_id"])
            if i % 50 == 0:
                X.flush()
        X.flush()
        save_json(tempty, empty)
        save_json(tmeta, test_clips)
        print("saved", tx_path, "empty", len(empty))

    hold = [c for c in train_clips if c["user_id"] in set(DEFAULT_HOLD_OUT_USERS)]
    print(f"holdout clips={len(hold)} users={DEFAULT_HOLD_OUT_USERS}")


if __name__ == "__main__":
    main()
