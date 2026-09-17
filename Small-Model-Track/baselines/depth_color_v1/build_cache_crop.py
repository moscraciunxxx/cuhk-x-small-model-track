"""Rebuild frame cache with motion/brightness person crops."""
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


def person_box_from_frames(arrays: list[np.ndarray], modality: str):
    """arrays: list of HxWxC uint8. Return (left, top, right, bottom) in pixel coords of first frame size."""
    if not arrays:
        return None
    h, w = arrays[0].shape[:2]
    grays = []
    for a in arrays:
        if a.ndim == 3:
            g = a.mean(axis=2)
        else:
            g = a
        grays.append(g.astype(np.float32))
    stack = np.stack(grays, axis=0)  # T,H,W
    if modality.lower().startswith("thermal"):
        score = stack.mean(axis=0)
        thr = np.percentile(score, 65)
        mask = score >= thr
    else:
        if len(grays) >= 2:
            motion = np.abs(np.diff(stack, axis=0)).mean(axis=0)
        else:
            motion = stack[0]
        # also emphasize mid intensities (person often not pure black/white bg)
        score = motion
        thr = np.percentile(score, 70)
        mask = score >= thr
    ys, xs = np.where(mask)
    if len(xs) < 20:
        # fallback center square
        side = int(min(h, w) * 0.75)
        left = (w - side) // 2
        top = (h - side) // 2
        return left, top, left + side, top + side
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    # pad
    bw, bh = x1 - x0, y1 - y0
    pad_x = int(0.15 * bw) + 2
    pad_y = int(0.15 * bh) + 2
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    # make square
    bw, bh = x1 - x0, y1 - y0
    side = max(bw, bh)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = side // 2
    left = max(0, min(w - side, cx - half))
    top = max(0, min(h - side, cy - half))
    side = min(side, w - left, h - top)
    return left, top, left + side, top + side


def load_raw_sampled(frame_paths, t: int):
    n = len(frame_paths)
    if n == 0:
        return []
    idxs = sample_indices(n, t, train=False)
    out = []
    for fi in idxs:
        try:
            img = Image.open(frame_paths[int(fi)]).convert("RGB")
            out.append(np.asarray(img, dtype=np.uint8))
        except Exception:
            continue
    return out


def resize_clip_cropped(frame_paths, t: int, size: int, modality: str) -> np.ndarray:
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    arrays = load_raw_sampled(frame_paths, t)
    if not arrays:
        return out
    box = person_box_from_frames(arrays, modality)
    for i, arr in enumerate(arrays[:t]):
        if box is not None:
            l, t0, r, b = box
            arr = arr[t0:b, l:r]
        img = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(img, dtype=np.uint8)
    # if fewer than t, pad by repeating last
    if 0 < len(arrays) < t:
        for i in range(len(arrays), t):
            out[i] = out[len(arrays) - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color", choices=["Depth_Color", "Thermal"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--tag", type=str, default="crop")
    args = ap.parse_args()

    out_dir = ROOT / "cache" / f"{args.modality.lower()}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_clips = discover_train_clips(args.modality)
    test_clips = discover_test_clips(args.modality)
    print(f"train={len(train_clips)} test={len(test_clips)} -> {out_dir}")

    N = len(train_clips)
    x_path = out_dir / f"train_x_t{args.t}_s{args.size}.npy"
    if not x_path.exists():
        X = np.memmap(x_path, dtype=np.uint8, mode="w+", shape=(N, args.t, args.size, args.size, 3))
        y = np.zeros(N, dtype=np.int64)
        users = np.zeros(N, dtype=np.int64)
        for i, c in enumerate(tqdm(train_clips, desc="crop-train")):
            frames = list_frame_paths(Path(c["clip_dir"]), args.modality)
            X[i] = resize_clip_cropped(frames, args.t, args.size, args.modality)
            y[i] = c["label"]
            users[i] = c["user_id"]
            if i % 100 == 0:
                X.flush()
        X.flush()
        np.save(out_dir / "train_y.npy", y)
        np.save(out_dir / "train_users.npy", users)
        save_json(out_dir / "train_meta.json", train_clips)
    else:
        print("train exists")

    Nt = len(test_clips)
    tx = out_dir / f"test_x_t{args.t}_s{args.size}.npy"
    if not tx.exists():
        X = np.memmap(tx, dtype=np.uint8, mode="w+", shape=(Nt, args.t, args.size, args.size, 3))
        empty = []
        for i, c in enumerate(tqdm(test_clips, desc="crop-test")):
            frames = list_frame_paths(Path(c["clip_dir"]), args.modality) if not c["empty"] else []
            X[i] = resize_clip_cropped(frames, args.t, args.size, args.modality)
            if c["empty"] or len(frames) == 0:
                empty.append(c["sample_id"])
            if i % 50 == 0:
                X.flush()
        X.flush()
        save_json(out_dir / "test_empty.json", empty)
        save_json(out_dir / "test_meta.json", test_clips)
        print("empty", len(empty))
    print("done holdout users", DEFAULT_HOLD_OUT_USERS)


if __name__ == "__main__":
    main()
