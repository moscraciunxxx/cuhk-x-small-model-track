"""Build memmap/npz cache of skeleton sequences for fast training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import (
    DEFAULT_T,
    FEAT_DIM,
    discover_test_clips,
    discover_train_samples,
    load_sequence,
    normalize_sequence,
    resample_sequence,
)

ROOT = Path(__file__).resolve().parent


def encode_one(pred_dir: str | None, T: int) -> np.ndarray:
    if pred_dir is None:
        return np.zeros((T, FEAT_DIM), dtype=np.float32)
    seq = load_sequence(Path(pred_dir))
    seq = resample_sequence(seq, T)
    seq = normalize_sequence(seq)
    return seq.reshape(T, FEAT_DIM).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument(
        "--skeleton-root",
        type=str,
        default=r"D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\Skeleton",
    )
    p.add_argument(
        "--test-root",
        type=str,
        default=r"D:\CUHK-X\Small-Model-Track\Testing\data\small_model_track_test",
    )
    p.add_argument("--out-dir", type=str, default=str(ROOT / "cache"))
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = discover_train_samples(Path(args.skeleton_root))
    n = len(samples)
    X = np.zeros((n, args.T, FEAT_DIM), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    users = np.zeros((n,), dtype=np.int64)
    meta = []
    for i, s in enumerate(tqdm(samples, desc="cache-train")):
        X[i] = encode_one(s["pred_dir"], args.T)
        y[i] = s["label"]
        users[i] = s["user_id"]
        meta.append(
            {
                "pred_dir": s["pred_dir"],
                "label": s["label"],
                "user_id": s["user_id"],
                "action_name": s["action_name"],
                "trial": s["trial"],
                "n_frames": s["n_frames"],
            }
        )
    np.savez_compressed(out / "train.npz", X=X, y=y, users=users)
    with open(out / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"Wrote train cache {X.shape} -> {out / 'train.npz'}")

    clips = discover_test_clips(Path(args.test_root))
    Xt = np.zeros((len(clips), args.T, FEAT_DIM), dtype=np.float32)
    paths = []
    for i, c in enumerate(tqdm(clips, desc="cache-test")):
        Xt[i] = encode_one(c["pred_dir"], args.T)
        paths.append(c["path"])
    np.savez_compressed(out / "test.npz", X=Xt, paths=np.array(paths, dtype=object))
    with open(out / "test_paths.json", "w", encoding="utf-8") as f:
        json.dump(paths, f)
    print(f"Wrote test cache {Xt.shape} -> {out / 'test.npz'}")


if __name__ == "__main__":
    main()
