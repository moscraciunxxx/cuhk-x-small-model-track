"""Build uint8 Depth+IR frame cache (K frames, HxW) aligned to skeleton train_meta / test clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
TRAIN_DC = ROOT / "Training" / "data" / "HAR" / "data" / "Depth_Color"
TRAIN_IR = ROOT / "Training" / "data" / "HAR" / "data" / "IR"
TEST_ROOT = ROOT / "Testing" / "data" / "small_model_track_test"
META = ROOT / "baselines" / "skeleton_imu_v2" / "cache" / "train_meta.json"
OUT = ROOT / "baselines" / "v5" / "cache"


def sample_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(k, dtype=np.int64)
    if n == 1:
        return np.zeros(k, dtype=np.int64)
    return np.linspace(0, n - 1, num=k).round().astype(np.int64)


def list_pngs(d: Path):
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])


def load_gray(path: Path, size_hw) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    h, w = size_hw
    im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_clip(dc_dir: Path, ir_dir: Path, k: int, size_hw):
    dc = list_pngs(dc_dir)
    ir = list_pngs(ir_dir)
    n = min(len(dc), len(ir))
    out = np.zeros((2, k, size_hw[0], size_hw[1]), dtype=np.uint8)
    if n == 0:
        return out, False
    idx = sample_indices(n, k)
    for t, i in enumerate(idx):
        out[0, t] = load_gray(dc[int(i)], size_hw)
        out[1, t] = load_gray(ir[int(i)], size_hw)
    return out, True


def build_train(k: int, h: int, w: int, users_filter=None):
    meta = json.loads(META.read_text(encoding="utf-8"))
    if users_filter is not None:
        users_filter = set(users_filter)
        meta_idx = [i for i, m in enumerate(meta) if m["user_id"] in users_filter]
    else:
        meta_idx = list(range(len(meta)))

    n = len(meta_idx)
    X = np.zeros((n, 2, k, h, w), dtype=np.uint8)
    y = np.zeros(n, dtype=np.int64)
    users = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=np.bool_)
    orig = np.zeros(n, dtype=np.int64)

    for j, i in enumerate(meta_idx):
        m = meta[i]
        action, uid, trial = m["action_name"], m["user_id"], m["trial"]
        dc = TRAIN_DC / action / f"user{uid}" / trial
        ir = TRAIN_IR / action / f"user{uid}" / trial
        clip, good = load_clip(dc, ir, k, (h, w))
        X[j] = clip
        y[j] = int(m["label"])
        users[j] = int(uid)
        ok[j] = good
        orig[j] = i
        if (j + 1) % 200 == 0 or j + 1 == n:
            print(f"train cache {j+1}/{n}", flush=True)
    return X, y, users, ok, orig


def build_test(k: int, h: int, w: int):
    ids = sorted([p.name for p in TEST_ROOT.iterdir() if p.is_dir() and p.name.startswith("SM_test_")])
    n = len(ids)
    X = np.zeros((n, 2, k, h, w), dtype=np.uint8)
    ok = np.zeros(n, dtype=np.bool_)
    paths = []
    for j, tid in enumerate(ids):
        dc = TEST_ROOT / tid / "Depth_Color"
        ir = TEST_ROOT / tid / "IR"
        clip, good = load_clip(dc, ir, k, (h, w))
        X[j] = clip
        ok[j] = good
        paths.append(f"small_model_track_test/{tid}/")
        if (j + 1) % 100 == 0 or j + 1 == n:
            print(f"test cache {j+1}/{n}", flush=True)
    return X, ok, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--holdout-only", action="store_true", help="Only cache users 8,9,24 + their complement later")
    ap.add_argument("--split", choices=["all", "holdout_users", "train_users", "test"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.split == "test":
        X, ok, paths = build_test(args.k, args.h, args.w)
        np.savez_compressed(OUT / "vision_test.npz", X=X, ok=ok, paths=np.array(paths))
        print("saved vision_test", X.shape, "ok", ok.sum())
        return

    hold = {8, 9, 24}
    if args.split == "holdout_users":
        X, y, users, ok, orig = build_train(args.k, args.h, args.w, users_filter=hold)
        np.savez_compressed(OUT / "vision_holdout.npz", X=X, y=y, users=users, ok=ok, orig_idx=orig)
        print("saved vision_holdout", X.shape, "ok", ok.sum())
    elif args.split == "train_users":
        # all except holdout
        meta = json.loads(META.read_text(encoding="utf-8"))
        train_users = sorted({m["user_id"] for m in meta} - hold)
        X, y, users, ok, orig = build_train(args.k, args.h, args.w, users_filter=train_users)
        np.savez_compressed(OUT / "vision_trainusers.npz", X=X, y=y, users=users, ok=ok, orig_idx=orig)
        print("saved vision_trainusers", X.shape, "ok", ok.sum())
    else:
        X, y, users, ok, orig = build_train(args.k, args.h, args.w, users_filter=None)
        np.savez_compressed(OUT / "vision_train.npz", X=X, y=y, users=users, ok=ok, orig_idx=orig)
        print("saved vision_train", X.shape, "ok", ok.sum())


if __name__ == "__main__":
    main()
