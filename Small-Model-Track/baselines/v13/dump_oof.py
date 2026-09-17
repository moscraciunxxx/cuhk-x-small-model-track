"""Dump OOF + holdout logits from fold/holdout checkpoints for v13 fuse."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
V7 = ROOT.parent / "v7_stgcn"


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # ensure deps resolve
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["stgcn_2s", "ms2s", "mfp"], required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--oof-out", type=str, required=True)
    p.add_argument("--holdout-out", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--key", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(V2))
    # dataset from v2 or v7 - same API; use v7 dataset which has dual
    sys.path.insert(0, str(V7))

    from dataset import (
        DEFAULT_HOLD_OUT_USERS,
        CachedDualDataset,
        load_skel_train_cache,
    )

    if args.kind == "stgcn_2s":
        v7m = load_mod(V7 / "model.py", "v7_model_dump")
        def build(ncls=40):
            return v7m.build_model("stgcn_2s", num_classes=ncls)
        key = args.key or "stgcn_2s"
    elif args.kind == "ms2s":
        ms = load_mod(ROOT / "model_ms_stgcn.py", "v13_ms2s_dump")
        def build(ncls=40):
            return ms.build_model("ms_stgcn_2s", num_classes=ncls)
        key = args.key or "ms_stgcn_2s"
    else:
        mf = load_mod(ROOT / "model.py", "v13_mfp_dump")
        def build(ncls=40):
            return mf.build_model("midfuse_plus", num_classes=ncls)
        key = args.key or "midfuse_wide"

    cache = V2 / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz")
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y)
    users = np.asarray(users)
    n = len(y)
    oof = np.zeros((n, 40), np.float32)
    ckpt_dir = Path(args.ckpt_dir)

    def make_ds(idx):
        return CachedDualDataset(X_skel, X_imu, y, users, idx, has_imu, augment=False, seed=0)

    @torch.no_grad()
    def predict(model, idx):
        ds = make_ds(idx)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        outs = []
        model.eval()
        for xs, xi, _y, _u, flag in loader:
            outs.append(model(xs.to(device), xi.to(device), flag.to(device)).float().cpu().numpy())
        return np.concatenate(outs, 0)

    gkf = GroupKFold(n_splits=5)
    for fi, (tr, va) in enumerate(gkf.split(np.arange(n), y, users)):
        ck = ckpt_dir / f"best_fold{fi}.pt"
        if not ck.exists():
            raise FileNotFoundError(ck)
        blob = torch.load(ck, map_location=device, weights_only=False)
        m = build(blob.get("num_classes", 40)).to(device)
        m.load_state_dict(blob["model_state"])
        oof[va] = predict(m, va)
        print(f"fold{fi} val_acc_ckpt={blob.get('val_acc')} oof_acc={float((oof[va].argmax(1)==y[va]).mean()):.4f}", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh = np.array([int(u) not in hold for u in users])
    print(f"OOF non-holdout={float((oof[nh].argmax(1)==y[nh]).mean()):.4f}", flush=True)
    np.savez_compressed(args.oof_out, **{key: oof, "y": y.astype(np.int64), "users": users.astype(np.int64)})
    print(f"Wrote {args.oof_out}", flush=True)

    hck = ckpt_dir / "best_holdout.pt"
    if hck.exists():
        blob = torch.load(hck, map_location=device, weights_only=False)
        m = build(blob.get("num_classes", 40)).to(device)
        m.load_state_dict(blob["model_state"])
        h_idx = np.where(~nh)[0]
        hl = predict(m, h_idx)
        print(f"holdout ckpt_acc={blob.get('val_acc')} pred_acc={float((hl.argmax(1)==y[h_idx]).mean()):.4f}", flush=True)
        np.savez_compressed(args.holdout_out, **{key: hl, "y": y[h_idx].astype(np.int64), "users": users[h_idx].astype(np.int64)})
        print(f"Wrote {args.holdout_out}", flush=True)
    else:
        print("NO holdout ckpt", flush=True)


if __name__ == "__main__":
    main()
