"""Nested late-fuse: select alpha on MidFuse/ST-GCN OOF (non-holdout), eval holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_imu_caches,
    load_skel_train_cache,
    load_skel_test_cache,
)
from model import build_model
import importlib.util

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
TRACK = ROOT.parent.parent


def load_midfuse_builder():
    spec = importlib.util.spec_from_file_location("v2_model", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt_model(builder, ckpt_path, device, default_name="midfuse"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ck.get("model_name", default_name)
    if builder is build_model:
        m = builder(name, num_classes=ck.get("num_classes", 40))
    else:
        m = builder.build_model(name, num_classes=ck.get("num_classes", 40))
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m, ck


@torch.no_grad()
def predict_idx(model, X_skel, X_imu, has_imu, indices, device, batch=64):
    ds = CachedDualDataset(X_skel, X_imu, np.zeros(len(X_skel), dtype=np.int64), np.zeros(len(X_skel), dtype=np.int64), indices, has_imu, augment=False)
    # CachedDualDataset needs y/users full arrays — pass real below in caller
    raise NotImplementedError


@torch.no_grad()
def predict_ds(model, ds, device, batch=64):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    outs = []
    for xs, xi, y, _u, flag in loader:
        outs.append(model(xs.to(device), xi.to(device), flag.to(device)).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    p.add_argument("--stgcn-ckpt-dir", type=str, default=str(ROOT / "checkpoints_cv"))
    p.add_argument("--midfuse-ckpt-dir", type=str, default=str(V2 / "checkpoints_midfuse_v2b"))
    p.add_argument("--stgcn-holdout-ckpt", type=str, default=str(ROOT / "checkpoints_v2" / "best_holdout.pt"))
    p.add_argument("--midfuse-holdout-ckpt", type=str, default=str(V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--out", type=str, default=str(ROOT / "nested_latefuse.json"))
    p.add_argument("--write-submission", action="store_true")
    p.add_argument("--submission-out", type=str, default=str(ROOT / "submission_v7_latefuse.csv"))
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache_dir)
    X_skel, y, users, _ = load_skel_train_cache(cache)
    X_imu, has_imu, X_imu_te, has_imu_te = load_imu_caches(cache)
    n = len(y)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nonhold_mask = ~np.isin(users, list(hold))
    nh_idx = np.where(nonhold_mask)[0]
    ho_idx = np.where(~nonhold_mask)[0]

    v2m = load_midfuse_builder()
    mf_dir = Path(args.midfuse_ckpt_dir)
    st_dir = Path(args.stgcn_ckpt_dir)

    # --- OOF on non-holdout via GroupKFold matching train.py ---
    gkf = GroupKFold(n_splits=args.cv_splits)
    oof_mf = np.zeros((n, 40), dtype=np.float32)
    oof_st = np.zeros((n, 40), dtype=np.float32)
    oof_filled = np.zeros(n, dtype=bool)

    for fi, (tr, va) in enumerate(gkf.split(np.arange(n), y, users)):
        # Only score val users that are non-holdout (GroupKFold uses all users;
        # MidFuse folds were trained on all users similarly)
        mf_ckpt = mf_dir / f"best_fold{fi}.pt"
        st_ckpt = st_dir / f"best_fold{fi}.pt"
        if not mf_ckpt.exists():
            raise FileNotFoundError(mf_ckpt)
        if not st_ckpt.exists():
            raise FileNotFoundError(f"Missing ST-GCN fold ckpt {st_ckpt}; run CV first")
        mf, _ = load_ckpt_model(v2m, mf_ckpt, device, "midfuse")
        st, _ = load_ckpt_model(build_model, st_ckpt, device, "stgcn_fuse")
        ds = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False)
        oof_mf[va] = predict_ds(mf, ds, device)
        oof_st[va] = predict_ds(st, ds, device)
        oof_filled[va] = True
        del mf, st
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert oof_filled.all()
    # Select alpha on NON-HOLDOUT OOF only
    nh = nh_idx
    yt_nh = y[nh]
    Sm = softmax(oof_mf[nh])
    Ss = softmax(oof_st[nh])
    best = {"alpha": 0.3, "oof_acc": -1.0}
    rows = []
    for a in args.alphas:
        pred = (a * Ss + (1 - a) * Sm).argmax(1)
        acc = float((pred == yt_nh).mean())
        rows.append({"alpha_stgcn": a, "oof_acc": acc})
        if acc > best["oof_acc"]:
            best = {"alpha": a, "oof_acc": acc}
    mf_oof_acc = float((oof_mf[nh].argmax(1) == yt_nh).mean())
    st_oof_acc = float((oof_st[nh].argmax(1) == yt_nh).mean())
    agree_oof = float((oof_mf[nh].argmax(1) == oof_st[nh].argmax(1)).mean())

    # Holdout eval with selected alpha using holdout-trained ckpts
    mf_h, _ = load_ckpt_model(v2m, Path(args.midfuse_holdout_ckpt), device, "midfuse")
    st_h, _ = load_ckpt_model(build_model, Path(args.stgcn_holdout_ckpt), device, "stgcn_fuse")
    ds_h = CachedDualDataset(X_skel, X_imu, y, users, ho_idx, has_imu, augment=False)
    Lmf = predict_ds(mf_h, ds_h, device)
    Lst = predict_ds(st_h, ds_h, device)
    yt_h = y[ho_idx]
    a = best["alpha"]
    mix = a * softmax(Lst) + (1 - a) * softmax(Lmf)
    hold_acc = float((mix.argmax(1) == yt_h).mean())
    hold_mf = float((Lmf.argmax(1) == yt_h).mean())
    hold_st = float((Lst.argmax(1) == yt_h).mean())
    agree_h = float((Lmf.argmax(1) == Lst.argmax(1)).mean())

    # Peeked alphas for reference only
    peek = []
    for aa in args.alphas:
        acc = float(((aa * softmax(Lst) + (1 - aa) * softmax(Lmf)).argmax(1) == yt_h).mean())
        peek.append({"alpha_stgcn": aa, "holdout_acc_peek": acc})

    out = {
        "oof_rows": rows,
        "selected_alpha_stgcn": best["alpha"],
        "oof_acc_selected": best["oof_acc"],
        "oof_midfuse": mf_oof_acc,
        "oof_stgcn": st_oof_acc,
        "oof_agreement": agree_oof,
        "holdout_latefuse_acc": hold_acc,
        "holdout_midfuse": hold_mf,
        "holdout_stgcn": hold_st,
        "holdout_agreement": agree_h,
        "holdout_peek_alphas": peek,
        "delta_vs_midfuse_holdout": hold_acc - hold_mf,
        "clear_win": hold_acc >= 0.547,
        "baseline_midfuse": 0.5366336633663367,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)

    if args.write_submission and out["clear_win"]:
        # Test: ensemble MidFuse fold softmax + ST-GCN fold/all softmax
        X_te, paths = load_skel_test_cache(cache)
        from infer import TestDual, predict_logits

        def test_logits(model):
            ds = TestDual(X_te, X_imu_te, has_imu_te if has_imu_te is not None else np.ones(len(X_te), bool))
            loader = DataLoader(ds, batch_size=64, shuffle=False)
            return predict_logits(model, loader, device, dual=True)

        # Use fold ensemble for both (matches v3 style) + selected alpha
        probs = None
        for fi in range(args.cv_splits):
            mf, _ = load_ckpt_model(v2m, mf_dir / f"best_fold{fi}.pt", device, "midfuse")
            st, _ = load_ckpt_model(build_model, st_dir / f"best_fold{fi}.pt", device, "stgcn_fuse")
            pm = softmax(test_logits(mf))
            ps = softmax(test_logits(st))
            mix_p = a * ps + (1 - a) * pm
            probs = mix_p if probs is None else probs + mix_p
            del mf, st
        probs /= args.cv_splits
        pred = probs.argmax(1)
        import pandas as pd

        df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
        outp = Path(args.submission_out)
        df.to_csv(outp, index=False)
        print(f"Wrote submission {outp}", flush=True)
        # Do NOT overwrite track submission here — parent/gate decides


if __name__ == "__main__":
    main()
