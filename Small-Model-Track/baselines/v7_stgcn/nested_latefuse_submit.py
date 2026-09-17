"""Nested late-fuse with finer alpha grid; submit if OOF clearly beats v3."""
from __future__ import annotations

import json
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
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
from infer import TestDual, predict_logits

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
TRACK = ROOT.parent.parent
V3_OOF = 0.5015


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    X_imu, has_imu, X_imu_te, has_imu_te = load_imu_caches(cache)
    n = len(y)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]

    v2m = load_midfuse_builder()
    mf_dir = V2 / "checkpoints_midfuse_v2b"
    st_dir = ROOT / "checkpoints_cv"

    oof_mf = np.zeros((n, 40), dtype=np.float32)
    oof_st = np.zeros((n, 40), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for fi, (tr, va) in enumerate(gkf.split(np.arange(n), y, users)):
        mf, _ = load_ckpt_model(v2m, mf_dir / f"best_fold{fi}.pt", device, "midfuse")
        st, _ = load_ckpt_model(build_model, st_dir / f"best_fold{fi}.pt", device, "stgcn_fuse")
        ds = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False)
        oof_mf[va] = predict_ds(mf, ds, device)
        oof_st[va] = predict_ds(st, ds, device)
        del mf, st
        torch.cuda.empty_cache()

    alphas = [round(x, 2) for x in np.linspace(0.25, 0.60, 15)]
    yt_nh = y[nh_idx]
    Sm, Ss = softmax(oof_mf[nh_idx]), softmax(oof_st[nh_idx])
    best = {"alpha": 0.5, "oof_acc": -1.0}
    rows = []
    for a in alphas:
        acc = float(((a * Ss + (1 - a) * Sm).argmax(1) == yt_nh).mean())
        rows.append({"alpha_stgcn": a, "oof_acc": acc})
        if acc > best["oof_acc"]:
            best = {"alpha": a, "oof_acc": acc}

    mf_h, _ = load_ckpt_model(v2m, mf_dir / "best_holdout.pt", device, "midfuse")
    st_h, _ = load_ckpt_model(build_model, ROOT / "checkpoints_v2" / "best_holdout.pt", device, "stgcn_fuse")
    ds_h = CachedDualDataset(X_skel, X_imu, y, users, ho_idx, has_imu, augment=False)
    Lmf, Lst = predict_ds(mf_h, ds_h, device), predict_ds(st_h, ds_h, device)
    yt_h = y[ho_idx]
    a = best["alpha"]
    hold_acc = float(((a * softmax(Lst) + (1 - a) * softmax(Lmf)).argmax(1) == yt_h).mean())
    hold_mf = float((Lmf.argmax(1) == yt_h).mean())
    hold_st = float((Lst.argmax(1) == yt_h).mean())

    oof_mf_acc = float((oof_mf[nh_idx].argmax(1) == yt_nh).mean())
    oof_st_acc = float((oof_st[nh_idx].argmax(1) == yt_nh).mean())
    agree = float((oof_mf[nh_idx].argmax(1) == oof_st[nh_idx].argmax(1)).mean())

    clearly_better_oof = best["oof_acc"] >= V3_OOF + 0.01
    clear_holdout = hold_acc >= 0.547
    ping = bool(clear_holdout or clearly_better_oof or best["oof_acc"] > 0.505 + 0.01)

    out = {
        "oof_rows": rows,
        "selected_alpha_stgcn": a,
        "oof_acc_selected": best["oof_acc"],
        "oof_midfuse": oof_mf_acc,
        "oof_stgcn": oof_st_acc,
        "oof_agreement": agree,
        "v3_oof_baseline": V3_OOF,
        "oof_delta_vs_v3": best["oof_acc"] - V3_OOF,
        "holdout_latefuse_acc": hold_acc,
        "holdout_midfuse": hold_mf,
        "holdout_stgcn": hold_st,
        "holdout_agreement": float((Lmf.argmax(1) == Lst.argmax(1)).mean()),
        "clear_holdout_win": clear_holdout,
        "clearly_better_oof": clearly_better_oof,
        "ping_disk_saver": ping,
        "overwrite_submission": ping,
    }
    (ROOT / "nested_latefuse_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)

    # Always write v7 submission artifact; overwrite track only if ping
    X_te, paths = load_skel_test_cache(cache)
    probs = None
    for fi in range(5):
        mf, _ = load_ckpt_model(v2m, mf_dir / f"best_fold{fi}.pt", device, "midfuse")
        st, _ = load_ckpt_model(build_model, st_dir / f"best_fold{fi}.pt", device, "stgcn_fuse")
        ds = TestDual(X_te, X_imu_te, has_imu_te if has_imu_te is not None else np.ones(len(X_te), bool))
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        pm = softmax(predict_logits(mf, loader, device, True))
        ps = softmax(predict_logits(st, loader, device, True))
        mix = a * ps + (1 - a) * pm
        probs = mix if probs is None else probs + mix
        del mf, st
        torch.cuda.empty_cache()
    probs /= 5.0
    pred = probs.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub_v7 = ROOT / "submission_v7_latefuse.csv"
    df.to_csv(sub_v7, index=False)
    np.savez_compressed(ROOT / "submission_v7_latefuse_probs.npz", probs=probs, paths=np.array(paths, dtype=object))
    print(f"Wrote {sub_v7}", flush=True)
    if ping:
        track_sub = TRACK / "submission.csv"
        df.to_csv(track_sub, index=False)
        (TRACK / "submissions" / "submission_v7_latefuse.csv").parent.mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v7_latefuse.csv", index=False)
        print(f"OVERWROTE {track_sub}", flush=True)
    else:
        print("Did NOT overwrite track submission.csv", flush=True)


if __name__ == "__main__":
    main()
