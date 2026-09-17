"""v9: 4-branch late-fuse + TTA + better conf/entropy weighting.

Branches: MidFuse + ST-GCN + GRU-attn + DeepConv
Protocol: GroupKFold OOF; nested hyperparam selection on NON-holdout only;
holdout evaluated LAST. No holdout peek for selection.

Win vs v8: nested OOF >= 0.548 OR holdout >= 0.566
Then overwrite track submission.csv and ping_disk_saver=true.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

V7 = Path(__file__).resolve().parent.parent / "v7_stgcn"
V2 = Path(__file__).resolve().parent.parent / "skeleton_imu_v2"
V8 = Path(__file__).resolve().parent.parent / "v8"
ROOT = Path(__file__).resolve().parent
TRACK = ROOT.parent.parent
sys.path.insert(0, str(V7))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_imu_caches,
    load_skel_train_cache,
    load_skel_test_cache,
)
from model import build_model as build_stgcn  # noqa: E402
from infer import TestDual, predict_logits  # noqa: E402

V8_OOF = 0.5383347073371806
V8_HOLD = 0.5564356435643565
NUM_CLASSES = 40
NUM_JOINTS = 17

# COCO17 L/R swap pairs
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v9", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(builder_kind: str, ckpt_path: Path, device, v2m, default_name: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ck.get("model_name", default_name)
    ncls = ck.get("num_classes", NUM_CLASSES)
    if builder_kind == "stgcn":
        m = build_stgcn(name, num_classes=ncls)
    else:
        m = v2m.build_model(name, num_classes=ncls)
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m, ck


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


def flip_lr_skel_np(x: np.ndarray) -> np.ndarray:
    """x: (..., T, 51) or (T, 51) -> left-right flip (swap joints, negate X)."""
    x = np.asarray(x, dtype=np.float32).copy()
    shape = x.shape
    x = x.reshape(*shape[:-1], NUM_JOINTS, 3)
    for a, b in LR_PAIRS:
        tmp = x[..., a, :].copy()
        x[..., a, :] = x[..., b, :]
        x[..., b, :] = tmp
    x[..., 0] *= -1.0  # negate X
    return x.reshape(shape)


def flip_lr_imu_np(x: np.ndarray) -> np.ndarray:
    """x: (..., T, 30) IMU device order WTLA,WTRA,WTC,WTLL,WTRL; 6 feats each."""
    x = np.asarray(x, dtype=np.float32).copy()
    # swap L/R arms (0 <-> 1), L/R legs (3 <-> 4)
    for i, j in ((0, 1), (3, 4)):
        a, b = i * 6, j * 6
        tmp = x[..., a : a + 6].copy()
        x[..., a : a + 6] = x[..., b : b + 6]
        x[..., b : b + 6] = tmp
    # negate ax,gx within each device (indices 0,3 of each 6)
    for d in range(5):
        base = d * 6
        x[..., base + 0] *= -1.0
        x[..., base + 3] *= -1.0
    return x


def temporal_shift_np(x: np.ndarray, shift: int) -> np.ndarray:
    """Roll along time axis (-2)."""
    return np.roll(x, shift=shift, axis=-2)


class ArrayDual(Dataset):
    def __init__(self, xs, xi, flag):
        self.xs = torch.from_numpy(np.asarray(xs, dtype=np.float32))
        self.xi = torch.from_numpy(np.asarray(xi, dtype=np.float32))
        self.flag = torch.from_numpy(np.asarray(flag, dtype=np.float32))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], self.xi[i], self.flag[i]


@torch.no_grad()
def predict_logits_arr(model, xs, xi, flag, device, batch=64):
    ds = ArrayDual(xs, xi, flag)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    outs = []
    for xb, ib, fb in loader:
        outs.append(model(xb.to(device), ib.to(device), fb.to(device)).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def predict_ds(model, ds, device, batch=64):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    outs = []
    for xs, xi, y, _u, flag in loader:
        outs.append(model(xs.to(device), xi.to(device), flag.to(device)).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def predict_with_tta(model, xs, xi, flag, device, use_tta=True, batch=64):
    """Average logits over identity, LR-flip, and small temporal shifts."""
    views = [(xs, xi)]
    if use_tta:
        views.append((flip_lr_skel_np(xs), flip_lr_imu_np(xi)))
        views.append((temporal_shift_np(xs, 4), temporal_shift_np(xi, 4)))
        views.append((temporal_shift_np(xs, -4), temporal_shift_np(xi, -4)))
    acc_logits = None
    for vx, vi in views:
        logits = predict_logits_arr(model, vx, vi, flag, device, batch=batch)
        acc_logits = logits if acc_logits is None else acc_logits + logits
    return acc_logits / float(len(views))


def collect_oof_branch(meta, X_skel, X_imu, y, users, has_imu, device, use_tta, n_splits=5):
    n = len(y)
    oof = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for fi, (tr, va) in enumerate(gkf.split(np.arange(n), y, users)):
        xs = X_skel[va]
        xi = X_imu[va]
        fl = has_imu[va].astype(np.float32)
        m, _ = load_ckpt(
            meta["kind"],
            meta["fold_dir"] / f"best_fold{fi}.pt",
            device,
            meta["v2m"],
            meta["default"],
        )
        oof[va] = predict_with_tta(m, xs, xi, fl, device, use_tta=use_tta)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  OOF {meta['default']} fold{fi} tta={use_tta}", flush=True)
    return oof


def holdout_branch(meta, X_skel, X_imu, has_imu, ho_idx, device, use_tta):
    xs = X_skel[ho_idx]
    xi = X_imu[ho_idx]
    fl = has_imu[ho_idx].astype(np.float32)
    m, _ = load_ckpt(meta["kind"], meta["holdout_ckpt"], device, meta["v2m"], meta["default"])
    out = predict_with_tta(m, xs, xi, fl, device, use_tta=use_tta)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# -------------------- fusion helpers --------------------

def apply_conf_weighted(probs_list, temp):
    confs = [p.max(1, keepdims=True) for p in probs_list]
    logits_c = np.concatenate([np.log(np.maximum(c, 1e-8)) / temp for c in confs], axis=1)
    lc = logits_c - logits_c.max(1, keepdims=True)
    w = np.exp(lc)
    w = w / w.sum(1, keepdims=True)
    mix = sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list)))
    return mix


def fit_conf_weighted(probs_list, y, temps=None):
    if temps is None:
        temps = (0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in temps:
        mix = apply_conf_weighted(probs_list, temp)
        ca = acc(mix.argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_entropy_weighted(probs_list, temp):
    # lower entropy -> higher weight; soft via temp
    ws = []
    for p in probs_list:
        ent = -(p * np.log(np.maximum(p, 1e-8))).sum(1, keepdims=True)
        # confidence proxy = -entropy
        ws.append(-ent)
    logits_c = np.concatenate([w / temp for w in ws], axis=1)
    lc = logits_c - logits_c.max(1, keepdims=True)
    w = np.exp(lc)
    w = w / w.sum(1, keepdims=True)
    return sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list)))


def fit_entropy_weighted(probs_list, y, temps=None):
    if temps is None:
        temps = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    best = {"temp": 0.5, "oof_acc": -1.0}
    for temp in temps:
        mix = apply_entropy_weighted(probs_list, temp)
        ca = acc(mix.argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_wN(probs_list, w):
    mix = w[0] * probs_list[0]
    for i in range(1, len(probs_list)):
        mix = mix + w[i] * probs_list[i]
    return mix


def fit_global_w3(probs_list, y, grid=13):
    best_w, best_acc = (1 / 3, 1 / 3, 1 / 3), -1.0
    vals = np.linspace(0.0, 1.0, grid)
    for w0 in vals:
        for w1 in vals:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            w2 = max(0.0, float(w2))
            s = w0 + w1 + w2
            if s <= 1e-9:
                continue
            ww = (w0 / s, w1 / s, w2 / s)
            a = acc(apply_wN(probs_list, ww).argmax(1), y)
            if a > best_acc:
                best_w, best_acc = ww, a
    return {"w": best_w, "oof_acc": best_acc}


def fit_global_w4(probs_list, y, grid=9):
    best_w, best_acc = (0.25,) * 4, -1.0
    vals = np.linspace(0.0, 1.0, grid)
    for w0 in vals:
        for w1 in vals:
            for w2 in vals:
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(0.0, float(w3))
                s = w0 + w1 + w2 + w3
                if s <= 1e-9:
                    continue
                ww = (w0 / s, w1 / s, w2 / s, w3 / s)
                a = acc(apply_wN(probs_list, ww).argmax(1), y)
                if a > best_acc:
                    best_w, best_acc = ww, a
    return {"w": best_w, "oof_acc": best_acc}


def apply_power_mean(probs_list, p):
    # p->0 geometric-ish via soft; p=1 arithmetic
    stacked = np.stack([np.maximum(x, 1e-8) for x in probs_list], axis=0)
    if abs(p) < 1e-8:
        mix = np.exp(np.mean(np.log(stacked), axis=0))
    else:
        mix = np.mean(stacked ** p, axis=0) ** (1.0 / p)
    mix = mix / mix.sum(1, keepdims=True)
    return mix.astype(np.float32)


def fit_power_mean(probs_list, y, powers=None):
    if powers is None:
        powers = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    best = {"p": 1.0, "oof_acc": -1.0}
    for p in powers:
        mix = apply_power_mean(probs_list, p)
        ca = acc(mix.argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"p": float(p), "oof_acc": ca}
    return best


def fit_branch_temps(logits_list, y, grid=None):
    """Independent temperature scaling per branch (minimize NLL proxy via acc grid)."""
    if grid is None:
        grid = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
    temps = []
    for logits in logits_list:
        best_t, best_a = 1.0, -1.0
        for t in grid:
            p = softmax(logits / t)
            a = acc(p.argmax(1), y)
            if a > best_a:
                best_t, best_a = float(t), a
        temps.append(best_t)
    return temps


def apply_temp_then_mean(logits_list, temps):
    probs = [softmax(logits_list[i] / temps[i]) for i in range(len(logits_list))]
    return apply_wN(probs, tuple([1.0 / len(probs)] * len(probs)))


def apply_temp_then_conf(logits_list, temps, conf_temp):
    probs = [softmax(logits_list[i] / temps[i]) for i in range(len(logits_list))]
    return apply_conf_weighted(probs, conf_temp)


def fit_temp_then_conf(logits_list, y):
    temps = fit_branch_temps(logits_list, y)
    probs = [softmax(logits_list[i] / temps[i]) for i in range(len(logits_list))]
    cw = fit_conf_weighted(probs, y)
    return {"temps": temps, "conf_temp": cw["temp"], "oof_acc": cw["oof_acc"]}


def nested_generic(fit_fn, apply_fn, data_list, y, users, is_logits=False):
    """Nested GroupKFold: fit on tr, apply on va. data_list is list of (N,C) arrays."""
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(data_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        tr_list = [stacked[i, tr] for i in range(stacked.shape[0])]
        cfg = fit_fn(tr_list, y[tr])
        va_list = [stacked[i, va] for i in range(stacked.shape[0])]
        mix = apply_fn(va_list, cfg)
        pred[va] = mix.argmax(1)
    return acc(pred, y), None


def nested_conf_oof(probs_list, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_conf_weighted([stacked[i, tr] for i in range(stacked.shape[0])], y[tr])
        mix = apply_conf_weighted([stacked[i, va] for i in range(stacked.shape[0])], cfg["temp"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_ent_oof(probs_list, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_entropy_weighted([stacked[i, tr] for i in range(stacked.shape[0])], y[tr])
        mix = apply_entropy_weighted([stacked[i, va] for i in range(stacked.shape[0])], cfg["temp"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_w3_oof(probs_list, y, users, grid=9):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_global_w3([stacked[i, tr] for i in range(3)], y[tr], grid=grid)
        mix = apply_wN([stacked[i, va] for i in range(3)], cfg["w"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_w4_oof(probs_list, y, users, grid=7):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_global_w4([stacked[i, tr] for i in range(4)], y[tr], grid=grid)
        mix = apply_wN([stacked[i, va] for i in range(4)], cfg["w"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_power_oof(probs_list, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_power_mean([stacked[i, tr] for i in range(stacked.shape[0])], y[tr])
        mix = apply_power_mean([stacked[i, va] for i in range(stacked.shape[0])], cfg["p"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_temp_conf_oof(logits_list, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(logits_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_temp_then_conf([stacked[i, tr] for i in range(stacked.shape[0])], y[tr])
        mix = apply_temp_then_conf(
            [stacked[i, va] for i in range(stacked.shape[0])],
            cfg["temps"],
            cfg["conf_temp"],
        )
        pred[va] = mix.argmax(1)
    return acc(pred, y)



def apply_sharp_equal(logits_list, temp):
    probs = [softmax(logits / temp) for logits in logits_list]
    return apply_wN(probs, tuple([1.0 / len(probs)] * len(probs)))


def fit_sharp_equal(logits_list, y, temps=None):
    if temps is None:
        temps = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in temps:
        mix = apply_sharp_equal(logits_list, temp)
        ca = acc(mix.argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def nested_sharp_equal_oof(logits_list, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(logits_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_sharp_equal([stacked[i, tr] for i in range(stacked.shape[0])], y[tr])
        mix = apply_sharp_equal([stacked[i, va] for i in range(stacked.shape[0])], cfg["temp"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_equal_oof(probs_list, y, users):
    """Equal-weight has no hyperparams; report in-sample = nested."""
    mix = apply_wN(probs_list, tuple([1.0 / len(probs_list)] * len(probs_list)))
    return acc(mix.argmax(1), y)


def test_branch_probs(branches, names, cache, device, use_tta, n_folds=5):
    X_te, paths = load_skel_test_cache(cache)
    _, _, X_imu_te, has_imu_te = load_imu_caches(cache)
    has = has_imu_te if has_imu_te is not None else np.ones(len(X_te), bool)
    fl = has.astype(np.float32)
    avg_logits = {n: None for n in names}
    for fi in range(n_folds):
        for n in names:
            meta = branches[n]
            m, _ = load_ckpt(
                meta["kind"],
                meta["fold_dir"] / f"best_fold{fi}.pt",
                device,
                meta["v2m"],
                meta["default"],
            )
            logits = predict_with_tta(m, X_te, X_imu_te, fl, device, use_tta=use_tta)
            avg_logits[n] = logits if avg_logits[n] is None else avg_logits[n] + logits
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"test fold{fi} tta={use_tta} done", flush=True)
    probs = {}
    logits_out = {}
    for n in names:
        logits_out[n] = avg_logits[n] / float(n_folds)
        probs[n] = softmax(logits_out[n])
    return probs, logits_out, paths


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = V7 / "cache"
    v2m = load_v2_model_mod()
    use_tta = False

    branches = {
        "midfuse": {
            "kind": "v2",
            "v2m": v2m,
            "default": "midfuse",
            "fold_dir": V2 / "checkpoints_midfuse_v2b",
            "holdout_ckpt": V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt",
        },
        "stgcn": {
            "kind": "stgcn",
            "v2m": v2m,
            "default": "stgcn_fuse",
            "fold_dir": V7 / "checkpoints_cv",
            "holdout_ckpt": V7 / "checkpoints_v2" / "best_holdout.pt",
        },
        "gru": {
            "kind": "v2",
            "v2m": v2m,
            "default": "gru_attn",
            "fold_dir": V2 / "checkpoints_gru",
            "holdout_ckpt": V2 / "checkpoints_gru" / "best_holdout.pt",
        },
        "deepconv": {
            "kind": "v2",
            "v2m": v2m,
            "default": "deepconv",
            "fold_dir": V2 / "checkpoints_deepconv",
            "holdout_ckpt": V2 / "checkpoints_deepconv" / "best_holdout.pt",
        },
    }

    X_skel, y, users, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    n = len(y)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]
    print(f"n={n} nh={len(nh_idx)} ho={len(ho_idx)} device={device} tta={use_tta}", flush=True)

    oof_path = ROOT / "oof_logits.npz"
    oof = {}
    if oof_path.exists():
        z = np.load(oof_path, allow_pickle=False)
        for k in ("midfuse", "stgcn", "gru", "deepconv"):
            if k in z.files:
                oof[k] = z[k]
        print(f"Loaded cached OOF keys={list(oof)}", flush=True)

    # Prefer TTA OOF; if cache is non-TTA (from partial run), recompute when flag file says so
    meta_path = ROOT / "oof_meta.json"
    need = []
    oof_meta = {}
    if meta_path.exists():
        oof_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name in branches:
        if name not in oof or oof_meta.get(name, {}).get("tta") != use_tta:
            need.append(name)
    # bootstrap non-TTA from v8 for speed only when use_tta=False
    if not use_tta:
        v8z = np.load(V8 / "oof_logits.npz", allow_pickle=False)
        for k in ("midfuse", "stgcn", "gru"):
            if k in need:
                oof[k] = v8z[k]
                oof_meta[k] = {"tta": False, "source": "v8"}
                need.remove(k)

    for name in need:
        print(f"Collecting OOF for {name} tta={use_tta} ...", flush=True)
        oof[name] = collect_oof_branch(
            branches[name], X_skel, X_imu, y, users, has_imu, device, use_tta=use_tta
        )
        oof_meta[name] = {"tta": use_tta, "source": "v9"}
        np.savez_compressed(oof_path, **oof, y=y, users=users)
        meta_path.write_text(json.dumps(oof_meta, indent=2), encoding="utf-8")

    np.savez_compressed(oof_path, **oof, y=y, users=users)
    meta_path.write_text(json.dumps(oof_meta, indent=2), encoding="utf-8")

    yt_nh = y[nh_idx]
    users_nh = users[nh_idx]
    names3 = ["midfuse", "stgcn", "gru"]
    names4 = ["midfuse", "stgcn", "gru", "deepconv"]
    L = {k: oof[k][nh_idx] for k in names4}
    P = {k: softmax(L[k]) for k in names4}

    single = {
        f"oof_{k}": acc(L[k].argmax(1), yt_nh) for k in names4
    }
    for a, b in [("midfuse", "stgcn"), ("midfuse", "gru"), ("midfuse", "deepconv"),
                 ("stgcn", "gru"), ("stgcn", "deepconv"), ("gru", "deepconv")]:
        single[f"agree_{a[:2]}_{b[:2]}"] = float((L[a].argmax(1) == L[b].argmax(1)).mean())
    print(json.dumps(single, indent=2), flush=True)

    methods = {}

    # --- 3-branch methods ---
    plist3 = [P[k] for k in names3]
    llist3 = [L[k] for k in names3]

    cw3 = fit_conf_weighted(plist3, yt_nh)
    methods["conf_w_3"] = {
        "oof_acc": nested_conf_oof(plist3, yt_nh, users_nh),
        "oof_acc_in_sample": cw3["oof_acc"],
        "params": {"temp": cw3["temp"]},
        "family": "conf_w",
        "branches": names3,
    }
    print(f"conf_w_3 nested={methods['conf_w_3']['oof_acc']:.4f} t={cw3['temp']}", flush=True)

    ew3 = fit_entropy_weighted(plist3, yt_nh)
    methods["ent_w_3"] = {
        "oof_acc": nested_ent_oof(plist3, yt_nh, users_nh),
        "oof_acc_in_sample": ew3["oof_acc"],
        "params": {"temp": ew3["temp"]},
        "family": "ent_w",
        "branches": names3,
    }
    print(f"ent_w_3 nested={methods['ent_w_3']['oof_acc']:.4f}", flush=True)

    w3 = fit_global_w3(plist3, yt_nh, grid=13)
    methods["w3_grid"] = {
        "oof_acc": nested_w3_oof(plist3, yt_nh, users_nh, grid=9),
        "oof_acc_in_sample": w3["oof_acc"],
        "params": {"w": list(w3["w"])},
        "family": "wN",
        "branches": names3,
    }
    methods["w3_equal"] = {
        "oof_acc": nested_equal_oof(plist3, yt_nh, users_nh),
        "oof_acc_in_sample": nested_equal_oof(plist3, yt_nh, users_nh),
        "params": {"w": [1 / 3] * 3},
        "family": "wN_equal",
        "branches": names3,
    }
    print(f"w3_grid nested={methods['w3_grid']['oof_acc']:.4f} equal={methods['w3_equal']['oof_acc']:.4f}", flush=True)

    pm3 = fit_power_mean(plist3, yt_nh)
    methods["power_3"] = {
        "oof_acc": nested_power_oof(plist3, yt_nh, users_nh),
        "oof_acc_in_sample": pm3["oof_acc"],
        "params": {"p": pm3["p"]},
        "family": "power",
        "branches": names3,
    }
    tc3 = fit_temp_then_conf(llist3, yt_nh)
    methods["temp_conf_3"] = {
        "oof_acc": nested_temp_conf_oof(llist3, yt_nh, users_nh),
        "oof_acc_in_sample": tc3["oof_acc"],
        "params": {"temps": tc3["temps"], "conf_temp": tc3["conf_temp"]},
        "family": "temp_conf",
        "branches": names3,
    }
    print(f"power_3 nested={methods['power_3']['oof_acc']:.4f} temp_conf_3={methods['temp_conf_3']['oof_acc']:.4f}", flush=True)

    # --- 4-branch methods ---
    plist4 = [P[k] for k in names4]
    llist4 = [L[k] for k in names4]

    cw4 = fit_conf_weighted(plist4, yt_nh)
    methods["conf_w_4"] = {
        "oof_acc": nested_conf_oof(plist4, yt_nh, users_nh),
        "oof_acc_in_sample": cw4["oof_acc"],
        "params": {"temp": cw4["temp"]},
        "family": "conf_w",
        "branches": names4,
    }
    print(f"conf_w_4 nested={methods['conf_w_4']['oof_acc']:.4f} t={cw4['temp']}", flush=True)

    ew4 = fit_entropy_weighted(plist4, yt_nh)
    methods["ent_w_4"] = {
        "oof_acc": nested_ent_oof(plist4, yt_nh, users_nh),
        "oof_acc_in_sample": ew4["oof_acc"],
        "params": {"temp": ew4["temp"]},
        "family": "ent_w",
        "branches": names4,
    }
    w4 = fit_global_w4(plist4, yt_nh, grid=9)
    methods["w4_grid"] = {
        "oof_acc": nested_w4_oof(plist4, yt_nh, users_nh, grid=7),
        "oof_acc_in_sample": w4["oof_acc"],
        "params": {"w": list(w4["w"])},
        "family": "wN",
        "branches": names4,
    }
    methods["w4_equal"] = {
        "oof_acc": nested_equal_oof(plist4, yt_nh, users_nh),
        "oof_acc_in_sample": nested_equal_oof(plist4, yt_nh, users_nh),
        "params": {"w": [0.25] * 4},
        "family": "wN_equal",
        "branches": names4,
    }
    pm4 = fit_power_mean(plist4, yt_nh)
    methods["power_4"] = {
        "oof_acc": nested_power_oof(plist4, yt_nh, users_nh),
        "oof_acc_in_sample": pm4["oof_acc"],
        "params": {"p": pm4["p"]},
        "family": "power",
        "branches": names4,
    }
    tc4 = fit_temp_then_conf(llist4, yt_nh)
    methods["temp_conf_4"] = {
        "oof_acc": nested_temp_conf_oof(llist4, yt_nh, users_nh),
        "oof_acc_in_sample": tc4["oof_acc"],
        "params": {"temps": tc4["temps"], "conf_temp": tc4["conf_temp"]},
        "family": "temp_conf",
        "branches": names4,
    }

    se3 = fit_sharp_equal(llist3, yt_nh)
    methods["sharp_eq_3"] = {
        "oof_acc": nested_sharp_equal_oof(llist3, yt_nh, users_nh),
        "oof_acc_in_sample": se3["oof_acc"],
        "params": {"temp": se3["temp"]},
        "family": "sharp_eq",
        "branches": names3,
    }
    se4 = fit_sharp_equal(llist4, yt_nh)
    methods["sharp_eq_4"] = {
        "oof_acc": nested_sharp_equal_oof(llist4, yt_nh, users_nh),
        "oof_acc_in_sample": se4["oof_acc"],
        "params": {"temp": se4["temp"]},
        "family": "sharp_eq",
        "branches": names4,
    }
    print(f"sharp_eq_3={methods['sharp_eq_3']['oof_acc']:.4f} sharp_eq_4={methods['sharp_eq_4']['oof_acc']:.4f}", flush=True)

    # MF+GRU+DC (drop weak ST-GCN) — diversity check
    names_mgd = ["midfuse", "gru", "deepconv"]
    plist_mgd = [P[k] for k in names_mgd]
    cw_mgd = fit_conf_weighted(plist_mgd, yt_nh)
    methods["conf_w_mgd"] = {
        "oof_acc": nested_conf_oof(plist_mgd, yt_nh, users_nh),
        "oof_acc_in_sample": cw_mgd["oof_acc"],
        "params": {"temp": cw_mgd["temp"]},
        "family": "conf_w",
        "branches": names_mgd,
    }
    methods["w3_equal_mgd"] = {
        "oof_acc": nested_equal_oof(plist_mgd, yt_nh, users_nh),
        "oof_acc_in_sample": nested_equal_oof(plist_mgd, yt_nh, users_nh),
        "params": {"w": [1 / 3] * 3},
        "family": "wN_equal",
        "branches": names_mgd,
    }
    print(f"conf_w_4={methods['conf_w_4']['oof_acc']:.4f} ent_w_4={methods['ent_w_4']['oof_acc']:.4f} "
          f"w4={methods['w4_grid']['oof_acc']:.4f} mgd={methods['conf_w_mgd']['oof_acc']:.4f}", flush=True)

    best_name = max(methods, key=lambda k: methods[k]["oof_acc"])
    best = methods[best_name]
    print(f"BEST method={best_name} nested_oof={best['oof_acc']:.4f}", flush=True)

    # Holdout LAST
    print("Holdout inference...", flush=True)
    H = {}
    for name, meta in branches.items():
        H[name] = holdout_branch(meta, X_skel, X_imu, has_imu, ho_idx, device, use_tta=use_tta)
        print(f"  holdout {name} done", flush=True)
    yt_h = y[ho_idx]
    HL = H
    HP = {k: softmax(H[k]) for k in names4}
    hold_single = {f"holdout_{k}": acc(H[k].argmax(1), yt_h) for k in names4}

    def apply_named(name, probs_map, logits_map):
        m = methods[name]
        fam, p, bl = m["family"], m["params"], m["branches"]
        plist = [probs_map[b] for b in bl]
        llist = [logits_map[b] for b in bl]
        if fam == "conf_w":
            return apply_conf_weighted(plist, p["temp"])
        if fam == "ent_w":
            return apply_entropy_weighted(plist, p["temp"])
        if fam in ("wN", "wN_equal"):
            return apply_wN(plist, tuple(p["w"]))
        if fam == "power":
            return apply_power_mean(plist, p["p"])
        if fam == "temp_conf":
            return apply_temp_then_conf(llist, p["temps"], p["conf_temp"])
        if fam == "sharp_eq":
            return apply_sharp_equal(llist, p["temp"])
        raise ValueError(name)

    hold_rows = {name: acc(apply_named(name, HP, HL).argmax(1), yt_h) for name in methods}
    hold_best = hold_rows[best_name]
    peek_best_name = max(hold_rows, key=lambda k: hold_rows[k])
    peek = {
        "best_holdout_method_peek": peek_best_name,
        "best_holdout_acc_peek": hold_rows[peek_best_name],
        "note": "NOT used for selection or overwrite decision",
    }

    oof_best = best["oof_acc"]
    clear_win = bool(oof_best >= 0.548 or hold_best >= 0.566)
    overwrite = bool(clear_win)
    ping = bool(clear_win and overwrite)

    print(f"selected {best_name}: nested_oof={oof_best:.4f} holdout={hold_best:.4f} "
          f"clear_win={clear_win} overwrite={overwrite}", flush=True)

    # Test submission for selected method
    bl = best["branches"]
    test_probs, test_logits, paths = test_branch_probs(branches, bl, cache, device, use_tta=use_tta)
    mix_te = apply_named(best_name, test_probs, test_logits)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub_v9 = ROOT / "submission_v9.csv"
    df.to_csv(sub_v9, index=False)
    np.savez_compressed(
        ROOT / "submission_v9_probs.npz",
        probs=mix_te,
        paths=np.array(paths, dtype=object),
        method=np.array(best_name),
    )

    overwritten = False
    if overwrite:
        df.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submissions").mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v9.csv", index=False)
        overwritten = True

    methods_out = {}
    for k, v in methods.items():
        methods_out[k] = {
            "oof_acc_nested": v["oof_acc"],
            "oof_acc_in_sample": v.get("oof_acc_in_sample"),
            "holdout_acc": hold_rows[k],
            "family": v["family"],
            "branches": v["branches"],
            "params": v["params"],
        }

    metrics = {
        "track": "v9_4branch_tta_fuse",
        "v8_baseline_oof": V8_OOF,
        "v8_baseline_holdout": V8_HOLD,
        "use_tta": use_tta,
        "tta_views": ["id", "lr_flip", "shift+4", "shift-4"] if use_tta else ["id"],
        "single_oof": single,
        "single_holdout": hold_single,
        "methods": methods_out,
        "selected_method": best_name,
        "selected_oof_acc_nested": oof_best,
        "selected_holdout_acc": hold_best,
        "delta_oof_vs_v8": oof_best - V8_OOF,
        "delta_holdout_vs_v8": hold_best - V8_HOLD,
        "clear_win": clear_win,
        "overwrite_submission": overwrite,
        "submission_overwrote_track": overwritten,
        "ping_disk_saver": ping,
        "holdout_peek_not_used_for_selection": peek,
        "elapsed_sec": time.time() - t0,
        "finished_at_unix": time.time(),
        "protocol": "GroupKFold OOF; nested hyperparam fit on non-holdout; holdout last; light TTA",
        "submission_v9": str(sub_v9),
        "win_criteria": "nested_OOF>=0.548 or holdout>=0.566",
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in (
        "selected_method", "selected_oof_acc_nested", "selected_holdout_acc",
        "delta_oof_vs_v8", "delta_holdout_vs_v8", "clear_win", "overwrite_submission",
        "ping_disk_saver", "elapsed_sec"
    )}, indent=2), flush=True)
    print(f"Wrote {sub_v9}; overwrite={overwritten}; ping={ping}", flush=True)


if __name__ == "__main__":
    main()
