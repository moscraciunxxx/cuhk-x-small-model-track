"""v10: expanded nested fuse search + MidFuse seed3 + OOF-only calibration.

Ideas:
  - Third MidFuse seed (s7) when ckpts ready
  - More method recipes (power/conf/entropy/temps/prior/hard-reweight)
  - Nested method-blend search (equal + nested-honest weight search)
  - Class-prior / temperature calibration from OOF only
  - No TTA, no sklearn stackers
  - Radar fill skipped (not a quick win here)

Clear-win vs v9: nested OOF >= 0.558 OR holdout >= 0.576.
Overwrite submission + ping_disk_saver only on clear win.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

V7 = Path(__file__).resolve().parent.parent / "v7_stgcn"
V2 = Path(__file__).resolve().parent.parent / "skeleton_imu_v2"
V9 = Path(__file__).resolve().parent.parent / "v9"
ROOT = Path(__file__).resolve().parent
TRACK = ROOT.parent.parent
sys.path.insert(0, str(V7))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    load_imu_caches,
    load_skel_train_cache,
    load_skel_test_cache,
)
from model import build_model as build_stgcn  # noqa: E402

V9_OOF = 0.5486397361912614
V9_HOLD = 0.5663366336633663
WIN_OOF = 0.558
WIN_HOLD = 0.576
NUM_CLASSES = 40
POWERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TEMPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
BRANCH_TEMPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
PRIOR_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v10", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(builder_kind, ckpt_path, device, v2m, default_name):
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


def collect_oof_branch(meta, X_skel, X_imu, y, users, has_imu, device, n_splits=5):
    n = len(y)
    oof = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for fi, (_, va) in enumerate(gkf.split(np.arange(n), y, users)):
        m, _ = load_ckpt(
            meta["kind"], meta["fold_dir"] / f"best_fold{fi}.pt", device, meta["v2m"], meta["default"]
        )
        oof[va] = predict_logits_arr(
            m, X_skel[va], X_imu[va], has_imu[va].astype(np.float32), device
        )
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  OOF {meta['tag']} fold{fi}", flush=True)
    return oof


def holdout_branch(meta, X_skel, X_imu, has_imu, ho_idx, device):
    m, _ = load_ckpt(meta["kind"], meta["holdout_ckpt"], device, meta["v2m"], meta["default"])
    out = predict_logits_arr(
        m, X_skel[ho_idx], X_imu[ho_idx], has_imu[ho_idx].astype(np.float32), device
    )
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def test_branch_logits(branches, names, cache, device, n_folds=5):
    X_te, paths = load_skel_test_cache(cache)
    _, _, X_imu_te, has_imu_te = load_imu_caches(cache)
    has = has_imu_te if has_imu_te is not None else np.ones(len(X_te), bool)
    fl = has.astype(np.float32)
    avg = {n: None for n in names}
    for fi in range(n_folds):
        for n in names:
            meta = branches[n]
            m, _ = load_ckpt(
                meta["kind"], meta["fold_dir"] / f"best_fold{fi}.pt", device, meta["v2m"], meta["default"]
            )
            logits = predict_logits_arr(m, X_te, X_imu_te, fl, device)
            avg[n] = logits if avg[n] is None else avg[n] + logits
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"test fold{fi} done", flush=True)
    out_l = {n: avg[n] / float(n_folds) for n in names}
    out_p = {n: softmax(out_l[n]) for n in names}
    return out_p, out_l, paths


def apply_power_mean(probs_list, p):
    stacked = np.stack([np.maximum(x, 1e-8) for x in probs_list], axis=0)
    if abs(p) < 1e-8:
        mix = np.exp(np.mean(np.log(stacked), axis=0))
    else:
        mix = np.mean(stacked ** p, axis=0) ** (1.0 / p)
    mix = mix / mix.sum(1, keepdims=True)
    return mix.astype(np.float32)


def fit_power_mean(probs_list, y, powers=POWERS):
    best = {"p": 1.0, "oof_acc": -1.0}
    for p in powers:
        ca = acc(apply_power_mean(probs_list, p).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"p": float(p), "oof_acc": ca}
    return best


def apply_conf_weighted(probs_list, temp):
    confs = [p.max(1, keepdims=True) for p in probs_list]
    logits_c = np.concatenate([np.log(np.maximum(c, 1e-8)) / temp for c in confs], axis=1)
    lc = logits_c - logits_c.max(1, keepdims=True)
    w = np.exp(lc)
    w = w / w.sum(1, keepdims=True)
    return sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list))).astype(np.float32)


def fit_conf_weighted(probs_list, y, temps=TEMPS):
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in temps:
        ca = acc(apply_conf_weighted(probs_list, temp).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_entropy_weighted(probs_list, temp):
    ents = []
    for p in probs_list:
        pp = np.maximum(p, 1e-8)
        ent = -(pp * np.log(pp)).sum(1, keepdims=True)
        ents.append(ent)
    # lower entropy -> higher weight
    scores = [np.exp(-e / temp) for e in ents]
    wmat = np.concatenate(scores, axis=1)
    wmat = wmat / wmat.sum(1, keepdims=True)
    return sum(wmat[:, i : i + 1] * probs_list[i] for i in range(len(probs_list))).astype(np.float32)


def fit_entropy_weighted(probs_list, y, temps=TEMPS):
    best = {"temp": 0.5, "oof_acc": -1.0}
    for temp in temps:
        ca = acc(apply_entropy_weighted(probs_list, temp).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_equal(probs_list):
    mix = sum(probs_list) / float(len(probs_list))
    return mix.astype(np.float32)


def class_prior(y, n_classes=NUM_CLASSES):
    c = np.bincount(y, minlength=n_classes).astype(np.float64)
    c = np.maximum(c, 1.0)
    return (c / c.sum()).astype(np.float32)


def apply_prior_calib(probs, prior, alpha):
    if abs(alpha) < 1e-8:
        return probs.astype(np.float32)
    adj = np.maximum(probs, 1e-8) / np.maximum(prior[None, :] ** alpha, 1e-8)
    adj = adj / adj.sum(1, keepdims=True)
    return adj.astype(np.float32)


def fit_prior_calib(probs, y, prior, alphas=PRIOR_ALPHAS):
    best = {"alpha": 0.0, "oof_acc": -1.0}
    for a in alphas:
        ca = acc(apply_prior_calib(probs, prior, a).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"alpha": float(a), "oof_acc": ca}
    return best


def apply_branch_temps(logits_list, temps):
    return [softmax(logits_list[i] / temps[i]) for i in range(len(logits_list))]


def fit_branch_temps(logits_list, y, grid=BRANCH_TEMPS):
    # coordinate ascent, 2 passes
    temps = [1.0] * len(logits_list)
    best_acc = -1.0
    for _ in range(2):
        for i in range(len(logits_list)):
            local_best_t = temps[i]
            local_best_a = -1.0
            for t in grid:
                temps[i] = float(t)
                mix = apply_equal(apply_branch_temps(logits_list, temps))
                ca = acc(mix.argmax(1), y)
                if ca > local_best_a:
                    local_best_a = ca
                    local_best_t = float(t)
            temps[i] = local_best_t
            best_acc = max(best_acc, local_best_a)
    return {"temps": [float(t) for t in temps], "oof_acc": float(best_acc)}


def apply_temp_then_conf(logits_list, temps, conf_temp):
    probs = apply_branch_temps(logits_list, temps)
    return apply_conf_weighted(probs, conf_temp)


def fit_temp_then_conf(logits_list, y):
    bt = fit_branch_temps(logits_list, y)
    probs = apply_branch_temps(logits_list, bt["temps"])
    cw = fit_conf_weighted(probs, y)
    mix = apply_conf_weighted(probs, cw["temp"])
    return {
        "temps": bt["temps"],
        "conf_temp": cw["temp"],
        "oof_acc": acc(mix.argmax(1), y),
    }


def hard_weights(probs_list, gamma=1.0):
    """Higher weight when branches disagree (low max agreement / high entropy of mean)."""
    stack = np.stack([p.argmax(1) for p in probs_list], axis=1)
    # fraction of branches agreeing with majority
    maj = []
    for row in stack:
        vals, counts = np.unique(row, return_counts=True)
        maj.append(counts.max() / float(len(row)))
    agree = np.asarray(maj, dtype=np.float64)
    w = (1.0 - agree) ** gamma
    w = w + 0.05
    w = w / w.mean()
    return w.astype(np.float64)


def fit_power_mean_hard(probs_list, y, powers=POWERS, gammas=(0.5, 1.0, 2.0)):
    """Power-mean selected by hard-example-weighted accuracy (still reports plain acc)."""
    best = {"p": 1.0, "gamma": 1.0, "oof_acc": -1.0, "hard_score": -1.0}
    for g in gammas:
        hw = hard_weights(probs_list, g)
        for p in powers:
            pred = apply_power_mean(probs_list, p).argmax(1)
            correct = (pred == y).astype(np.float64)
            hs = float((correct * hw).sum() / hw.sum())
            ca = float(correct.mean())
            # select by hard score but keep plain acc
            if hs > best["hard_score"] or (abs(hs - best["hard_score"]) < 1e-12 and ca > best["oof_acc"]):
                best = {"p": float(p), "gamma": float(g), "oof_acc": ca, "hard_score": hs}
    return best


def simplex_weights(n, grid):
    """Yield weight tuples summing to 1 on a coarse grid."""
    if n == 2:
        for i in range(grid + 1):
            a = i / grid
            yield (a, 1.0 - a)
    elif n == 3:
        for i in range(grid + 1):
            for j in range(grid + 1 - i):
                a = i / grid
                b = j / grid
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                yield (a, b, max(0.0, c))
    elif n == 4:
        for i in range(grid + 1):
            for j in range(grid + 1 - i):
                for k in range(grid + 1 - i - j):
                    a = i / grid
                    b = j / grid
                    c = k / grid
                    d = 1.0 - a - b - c
                    if d < -1e-9:
                        continue
                    yield (a, b, c, max(0.0, d))
    else:
        yield tuple([1.0 / n] * n)


def apply_wN(probs_list, w):
    mix = w[0] * probs_list[0]
    for i in range(1, len(probs_list)):
        mix = mix + w[i] * probs_list[i]
    return mix.astype(np.float32)


def fit_wN(probs_list, y, grid=7):
    n = len(probs_list)
    best_w = tuple([1.0 / n] * n)
    best_acc = -1.0
    for w in simplex_weights(n, grid):
        ca = acc(apply_wN(probs_list, w).argmax(1), y)
        if ca > best_acc:
            best_acc = ca
            best_w = w
    return {"w": [float(x) for x in best_w], "oof_acc": float(best_acc)}


# ---------- nested OOF builders ----------

def nested_apply(keys, P, yt, us, family, L=None):
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    prior_full = class_prior(yt)
    for tr, va in gkf.split(np.arange(n), yt, us):
        plist_tr = [P[k][tr] for k in keys]
        plist_va = [P[k][va] for k in keys]
        if family == "eq":
            out[va] = apply_equal(plist_va)
        elif family == "pow":
            cfg = fit_power_mean(plist_tr, yt[tr])
            out[va] = apply_power_mean(plist_va, cfg["p"])
        elif family == "conf":
            cfg = fit_conf_weighted(plist_tr, yt[tr])
            out[va] = apply_conf_weighted(plist_va, cfg["temp"])
        elif family == "ent":
            cfg = fit_entropy_weighted(plist_tr, yt[tr])
            out[va] = apply_entropy_weighted(plist_va, cfg["temp"])
        elif family == "wN":
            grid = 9 if len(keys) <= 3 else 5
            cfg = fit_wN(plist_tr, yt[tr], grid=grid)
            out[va] = apply_wN(plist_va, cfg["w"])
        elif family == "pow_hard":
            cfg = fit_power_mean_hard(plist_tr, yt[tr])
            out[va] = apply_power_mean(plist_va, cfg["p"])
        elif family == "prior_eq":
            mix_tr = apply_equal(plist_tr)
            mix_va = apply_equal(plist_va)
            prior = class_prior(yt[tr])
            cfg = fit_prior_calib(mix_tr, yt[tr], prior)
            out[va] = apply_prior_calib(mix_va, prior, cfg["alpha"])
        elif family == "tc":
            assert L is not None
            llist_tr = [L[k][tr] for k in keys]
            llist_va = [L[k][va] for k in keys]
            cfg = fit_temp_then_conf(llist_tr, yt[tr])
            out[va] = apply_temp_then_conf(llist_va, cfg["temps"], cfg["conf_temp"])
        elif family == "pow_prior":
            cfg = fit_power_mean(plist_tr, yt[tr])
            mix_tr = apply_power_mean(plist_tr, cfg["p"])
            mix_va = apply_power_mean(plist_va, cfg["p"])
            prior = class_prior(yt[tr])
            pc = fit_prior_calib(mix_tr, yt[tr], prior)
            out[va] = apply_prior_calib(mix_va, prior, pc["alpha"])
        else:
            raise ValueError(family)
    return out


def fit_family(keys, P, y, family, L=None):
    plist = [P[k] for k in keys]
    if family == "eq":
        mix = apply_equal(plist)
        return {"oof_acc": acc(mix.argmax(1), y)}
    if family == "pow":
        return fit_power_mean(plist, y)
    if family == "conf":
        return fit_conf_weighted(plist, y)
    if family == "ent":
        return fit_entropy_weighted(plist, y)
    if family == "wN":
        grid = 9 if len(keys) <= 3 else 5
        return fit_wN(plist, y, grid=grid)
    if family == "pow_hard":
        return fit_power_mean_hard(plist, y)
    if family == "prior_eq":
        mix = apply_equal(plist)
        prior = class_prior(y)
        return fit_prior_calib(mix, y, prior)
    if family == "tc":
        llist = [L[k] for k in keys]
        return fit_temp_then_conf(llist, y)
    if family == "pow_prior":
        pm = fit_power_mean(plist, y)
        mix = apply_power_mean(plist, pm["p"])
        prior = class_prior(y)
        pc = fit_prior_calib(mix, y, prior)
        return {"p": pm["p"], "alpha": pc["alpha"], "oof_acc": pc["oof_acc"]}
    raise ValueError(family)


def apply_family(keys, P, params, family, L=None):
    plist = [P[k] for k in keys]
    if family == "eq":
        return apply_equal(plist)
    if family == "pow":
        return apply_power_mean(plist, params["p"])
    if family == "conf":
        return apply_conf_weighted(plist, params["temp"])
    if family == "ent":
        return apply_entropy_weighted(plist, params["temp"])
    if family == "wN":
        return apply_wN(plist, params["w"])
    if family == "pow_hard":
        return apply_power_mean(plist, params["p"])
    if family == "prior_eq":
        return apply_prior_calib(apply_equal(plist), class_prior_from_params(params), params["alpha"])
    if family == "tc":
        llist = [L[k] for k in keys]
        return apply_temp_then_conf(llist, params["temps"], params["conf_temp"])
    if family == "pow_prior":
        mix = apply_power_mean(plist, params["p"])
        return apply_prior_calib(mix, class_prior_from_params(params), params["alpha"])
    raise ValueError(family)


def class_prior_from_params(params):
    if "prior" in params:
        return np.asarray(params["prior"], dtype=np.float32)
    return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES


def folds_complete(fold_dir, n=5):
    return all((fold_dir / f"best_fold{i}.pt").exists() for i in range(n)) and (
        fold_dir / "best_holdout.pt"
    ).exists()


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = V7 / "cache"
    v2m = load_v2_model_mod()

    branches = {
        "midfuse": {
            "kind": "v2", "v2m": v2m, "default": "midfuse", "tag": "midfuse",
            "fold_dir": V2 / "checkpoints_midfuse_v2b",
            "holdout_ckpt": V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt",
        },
        "midfuse2": {
            "kind": "v2", "v2m": v2m, "default": "midfuse", "tag": "midfuse2",
            "fold_dir": V2 / "checkpoints_midfuse_s123",
            "holdout_ckpt": V2 / "checkpoints_midfuse_s123" / "best_holdout.pt",
        },
        "stgcn": {
            "kind": "stgcn", "v2m": v2m, "default": "stgcn_fuse", "tag": "stgcn",
            "fold_dir": V7 / "checkpoints_cv",
            "holdout_ckpt": V7 / "checkpoints_v2" / "best_holdout.pt",
        },
        "gru": {
            "kind": "v2", "v2m": v2m, "default": "gru_attn", "tag": "gru",
            "fold_dir": V2 / "checkpoints_gru",
            "holdout_ckpt": V2 / "checkpoints_gru" / "best_holdout.pt",
        },
        "deepconv": {
            "kind": "v2", "v2m": v2m, "default": "deepconv", "tag": "deepconv",
            "fold_dir": V2 / "checkpoints_deepconv",
            "holdout_ckpt": V2 / "checkpoints_deepconv" / "best_holdout.pt",
        },
    }

    mf3_dir = V2 / "checkpoints_midfuse_s7"
    use_mf3 = folds_complete(mf3_dir)
    if use_mf3:
        branches["midfuse3"] = {
            "kind": "v2", "v2m": v2m, "default": "midfuse", "tag": "midfuse3",
            "fold_dir": mf3_dir,
            "holdout_ckpt": mf3_dir / "best_holdout.pt",
        }
        print("midfuse3 (seed7) READY", flush=True)
    else:
        print("midfuse3 (seed7) not ready — continuing without", flush=True)

    # Load / extend OOF
    oof_path = ROOT / "oof_logits_v10.npz"
    oof = {}
    for src in (oof_path, V9 / "oof_logits_v9b.npz", V9 / "oof_logits.npz"):
        if src.exists():
            z = np.load(src, allow_pickle=False)
            for k in z.files:
                if k not in oof:
                    oof[k] = z[k]
            print(f"seeded OOF from {src.name} keys={list(oof)}", flush=True)
            break

    X_skel, y, users, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    if "y" not in oof:
        oof["y"] = y
        oof["users"] = users
    y = oof["y"]
    users = oof["users"]

    for name, meta in branches.items():
        if name not in oof:
            print(f"Collecting OOF {name}...", flush=True)
            oof[name] = collect_oof_branch(meta, X_skel, X_imu, y, users, has_imu, device)
            np.savez_compressed(oof_path, **{k: oof[k] for k in oof})
    np.savez_compressed(oof_path, **{k: oof[k] for k in oof})

    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]
    yt_nh = y[nh_idx]
    users_nh = users[nh_idx]
    print(f"n={len(y)} nh={len(nh_idx)} ho={len(ho_idx)} device={device} mf3={use_mf3}", flush=True)

    L = {k: oof[k][nh_idx] for k in branches}
    P = {k: softmax(L[k]) for k in branches}

    # MidFuse averages
    mf_keys = [k for k in ("midfuse", "midfuse2", "midfuse3") if k in branches]
    L["mf_avg"] = sum(L[k] for k in mf_keys) / float(len(mf_keys))
    P["mf_avg"] = softmax(L["mf_avg"])
    if len(mf_keys) >= 2:
        L["mf12"] = 0.5 * (L["midfuse"] + L["midfuse2"])
        P["mf12"] = softmax(L["mf12"])
    if use_mf3:
        L["mf13"] = 0.5 * (L["midfuse"] + L["midfuse3"])
        P["mf13"] = softmax(L["mf13"])
        L["mf23"] = 0.5 * (L["midfuse2"] + L["midfuse3"])
        P["mf23"] = softmax(L["mf23"])

    single = {f"oof_{k}": acc(L[k].argmax(1), yt_nh) for k in list(branches) + ["mf_avg"]}
    print(json.dumps(single, indent=2), flush=True)

    # Bundle recipes: (name, keys, family)
    bundles = []

    # Core v9-like
    bundles += [
        ("eq_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "eq"),
        ("conf_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "conf"),
        ("ent_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "ent"),
        ("pow_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "pow"),
        ("tc_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "tc"),
        ("w3_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "wN"),
        ("prior_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "prior_eq"),
        ("powprior_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "pow_prior"),
        ("powhard_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "pow_hard"),
    ]

    keys4 = mf_keys + ["stgcn", "gru"]
    bundles += [
        ("eq_mfs_st_gru", list(keys4), "eq"),
        ("pow_mfs_st_gru", list(keys4), "pow"),
        ("conf_mfs_st_gru", list(keys4), "conf"),
        ("ent_mfs_st_gru", list(keys4), "ent"),
        ("tc_mfs_st_gru", list(keys4), "tc"),
        ("powprior_mfs_st_gru", list(keys4), "pow_prior"),
        ("powhard_mfs_st_gru", list(keys4), "pow_hard"),
    ]
    if len(keys4) <= 4:
        bundles.append(("wN_mfs_st_gru", list(keys4), "wN"))

    keys5 = mf_keys + ["stgcn", "gru", "deepconv"]
    bundles += [
        ("eq_all", list(keys5), "eq"),
        ("pow_all", list(keys5), "pow"),
        ("conf_all", list(keys5), "conf"),
        ("ent_all", list(keys5), "ent"),
        ("powprior_all", list(keys5), "pow_prior"),
        ("powhard_all", list(keys5), "pow_hard"),
        ("tc_all", list(keys5), "tc"),
    ]

    # mf12 classic 4-way / 5-way like v9
    if "mf12" in P:
        bundles += [
            ("pow_mf12_st_gru", ["midfuse", "midfuse2", "stgcn", "gru"], "pow"),
            ("eq_mf12_st_gru", ["mf12", "stgcn", "gru"], "eq"),
            ("conf_mf12_st_gru", ["mf12", "stgcn", "gru"], "conf"),
            ("pow_mf_mf2_st_gru_dc", ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "pow"),
        ]

    if use_mf3:
        bundles += [
            ("eq_mf3avg_st_gru", ["mf_avg", "stgcn", "gru"], "eq"),
            ("pow_mf123_st_gru", ["midfuse", "midfuse2", "midfuse3", "stgcn", "gru"], "pow"),
            ("conf_mf123_st_gru", ["midfuse", "midfuse2", "midfuse3", "stgcn", "gru"], "conf"),
            ("pow_mf123_all", ["midfuse", "midfuse2", "midfuse3", "stgcn", "gru", "deepconv"], "pow"),
            ("eq_mf23_st_gru", ["mf23", "stgcn", "gru"], "eq"),
            ("conf_mf23_st_gru", ["mf23", "stgcn", "gru"], "conf"),
        ]

    # v8 replay
    bundles += [
        ("conf_v8_3", ["midfuse", "stgcn", "gru"], "conf"),
        ("eq_v8_3", ["midfuse", "stgcn", "gru"], "eq"),
        ("pow_v8_3", ["midfuse", "stgcn", "gru"], "pow"),
    ]

    # Dedup by name
    seen = set()
    uniq = []
    for b in bundles:
        if b[0] in seen:
            continue
        seen.add(b[0])
        uniq.append(b)
    bundles = uniq

    nested_probs = {}
    methods = {}
    prior_nh = class_prior(yt_nh)

    for name, keys, family in bundles:
        # skip if missing keys
        if any(k not in P and k not in L for k in keys):
            print(f"skip {name} missing keys", flush=True)
            continue
        try:
            np_ = nested_apply(keys, P, yt_nh, users_nh, family, L=L)
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)
            continue
        nested_probs[name] = np_
        nested_a = acc(np_.argmax(1), yt_nh)
        params = fit_family(keys, P, yt_nh, family, L=L)
        # stash prior for apply stage
        if family in ("prior_eq", "pow_prior"):
            params = dict(params)
            params["prior"] = prior_nh.tolist()
        methods[name] = {
            "oof_acc": nested_a,
            "oof_acc_in_sample": params.get("oof_acc"),
            "params": {k: v for k, v in params.items() if k != "oof_acc"},
            "family": family,
            "keys": keys,
        }
        print(f"{name} nested={nested_a:.4f}", flush=True)

    # Method-level equal blends of top base methods (by nested), plus v9 ABD-style
    base_sorted = sorted(methods.keys(), key=lambda k: methods[k]["oof_acc"], reverse=True)
    top_bases = base_sorted[:8]
    print("TOP BASES:", [(k, round(methods[k]["oof_acc"], 4)) for k in top_bases], flush=True)

    blend_specs = {}
    # Always include v9 ABD analogues if present
    abd = [x for x in ("pow_mf_mf2_st_gru_dc", "pow_all", "eq_mfavg_st_gru", "pow_mfs_st_gru") if x in nested_probs]
    # rebuild classic ABD names if available
    classic = []
    for cand in ("pow_all", "pow_mf_mf2_st_gru_dc", "eq_mfavg_st_gru", "pow_mfs_st_gru", "conf_mfavg_st_gru"):
        if cand in nested_probs and cand not in classic:
            classic.append(cand)
    if len(classic) >= 3:
        blend_specs["blend_v9style"] = classic[:3]
    if len(classic) >= 2:
        blend_specs["blend_v9top2"] = classic[:2]

    # Combinations of top 5 bases: all size 2/3/4 equal blends
    top5 = top_bases[:5]
    for r in (2, 3, 4):
        if len(top5) < r:
            continue
        for comb in combinations(top5, r):
            blend_specs["eqblend_" + "_".join(comb)] = list(comb)

    # Nested-weight blends for top3
    if len(top5) >= 3:
        blend_specs["wblend_top3"] = top5[:3]  # special: nested weight fit

    for bname, members in blend_specs.items():
        if bname.startswith("wblend_"):
            # nested GroupKFold weight fit on method nested probs
            n = len(yt_nh)
            out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
            gkf = GroupKFold(n_splits=5)
            for tr, va in gkf.split(np.arange(n), yt_nh, users_nh):
                plist_tr = [nested_probs[m][tr] for m in members]
                plist_va = [nested_probs[m][va] for m in members]
                cfg = fit_wN(plist_tr, yt_nh[tr], grid=7)
                out[va] = apply_wN(plist_va, cfg["w"])
            nested_a = acc(out.argmax(1), yt_nh)
            nested_probs[bname] = out
            methods[bname] = {
                "oof_acc": nested_a,
                "oof_acc_in_sample": nested_a,
                "params": {"members": members, "w": "nested_fit"},
                "family": "method_wblend",
                "keys": members,
            }
        else:
            mix = sum(nested_probs[m] for m in members) / float(len(members))
            nested_a = acc(mix.argmax(1), yt_nh)
            nested_probs[bname] = mix
            methods[bname] = {
                "oof_acc": nested_a,
                "oof_acc_in_sample": nested_a,
                "params": {"members": members, "w": "equal"},
                "family": "method_blend",
                "keys": members,
            }
        print(f"{bname} nested={methods[bname]['oof_acc']:.4f}", flush=True)

    best_name = max(methods, key=lambda k: methods[k]["oof_acc"])
    best = methods[best_name]
    print(f"BEST={best_name} nested={best['oof_acc']:.4f}", flush=True)

    # Holdout LAST
    print("Holdout...", flush=True)
    H = {}
    for name in branches:
        H[name] = holdout_branch(branches[name], X_skel, X_imu, has_imu, ho_idx, device)
        print(f"  {name}", flush=True)
    yt_h = y[ho_idx]
    HL = dict(H)
    HL["mf_avg"] = sum(H[k] for k in mf_keys) / float(len(mf_keys))
    if "mf12" in P:
        HL["mf12"] = 0.5 * (H["midfuse"] + H["midfuse2"])
    if use_mf3:
        HL["mf13"] = 0.5 * (H["midfuse"] + H["midfuse3"])
        HL["mf23"] = 0.5 * (H["midfuse2"] + H["midfuse3"])
    HP = {k: softmax(HL[k]) for k in HL}
    hold_single = {f"holdout_{k}": acc(H[k].argmax(1), yt_h) for k in branches}
    hold_single["holdout_mf_avg"] = acc(HL["mf_avg"].argmax(1), yt_h)

    # Fit final params on full nh
    final_params = {}
    for name, m in methods.items():
        if m["family"] in ("method_blend", "method_wblend"):
            continue
        params = fit_family(m["keys"], P, yt_nh, m["family"], L=L)
        if m["family"] in ("prior_eq", "pow_prior"):
            params = dict(params)
            params["prior"] = prior_nh.tolist()
        final_params[name] = params

    # For method blends, also fit member params
    member_names = set()
    for m in methods.values():
        if m["family"] in ("method_blend", "method_wblend"):
            member_names.update(m["params"]["members"])
    for mem in member_names:
        if mem not in final_params and mem in methods and methods[mem]["family"] not in (
            "method_blend",
            "method_wblend",
        ):
            mm = methods[mem]
            params = fit_family(mm["keys"], P, yt_nh, mm["family"], L=L)
            if mm["family"] in ("prior_eq", "pow_prior"):
                params = dict(params)
                params["prior"] = prior_nh.tolist()
            final_params[mem] = params

    def apply_named(name, probs_map, logits_map):
        m = methods[name]
        fam = m["family"]
        if fam == "method_blend":
            members = m["params"]["members"]
            mixes = [apply_named(mem, probs_map, logits_map) for mem in members]
            return sum(mixes) / float(len(mixes))
        if fam == "method_wblend":
            members = m["params"]["members"]
            mixes = [apply_named(mem, probs_map, logits_map) for mem in members]
            # fit weights on nh nested? use full nh method probs via final_params if stored
            # Fit weights on nh using member apply on nh P
            plist = []
            for mem in members:
                mm = methods[mem]
                plist.append(
                    apply_family(mm["keys"], P, final_params[mem], mm["family"], L=L)
                )
            cfg = fit_wN(plist, yt_nh, grid=7)
            return apply_wN(mixes, cfg["w"])
        return apply_family(m["keys"], probs_map, final_params[name], fam, L=logits_map)

    hold_rows = {}
    for name in methods:
        try:
            hold_rows[name] = acc(apply_named(name, HP, HL).argmax(1), yt_h)
        except Exception as e:
            print(f"hold fail {name}: {e}", flush=True)
            hold_rows[name] = -1.0

    hold_best = hold_rows[best_name]
    peek_best = max(hold_rows, key=lambda k: hold_rows[k])
    peek = {
        "best_holdout_method_peek": peek_best,
        "best_holdout_acc_peek": hold_rows[peek_best],
        "note": "NOT used for selection or overwrite decision",
    }
    oof_best = best["oof_acc"]
    clear_win = bool(oof_best >= WIN_OOF or hold_best >= WIN_HOLD)
    overwrite = bool(clear_win)
    ping = bool(clear_win)
    print(
        f"selected {best_name}: nested={oof_best:.4f} hold={hold_best:.4f} win={clear_win}",
        flush=True,
    )

    # Test (always build artifact; overwrite track only on clear win)
    need = set()
    for name in [best_name]:
        m = methods[name]
        if m["family"] in ("method_blend", "method_wblend"):
            for mem in m["params"]["members"]:
                need.update(methods[mem]["keys"])
        else:
            need.update(m["keys"])
    # resolve aliases
    resolved = set()
    for k in need:
        if k in ("mf_avg",):
            resolved.update(mf_keys)
        elif k == "mf12":
            resolved.update(["midfuse", "midfuse2"])
        elif k == "mf13":
            resolved.update(["midfuse", "midfuse3"])
        elif k == "mf23":
            resolved.update(["midfuse2", "midfuse3"])
        else:
            resolved.add(k)
    need = sorted(resolved & set(branches.keys()))
    print(f"test branches: {need}", flush=True)
    test_probs, test_logits, paths = test_branch_logits(branches, need, cache, device)
    # fill averages
    if set(mf_keys).issubset(test_logits):
        test_logits["mf_avg"] = sum(test_logits[k] for k in mf_keys) / float(len(mf_keys))
        test_probs["mf_avg"] = softmax(test_logits["mf_avg"])
    if "midfuse" in test_logits and "midfuse2" in test_logits:
        test_logits["mf12"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse2"])
        test_probs["mf12"] = softmax(test_logits["mf12"])
    if use_mf3 and "midfuse3" in test_logits:
        if "midfuse" in test_logits:
            test_logits["mf13"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse3"])
            test_probs["mf13"] = softmax(test_logits["mf13"])
        if "midfuse2" in test_logits:
            test_logits["mf23"] = 0.5 * (test_logits["midfuse2"] + test_logits["midfuse3"])
            test_probs["mf23"] = softmax(test_logits["mf23"])

    mix_te = apply_named(best_name, test_probs, test_logits)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub = ROOT / "submission_v10.csv"
    df.to_csv(sub, index=False)
    np.savez_compressed(
        ROOT / "submission_v10_probs.npz",
        probs=mix_te,
        paths=np.array(paths, dtype=object),
        method=np.array(best_name),
    )

    overwritten = False
    if overwrite:
        df.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submissions").mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v10.csv", index=False)
        overwritten = True

    methods_out = {
        k: {
            "oof_acc_nested": v["oof_acc"],
            "oof_acc_in_sample": v.get("oof_acc_in_sample"),
            "holdout_acc": hold_rows.get(k),
            "family": v["family"],
            "keys": v["keys"],
            "params": v["params"],
        }
        for k, v in methods.items()
    }
    metrics = {
        "track": "v10_expanded_fuse",
        "v9_baseline_oof": V9_OOF,
        "v9_baseline_holdout": V9_HOLD,
        "use_tta": False,
        "use_midfuse3": use_mf3,
        "midfuse_seeds": {
            "midfuse": "checkpoints_midfuse_v2b/seed42",
            "midfuse2": "checkpoints_midfuse_s123/seed123",
            "midfuse3": "checkpoints_midfuse_s7/seed7" if use_mf3 else None,
        },
        "single_oof": single,
        "single_holdout": hold_single,
        "methods": methods_out,
        "selected_method": best_name,
        "selected_oof_acc_nested": oof_best,
        "selected_holdout_acc": hold_best,
        "delta_oof_vs_v9": oof_best - V9_OOF,
        "delta_holdout_vs_v9": hold_best - V9_HOLD,
        "clear_win": clear_win,
        "overwrite_submission": overwrite,
        "submission_overwrote_track": overwritten,
        "ping_disk_saver": ping,
        "holdout_peek_not_used_for_selection": peek,
        "elapsed_sec": time.time() - t0,
        "finished_at_unix": time.time(),
        "protocol": "GroupKFold nested method OOF; expanded recipes; holdout last; no TTA/sklearn",
        "submission_v10": str(sub),
        "win_criteria": f"nested_OOF>={WIN_OOF} or holdout>={WIN_HOLD}",
        "skipped": ["TTA", "sklearn_stackers", "radar_fill"],
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    top = sorted(methods_out.items(), key=lambda kv: kv[1]["oof_acc_nested"], reverse=True)[:20]
    print("TOP:", flush=True)
    for k, v in top:
        print(f"  {v['oof_acc_nested']:.4f} hold={v['holdout_acc']:.4f} {k}", flush=True)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "selected_method",
                    "selected_oof_acc_nested",
                    "selected_holdout_acc",
                    "delta_oof_vs_v9",
                    "delta_holdout_vs_v9",
                    "clear_win",
                    "overwrite_submission",
                    "ping_disk_saver",
                    "use_midfuse3",
                    "elapsed_sec",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
