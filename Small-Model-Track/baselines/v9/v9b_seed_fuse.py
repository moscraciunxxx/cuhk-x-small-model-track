"""v9b: seed-ensemble MidFuse + ST-GCN + GRU (+ optional DeepConv) late-fuse.

No TTA (hurt v9a). Nested GroupKFold selection on non-holdout only.
Win: nested OOF >= 0.548 or holdout >= 0.566.
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
    load_imu_caches,
    load_skel_train_cache,
    load_skel_test_cache,
)
from model import build_model as build_stgcn  # noqa: E402

V8_OOF = 0.5383347073371806
V8_HOLD = 0.5564356435643565
NUM_CLASSES = 40


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v9b", V2 / "model.py")
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
    for fi, (tr, va) in enumerate(gkf.split(np.arange(n), y, users)):
        m, _ = load_ckpt(meta["kind"], meta["fold_dir"] / f"best_fold{fi}.pt", device, meta["v2m"], meta["default"])
        oof[va] = predict_logits_arr(m, X_skel[va], X_imu[va], has_imu[va].astype(np.float32), device)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  OOF {meta['tag']} fold{fi}", flush=True)
    return oof


def holdout_branch(meta, X_skel, X_imu, has_imu, ho_idx, device):
    m, _ = load_ckpt(meta["kind"], meta["holdout_ckpt"], device, meta["v2m"], meta["default"])
    out = predict_logits_arr(m, X_skel[ho_idx], X_imu[ho_idx], has_imu[ho_idx].astype(np.float32), device)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def apply_conf_weighted(probs_list, temp):
    confs = [p.max(1, keepdims=True) for p in probs_list]
    logits_c = np.concatenate([np.log(np.maximum(c, 1e-8)) / temp for c in confs], axis=1)
    lc = logits_c - logits_c.max(1, keepdims=True)
    w = np.exp(lc)
    w = w / w.sum(1, keepdims=True)
    return sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list)))


def fit_conf_weighted(probs_list, y, temps=None):
    if temps is None:
        temps = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in temps:
        ca = acc(apply_conf_weighted(probs_list, temp).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_entropy_weighted(probs_list, temp):
    ws = []
    for p in probs_list:
        ent = -(p * np.log(np.maximum(p, 1e-8))).sum(1, keepdims=True)
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
        ca = acc(apply_entropy_weighted(probs_list, temp).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_wN(probs_list, w):
    mix = w[0] * probs_list[0]
    for i in range(1, len(probs_list)):
        mix = mix + w[i] * probs_list[i]
    return mix


def fit_global_w3(probs_list, y, grid=13):
    best_w, best_acc = (1 / 3,) * 3, -1.0
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
    stacked = np.stack([np.maximum(x, 1e-8) for x in probs_list], axis=0)
    if abs(p) < 1e-8:
        mix = np.exp(np.mean(np.log(stacked), axis=0))
    else:
        mix = np.mean(stacked ** p, axis=0) ** (1.0 / p)
    mix = mix / mix.sum(1, keepdims=True)
    return mix.astype(np.float32)


def fit_power_mean(probs_list, y, powers=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
    best = {"p": 1.0, "oof_acc": -1.0}
    for p in powers:
        ca = acc(apply_power_mean(probs_list, p).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"p": float(p), "oof_acc": ca}
    return best


def fit_branch_temps(logits_list, y, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)):
    temps = []
    for logits in logits_list:
        best_t, best_a = 1.0, -1.0
        for t in grid:
            a = acc(softmax(logits / t).argmax(1), y)
            if a > best_a:
                best_t, best_a = float(t), a
        temps.append(best_t)
    return temps


def apply_temp_then_conf(logits_list, temps, conf_temp):
    probs = [softmax(logits_list[i] / temps[i]) for i in range(len(logits_list))]
    return apply_conf_weighted(probs, conf_temp)


def fit_temp_then_conf(logits_list, y):
    temps = fit_branch_temps(logits_list, y)
    probs = [softmax(logits_list[i] / temps[i]) for i in range(len(logits_list))]
    cw = fit_conf_weighted(probs, y)
    return {"temps": temps, "conf_temp": cw["temp"], "oof_acc": cw["oof_acc"]}


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
            [stacked[i, va] for i in range(stacked.shape[0])], cfg["temps"], cfg["conf_temp"]
        )
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def equal_acc(probs_list, y):
    w = tuple([1.0 / len(probs_list)] * len(probs_list))
    return acc(apply_wN(probs_list, w).argmax(1), y)


def test_branch_logits(branches, names, cache, device, n_folds=5):
    X_te, paths = load_skel_test_cache(cache)
    _, _, X_imu_te, has_imu_te = load_imu_caches(cache)
    has = has_imu_te if has_imu_te is not None else np.ones(len(X_te), bool)
    fl = has.astype(np.float32)
    avg = {n: None for n in names}
    for fi in range(n_folds):
        for n in names:
            meta = branches[n]
            m, _ = load_ckpt(meta["kind"], meta["fold_dir"] / f"best_fold{fi}.pt", device, meta["v2m"], meta["default"])
            logits = predict_logits_arr(m, X_te, X_imu_te, fl, device)
            avg[n] = logits if avg[n] is None else avg[n] + logits
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"test fold{fi} done", flush=True)
    out_l = {n: avg[n] / float(n_folds) for n in names}
    out_p = {n: softmax(out_l[n]) for n in names}
    return out_p, out_l, paths


def register_bundle(methods, key, family, branches, nested, in_sample, params):
    methods[key] = {
        "oof_acc": nested,
        "oof_acc_in_sample": in_sample,
        "params": params,
        "family": family,
        "branches": list(branches),
    }
    print(f"{key} nested={nested:.4f} in_sample={in_sample:.4f}", flush=True)


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

    X_skel, y, users, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]
    print(f"n={len(y)} nh={len(nh_idx)} ho={len(ho_idx)} device={device}", flush=True)

    oof_path = ROOT / "oof_logits_v9b.npz"
    oof = {}
    if oof_path.exists():
        z = np.load(oof_path, allow_pickle=False)
        for k in branches:
            if k in z.files:
                oof[k] = z[k]
        print(f"Loaded OOF keys={list(oof)}", flush=True)

    # bootstrap known OOF
    if "midfuse" not in oof or "stgcn" not in oof or "gru" not in oof:
        v8z = np.load(V8 / "oof_logits.npz", allow_pickle=False)
        for k in ("midfuse", "stgcn", "gru"):
            oof[k] = v8z[k]
    if "deepconv" not in oof:
        prev = ROOT / "oof_logits.npz"
        if prev.exists():
            z = np.load(prev, allow_pickle=False)
            if "deepconv" in z.files:
                # only use if non-TTA: check meta
                meta_p = ROOT / "oof_meta.json"
                meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
                if meta.get("deepconv", {}).get("tta") is False:
                    oof["deepconv"] = z["deepconv"]

    for name in branches:
        if name not in oof:
            print(f"Collecting OOF {name}...", flush=True)
            oof[name] = collect_oof_branch(branches[name], X_skel, X_imu, y, users, has_imu, device)
            np.savez_compressed(oof_path, **oof, y=y, users=users)

    np.savez_compressed(oof_path, **oof, y=y, users=users)

    yt_nh = y[nh_idx]
    users_nh = users[nh_idx]
    L = {k: oof[k][nh_idx] for k in branches}
    P = {k: softmax(L[k]) for k in branches}
    # seed-averaged midfuse logits/probs
    L["mf_avg"] = 0.5 * (L["midfuse"] + L["midfuse2"])
    P["mf_avg"] = softmax(L["mf_avg"])

    single = {f"oof_{k}": acc(L[k].argmax(1), yt_nh) for k in list(branches) + ["mf_avg"]}
    single["agree_mf_mf2"] = float((L["midfuse"].argmax(1) == L["midfuse2"].argmax(1)).mean())
    single["agree_mf_gru"] = float((L["midfuse"].argmax(1) == L["gru"].argmax(1)).mean())
    single["agree_mf2_gru"] = float((L["midfuse2"].argmax(1) == L["gru"].argmax(1)).mean())
    print(json.dumps(single, indent=2), flush=True)

    methods = {}

    bundles = [
        ("v8_3", ["midfuse", "stgcn", "gru"]),
        ("mfavg_st_gru", ["mf_avg", "stgcn", "gru"]),
        ("mf_mf2_gru", ["midfuse", "midfuse2", "gru"]),
        ("mf_mf2_st", ["midfuse", "midfuse2", "stgcn"]),
        ("mf_mf2_st_gru", ["midfuse", "midfuse2", "stgcn", "gru"]),
        ("mfavg_st_gru_dc", ["mf_avg", "stgcn", "gru", "deepconv"]),
        ("mf_mf2_gru_dc", ["midfuse", "midfuse2", "gru", "deepconv"]),
        ("all5", ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"]),
    ]

    for bname, blist in bundles:
        # resolve probs/logits (mf_avg is derived)
        plist = [P[b] for b in blist]
        llist = [L[b] for b in blist]
        # equal
        eq = equal_acc(plist, yt_nh)
        register_bundle(methods, f"eq_{bname}", "wN_equal", blist, eq, eq, {"w": [1 / len(blist)] * len(blist)})
        # conf
        cw = fit_conf_weighted(plist, yt_nh)
        register_bundle(
            methods, f"conf_{bname}", "conf_w", blist,
            nested_conf_oof(plist, yt_nh, users_nh), cw["oof_acc"], {"temp": cw["temp"]},
        )
        # entropy
        ew = fit_entropy_weighted(plist, yt_nh)
        register_bundle(
            methods, f"ent_{bname}", "ent_w", blist,
            nested_ent_oof(plist, yt_nh, users_nh), ew["oof_acc"], {"temp": ew["temp"]},
        )
        # power
        pm = fit_power_mean(plist, yt_nh)
        register_bundle(
            methods, f"pow_{bname}", "power", blist,
            nested_power_oof(plist, yt_nh, users_nh), pm["oof_acc"], {"p": pm["p"]},
        )
        # temp+conf
        tc = fit_temp_then_conf(llist, yt_nh)
        register_bundle(
            methods, f"tc_{bname}", "temp_conf", blist,
            nested_temp_conf_oof(llist, yt_nh, users_nh), tc["oof_acc"],
            {"temps": tc["temps"], "conf_temp": tc["conf_temp"]},
        )
        if len(blist) == 3:
            w3 = fit_global_w3(plist, yt_nh, grid=13)
            register_bundle(
                methods, f"w3_{bname}", "wN", blist,
                nested_w3_oof(plist, yt_nh, users_nh, grid=9), w3["oof_acc"], {"w": list(w3["w"])},
            )
        if len(blist) == 4:
            w4 = fit_global_w4(plist, yt_nh, grid=9)
            register_bundle(
                methods, f"w4_{bname}", "wN", blist,
                nested_w4_oof(plist, yt_nh, users_nh, grid=7), w4["oof_acc"], {"w": list(w4["w"])},
            )

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
    HL["mf_avg"] = 0.5 * (H["midfuse"] + H["midfuse2"])
    HP = {k: softmax(HL[k]) for k in HL}
    hold_single = {f"holdout_{k}": acc(H[k].argmax(1), yt_h) for k in branches}
    hold_single["holdout_mf_avg"] = acc(HL["mf_avg"].argmax(1), yt_h)

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
        raise ValueError(name)

    hold_rows = {name: acc(apply_named(name, HP, HL).argmax(1), yt_h) for name in methods}
    hold_best = hold_rows[best_name]
    peek_best = max(hold_rows, key=lambda k: hold_rows[k])
    peek = {
        "best_holdout_method_peek": peek_best,
        "best_holdout_acc_peek": hold_rows[peek_best],
        "note": "NOT used for selection or overwrite decision",
    }
    oof_best = best["oof_acc"]
    clear_win = bool(oof_best >= 0.548 or hold_best >= 0.566)
    overwrite = bool(clear_win)
    ping = bool(clear_win)

    print(f"selected {best_name}: nested={oof_best:.4f} hold={hold_best:.4f} win={clear_win}", flush=True)

    # Test
    need = set(best["branches"]) - {"mf_avg"}
    if "mf_avg" in best["branches"]:
        need |= {"midfuse", "midfuse2"}
    test_probs, test_logits, paths = test_branch_logits(branches, sorted(need), cache, device)
    if "mf_avg" in best["branches"]:
        test_logits["mf_avg"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse2"])
        test_probs["mf_avg"] = softmax(test_logits["mf_avg"])
    mix_te = apply_named(best_name, test_probs, test_logits)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub = ROOT / "submission_v9.csv"
    df.to_csv(sub, index=False)
    np.savez_compressed(ROOT / "submission_v9_probs.npz", probs=mix_te, paths=np.array(paths, dtype=object), method=np.array(best_name))

    overwritten = False
    if overwrite:
        df.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submissions").mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v9.csv", index=False)
        overwritten = True

    methods_out = {
        k: {
            "oof_acc_nested": v["oof_acc"],
            "oof_acc_in_sample": v.get("oof_acc_in_sample"),
            "holdout_acc": hold_rows[k],
            "family": v["family"],
            "branches": v["branches"],
            "params": v["params"],
        }
        for k, v in methods.items()
    }
    metrics = {
        "track": "v9b_seed_ensemble_fuse",
        "v8_baseline_oof": V8_OOF,
        "v8_baseline_holdout": V8_HOLD,
        "use_tta": False,
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
        "protocol": "GroupKFold OOF; nested hyperparam on non-holdout; holdout last; MidFuse seed ensemble",
        "submission_v9": str(sub),
        "win_criteria": "nested_OOF>=0.548 or holdout>=0.566",
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    top = sorted(methods_out.items(), key=lambda kv: kv[1]["oof_acc_nested"], reverse=True)[:12]
    print("TOP nested:", flush=True)
    for k, v in top:
        print(f"  {v['oof_acc_nested']:.4f} hold={v['holdout_acc']:.4f} {k}", flush=True)
    print(json.dumps({k: metrics[k] for k in (
        "selected_method", "selected_oof_acc_nested", "selected_holdout_acc",
        "delta_oof_vs_v8", "delta_holdout_vs_v8", "clear_win", "overwrite_submission",
        "ping_disk_saver", "elapsed_sec")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
