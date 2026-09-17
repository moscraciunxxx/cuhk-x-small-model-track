"""v10b: advanced nested fusion on existing OOF (+ optional gru2 / mf extras).

Extra recipes beyond v10:
  - exact v9 blend_ABD replay
  - logit-mean / rank-average
  - nested branch-subset search
  - light nested logistic stacker (strong C only; skipped if nested gap huge)
  - method blends of diverse top families

Clear-win vs v9: nested >= 0.558 OR holdout >= 0.576.
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
from sklearn.linear_model import LogisticRegression
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
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
TEMPS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v10b", V2 / "model.py")
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


def fit_power_mean(probs_list, y):
    best = {"p": 1.0, "oof_acc": -1.0}
    for p in POWERS:
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


def fit_conf_weighted(probs_list, y):
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in TEMPS:
        ca = acc(apply_conf_weighted(probs_list, temp).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_equal(probs_list):
    return (sum(probs_list) / float(len(probs_list))).astype(np.float32)


def apply_logit_mean(logits_list):
    return softmax(sum(logits_list) / float(len(logits_list)))


def apply_rank_avg(probs_list):
    # higher prob -> better rank; average ranks then invert
    ranks = []
    for p in probs_list:
        # rank ascending so argmax gets highest rank number
        order = np.argsort(p, axis=1)
        r = np.empty_like(order, dtype=np.float64)
        # assign ranks 1..C
        rows = np.arange(p.shape[0])[:, None]
        r[rows, order] = np.arange(1, p.shape[1] + 1)[None, :]
        ranks.append(r)
    mean_r = sum(ranks) / float(len(ranks))
    # convert ranks to soft scores
    scores = mean_r
    scores = scores / scores.sum(1, keepdims=True)
    return scores.astype(np.float32)


def folds_complete(fold_dir, n=5):
    return all((fold_dir / f"best_fold{i}.pt").exists() for i in range(n)) and (
        fold_dir / "best_holdout.pt"
    ).exists()


def nested_family(keys, P, L, yt, us, family):
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), yt, us):
        if family == "eq":
            out[va] = apply_equal([P[k][va] for k in keys])
        elif family == "pow":
            cfg = fit_power_mean([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_power_mean([P[k][va] for k in keys], cfg["p"])
        elif family == "conf":
            cfg = fit_conf_weighted([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_conf_weighted([P[k][va] for k in keys], cfg["temp"])
        elif family == "logit_mean":
            out[va] = apply_logit_mean([L[k][va] for k in keys])
        elif family == "rank":
            out[va] = apply_rank_avg([P[k][va] for k in keys])
        else:
            raise ValueError(family)
    return out


def fit_apply_family(keys, P, L, y, family, params=None):
    if family == "eq":
        return apply_equal([P[k] for k in keys]), {}
    if family == "pow":
        cfg = fit_power_mean([P[k] for k in keys], y) if params is None else params
        return apply_power_mean([P[k] for k in keys], cfg["p"]), cfg
    if family == "conf":
        cfg = fit_conf_weighted([P[k] for k in keys], y) if params is None else params
        return apply_conf_weighted([P[k] for k in keys], cfg["temp"]), cfg
    if family == "logit_mean":
        return apply_logit_mean([L[k] for k in keys]), {}
    if family == "rank":
        return apply_rank_avg([P[k] for k in keys]), {}
    raise ValueError(family)


def nested_subset_search(cand_keys, P, yt, us, family="pow", max_k=4):
    """For each outer fold, pick best subset on train via inner GroupKFold-ish hold of groups.
    Simplified: pick subset by in-fold train accuracy of family fit (mildly optimistic inside),
    but outer val is still honest. Better: use inner split.
    """
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    chosen = []
    for tr, va in gkf.split(np.arange(n), yt, us):
        # inner split on tr
        gkf2 = GroupKFold(n_splits=3)
        # evaluate each subset by mean inner nested score on tr
        best_subset = None
        best_score = -1.0
        # limit combinations
        subsets = []
        for k in range(2, min(max_k, len(cand_keys)) + 1):
            subsets.extend(list(combinations(cand_keys, k)))
        # cap
        if len(subsets) > 40:
            # prefer ones including mf_avg or midfuse
            pref = [s for s in subsets if ("mf_avg" in s or "midfuse" in s)]
            subsets = pref[:40] if pref else subsets[:40]
        for subset in subsets:
            inner_accs = []
            try:
                for tr2, va2 in gkf2.split(tr, yt[tr], us[tr]):
                    idx_tr = tr[tr2]
                    idx_va = tr[va2]
                    if family == "pow":
                        cfg = fit_power_mean([P[k][idx_tr] for k in subset], yt[idx_tr])
                        pred = apply_power_mean([P[k][idx_va] for k in subset], cfg["p"]).argmax(1)
                    elif family == "eq":
                        pred = apply_equal([P[k][idx_va] for k in subset]).argmax(1)
                    else:
                        cfg = fit_conf_weighted([P[k][idx_tr] for k in subset], yt[idx_tr])
                        pred = apply_conf_weighted([P[k][idx_va] for k in subset], cfg["temp"]).argmax(1)
                    inner_accs.append(acc(pred, yt[idx_va]))
            except Exception:
                continue
            if not inner_accs:
                continue
            sc = float(np.mean(inner_accs))
            if sc > best_score:
                best_score = sc
                best_subset = subset
        if best_subset is None:
            best_subset = tuple(cand_keys[:3])
        chosen.append(list(best_subset))
        if family == "pow":
            cfg = fit_power_mean([P[k][tr] for k in best_subset], yt[tr])
            out[va] = apply_power_mean([P[k][va] for k in best_subset], cfg["p"])
        elif family == "eq":
            out[va] = apply_equal([P[k][va] for k in best_subset])
        else:
            cfg = fit_conf_weighted([P[k][tr] for k in best_subset], yt[tr])
            out[va] = apply_conf_weighted([P[k][va] for k in best_subset], cfg["temp"])
    return out, chosen


def nested_logit_stack(keys, P, yt, us, C=0.01):
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    X_all = np.concatenate([P[k] for k in keys], axis=1)
    for tr, va in gkf.split(np.arange(n), yt, us):
        clf = LogisticRegression(
            C=C,
            max_iter=500,
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(X_all[tr], yt[tr])
        out[va] = clf.predict_proba(X_all[va]).astype(np.float32)
    return out


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
    if folds_complete(V2 / "checkpoints_midfuse_s7"):
        branches["midfuse3"] = {
            "kind": "v2", "v2m": v2m, "default": "midfuse", "tag": "midfuse3",
            "fold_dir": V2 / "checkpoints_midfuse_s7",
            "holdout_ckpt": V2 / "checkpoints_midfuse_s7" / "best_holdout.pt",
        }
    if folds_complete(V2 / "checkpoints_gru_s7"):
        branches["gru2"] = {
            "kind": "v2", "v2m": v2m, "default": "gru_attn", "tag": "gru2",
            "fold_dir": V2 / "checkpoints_gru_s7",
            "holdout_ckpt": V2 / "checkpoints_gru_s7" / "best_holdout.pt",
        }
    if folds_complete(V2 / "checkpoints_compact_fuse"):
        branches["cfuse"] = {
            "kind": "v2", "v2m": v2m, "default": "compact_fuse", "tag": "cfuse",
            "fold_dir": V2 / "checkpoints_compact_fuse",
            "holdout_ckpt": V2 / "checkpoints_compact_fuse" / "best_holdout.pt",
        }

    oof_path = ROOT / "oof_logits_v10.npz"
    oof = {}
    for src in (oof_path, V9 / "oof_logits_v9b.npz"):
        if src.exists():
            z = np.load(src, allow_pickle=False)
            for k in z.files:
                if k not in oof:
                    oof[k] = z[k]
            print(f"seeded OOF from {src.name}", flush=True)
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
    print(
        f"n={len(y)} nh={len(nh_idx)} ho={len(ho_idx)} device={device} branches={list(branches)}",
        flush=True,
    )

    L = {k: oof[k][nh_idx] for k in branches}
    P = {k: softmax(L[k]) for k in branches}
    mf_keys = [k for k in ("midfuse", "midfuse2", "midfuse3") if k in branches]
    gru_keys = [k for k in ("gru", "gru2") if k in branches]
    L["mf_avg"] = sum(L[k] for k in mf_keys) / float(len(mf_keys))
    P["mf_avg"] = softmax(L["mf_avg"])
    L["mf12"] = 0.5 * (L["midfuse"] + L["midfuse2"])
    P["mf12"] = softmax(L["mf12"])
    if len(gru_keys) >= 2:
        L["gru_avg"] = sum(L[k] for k in gru_keys) / float(len(gru_keys))
        P["gru_avg"] = softmax(L["gru_avg"])

    single = {f"oof_{k}": acc(L[k].argmax(1), yt_nh) for k in list(branches) + ["mf_avg"]}
    print(json.dumps(single, indent=2), flush=True)

    methods = {}
    nested_probs = {}

    def register(name, family, keys, np_, params=None, in_sample=None):
        nested_probs[name] = np_
        methods[name] = {
            "oof_acc": acc(np_.argmax(1), yt_nh),
            "oof_acc_in_sample": in_sample,
            "params": params or {},
            "family": family,
            "keys": list(keys),
        }
        print(f"{name} nested={methods[name]['oof_acc']:.4f}", flush=True)

    # Core recipes
    recipe_list = [
        ("eq_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "eq"),
        ("conf_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "conf"),
        ("pow_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "pow"),
        ("logit_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "logit_mean"),
        ("rank_mfavg_st_gru", ["mf_avg", "stgcn", "gru"], "rank"),
        ("eq_mf12_st_gru", ["mf12", "stgcn", "gru"], "eq"),
        ("conf_mf12_st_gru", ["mf12", "stgcn", "gru"], "conf"),
        ("pow_mf12_st_gru", ["midfuse", "midfuse2", "stgcn", "gru"], "pow"),
        ("logit_mf12_st_gru", ["midfuse", "midfuse2", "stgcn", "gru"], "logit_mean"),
        ("pow_all5", ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "pow"),
        ("eq_all5", ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "eq"),
        ("logit_all5", ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "logit_mean"),
        ("rank_all5", ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "rank"),
        ("pow_mfs_st_gru", mf_keys + ["stgcn", "gru"], "pow"),
        ("logit_mfs_st_gru", mf_keys + ["stgcn", "gru"], "logit_mean"),
        ("eq_mfs_st_gru", ["mf_avg", "stgcn", "gru"] if False else mf_keys + ["stgcn", "gru"], "eq"),
        ("pow_mfs_st_gru_dc", mf_keys + ["stgcn", "gru", "deepconv"], "pow"),
        ("conf_mfs_st_gru", mf_keys + ["stgcn", "gru"], "conf"),
    ]
    if "cfuse" in P:
        recipe_list += [
            ("eq_mfavg_st_gru_cf", ["mf_avg", "stgcn", "gru", "cfuse"], "eq"),
            ("pow_mfavg_st_gru_cf", ["mf_avg", "stgcn", "gru", "cfuse"], "pow"),
            ("conf_mfavg_st_gru_cf", ["mf_avg", "stgcn", "gru", "cfuse"], "conf"),
            ("pow_mfs_st_gru_cf", mf_keys + ["stgcn", "gru", "cfuse"], "pow"),
            ("logit_mfs_st_gru_cf", mf_keys + ["stgcn", "gru", "cfuse"], "logit_mean"),
            ("pow_all_cf", mf_keys + ["stgcn", "gru", "deepconv", "cfuse"], "pow"),
            ("pow_all_grus_cf", mf_keys + ["stgcn"] + gru_keys + ["deepconv", "cfuse"] if "gru2" in branches else mf_keys + ["stgcn", "gru", "deepconv", "cfuse"], "pow"),
            ("eq_mfavg_gru_cf", ["mf_avg", "gru", "cfuse"], "eq"),
            ("pow_mf12_st_gru_cf", ["midfuse", "midfuse2", "stgcn", "gru", "cfuse"], "pow"),
        ]
    if "gru_avg" in P:
        recipe_list += [
            ("eq_mfavg_st_gruavg", ["mf_avg", "stgcn", "gru_avg"], "eq"),
            ("conf_mfavg_st_gruavg", ["mf_avg", "stgcn", "gru_avg"], "conf"),
            ("pow_mfavg_st_gruavg", ["mf_avg", "stgcn", "gru_avg"], "pow"),
            ("pow_mfs_st_grus", mf_keys + ["stgcn"] + gru_keys, "pow"),
            ("logit_mfs_st_grus", mf_keys + ["stgcn"] + gru_keys, "logit_mean"),
            ("pow_all_grus", mf_keys + ["stgcn"] + gru_keys + ["deepconv"], "pow"),
        ]

    for name, keys, fam in recipe_list:
        if any(k not in P and k not in L for k in keys):
            continue
        np_ = nested_family(keys, P, L, yt_nh, users_nh, fam)
        mix, cfg = fit_apply_family(keys, P, L, yt_nh, fam)
        register(name, fam, keys, np_, {k: v for k, v in cfg.items() if k != "oof_acc"}, cfg.get("oof_acc"))

    # Exact v9 ABD components + blend
    for nm in ("pow_all5", "eq_mf12_st_gru", "pow_mf12_st_gru"):
        assert nm in nested_probs, nm
    # v9 used eq_mfavg with 2-seed avg and pow_mf_mf2 and pow_all5
    # Reconstruct with mf12 (2-seed) to match v9
    if "eq_mfavg_st_gru" in nested_probs:
        # If mf3 present, eq_mfavg uses 3-seed; also make explicit 2-seed version
        np_ = nested_family(["mf12", "stgcn", "gru"], P, L, yt_nh, users_nh, "eq")
        register("eq_mf12avg_st_gru", "eq", ["mf12", "stgcn", "gru"], np_)

    abd_members = ["pow_all5", "eq_mf12avg_st_gru" if "eq_mf12avg_st_gru" in nested_probs else "eq_mf12_st_gru", "pow_mf12_st_gru"]
    mix = sum(nested_probs[m] for m in abd_members) / 3.0
    register("blend_ABD_v9replay", "method_blend", abd_members, mix, {"members": abd_members, "w": "equal"})

    # Nested subset search
    cand = mf_keys + ["stgcn", "gru", "deepconv"] + (["cfuse"] if "cfuse" in branches else [])
    if "gru2" in branches:
        cand.append("gru2")
    print("subset search...", flush=True)
    np_sub, chosen = nested_subset_search(cand, P, yt_nh, users_nh, family="pow", max_k=4)
    register("subset_pow", "subset_pow", cand, np_sub, {"chosen_per_fold": chosen})
    np_sub2, chosen2 = nested_subset_search(
        ["mf_avg", "mf12", "stgcn", "gru", "deepconv"] + ([ "gru_avg"] if "gru_avg" in P else []),
        P,
        yt_nh,
        users_nh,
        family="eq",
        max_k=3,
    )
    register("subset_eq", "subset_eq", ["mf_avg", "stgcn", "gru"], np_sub2, {"chosen_per_fold": chosen2})

    # Logistic stacker (cautious)
    stack_keys = mf_keys + ["stgcn", "gru", "deepconv"] + (["cfuse"] if "cfuse" in branches else [])
    for C in (0.001, 0.01, 0.1):
        print(f"logit stack C={C}...", flush=True)
        np_ = nested_logit_stack(stack_keys, P, yt_nh, users_nh, C=C)
        # in-sample fit for gap
        X_all = np.concatenate([P[k] for k in stack_keys], axis=1)
        clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
        clf.fit(X_all, yt_nh)
        in_s = acc(clf.predict(X_all), yt_nh)
        nested_a = acc(np_.argmax(1), yt_nh)
        gap = in_s - nested_a
        name = f"logitstack_C{C}"
        register(name, "logitstack", stack_keys, np_, {"C": C, "gap": gap}, in_s)
        if gap > 0.05:
            print(f"  WARNING large gap {gap:.4f} for {name}", flush=True)

    # Method blends of diverse tops
    base_sorted = sorted(
        [k for k, v in methods.items() if v["family"] not in ("method_blend", "subset_pow", "subset_eq", "logitstack")],
        key=lambda k: methods[k]["oof_acc"],
        reverse=True,
    )
    # force include some diverse
    diverse = []
    for k in base_sorted:
        if k not in diverse:
            diverse.append(k)
        if len(diverse) >= 6:
            break
    for extra in ("blend_ABD_v9replay", "eq_mf12_st_gru", "conf_mf12_st_gru", "pow_mfs_st_gru", "logit_mfs_st_gru"):
        if extra in nested_probs and extra not in diverse:
            diverse.append(extra)

    # equal blends size 2-3 of top5
    top5 = base_sorted[:5]
    for r in (2, 3):
        for comb in combinations(top5, r):
            name = "eqblend_" + "_".join(comb)
            if len(name) > 90:
                name = f"eqblend_top{r}_" + str(hash(comb) % 10**8)
            mix = sum(nested_probs[m] for m in comb) / float(len(comb))
            register(name, "method_blend", list(comb), mix, {"members": list(comb), "w": "equal"})

    # Special: blend high-nested pow_mfs with high-stability eq/conf mf12
    specials = []
    for a in ("pow_mfs_st_gru", "logit_mfs_st_gru", "pow_mf12_st_gru"):
        for b in ("eq_mf12_st_gru", "conf_mf12_st_gru", "eq_mf12avg_st_gru", "blend_ABD_v9replay"):
            if a in nested_probs and b in nested_probs:
                specials.append((a, b))
    for a, b in specials:
        name = f"eqblend_{a}__{b}"
        mix = 0.5 * (nested_probs[a] + nested_probs[b])
        register(name, "method_blend", [a, b], mix, {"members": [a, b], "w": "equal"})
        # 2:1 toward higher nested member
        if methods[a]["oof_acc"] >= methods[b]["oof_acc"]:
            mix2 = (2 * nested_probs[a] + nested_probs[b]) / 3.0
            register(f"w21_{a}__{b}", "method_blend", [a, b], mix2, {"members": [a, b], "w": [2, 1]})
        else:
            mix2 = (nested_probs[a] + 2 * nested_probs[b]) / 3.0
            register(f"w12_{a}__{b}", "method_blend", [a, b], mix2, {"members": [a, b], "w": [1, 2]})

    best_name = max(methods, key=lambda k: methods[k]["oof_acc"])
    print(f"BEST={best_name} nested={methods[best_name]['oof_acc']:.4f}", flush=True)

    # Drop logitstack if nested worse than v9 and gap large
    # (still in table; selection is pure max nested)

    # Holdout LAST
    print("Holdout...", flush=True)
    H = {}
    for name in branches:
        H[name] = holdout_branch(branches[name], X_skel, X_imu, has_imu, ho_idx, device)
        print(f"  {name}", flush=True)
    yt_h = y[ho_idx]
    HL = dict(H)
    HL["mf_avg"] = sum(H[k] for k in mf_keys) / float(len(mf_keys))
    HL["mf12"] = 0.5 * (H["midfuse"] + H["midfuse2"])
    if len(gru_keys) >= 2:
        HL["gru_avg"] = sum(H[k] for k in gru_keys) / float(len(gru_keys))
    HP = {k: softmax(HL[k]) for k in HL}
    hold_single = {f"holdout_{k}": acc(H[k].argmax(1), yt_h) for k in branches}

    # Final params for base families
    final = {}
    for name, m in methods.items():
        if m["family"] in ("eq", "pow", "conf", "logit_mean", "rank"):
            _, cfg = fit_apply_family(m["keys"], P, L, yt_nh, m["family"])
            final[name] = {k: v for k, v in cfg.items() if k != "oof_acc"}
        elif m["family"] == "logitstack":
            final[name] = {"C": m["params"]["C"]}
        elif m["family"] in ("subset_pow", "subset_eq"):
            # fit best subset on full nh via inner search once
            fam = "pow" if m["family"] == "subset_pow" else "eq"
            # use majority chosen or re-search once
            np_tmp, chosen = nested_subset_search(m["keys"], P, yt_nh, users_nh, family=fam, max_k=4)
            # majority keys
            from collections import Counter
            flat = [tuple(c) for c in chosen]
            maj = Counter(flat).most_common(1)[0][0]
            final[name] = {"subset": list(maj), "family": fam}
        elif m["family"] == "method_blend":
            final[name] = m["params"]

    def apply_named(name, probs_map, logits_map, y_fit=None, P_fit=None, L_fit=None):
        m = methods[name]
        fam = m["family"]
        keys = m["keys"]
        if fam in ("eq", "pow", "conf", "logit_mean", "rank"):
            mix, _ = fit_apply_family(keys, probs_map, logits_map, None, fam, params=final.get(name) or None)
            # fit_apply with params
            if fam == "eq":
                return apply_equal([probs_map[k] for k in keys])
            if fam == "pow":
                return apply_power_mean([probs_map[k] for k in keys], final[name]["p"])
            if fam == "conf":
                return apply_conf_weighted([probs_map[k] for k in keys], final[name]["temp"])
            if fam == "logit_mean":
                return apply_logit_mean([logits_map[k] for k in keys])
            if fam == "rank":
                return apply_rank_avg([probs_map[k] for k in keys])
        if fam == "method_blend":
            members = m["params"]["members"]
            w = m["params"].get("w", "equal")
            mixes = [apply_named(mem, probs_map, logits_map) for mem in members]
            if w == "equal":
                return sum(mixes) / float(len(mixes))
            if isinstance(w, list):
                s = float(sum(w))
                return sum(w[i] * mixes[i] for i in range(len(mixes))) / s
            return sum(mixes) / float(len(mixes))
        if fam in ("subset_pow", "subset_eq"):
            subset = final[name]["subset"]
            sfam = final[name]["family"]
            if sfam == "pow":
                # need p from nh fit
                cfg = fit_power_mean([P[k] for k in subset], yt_nh)
                return apply_power_mean([probs_map[k] for k in subset], cfg["p"])
            return apply_equal([probs_map[k] for k in subset])
        if fam == "logitstack":
            C = final[name]["C"]
            X_fit = np.concatenate([P[k] for k in keys], axis=1)
            clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
            clf.fit(X_fit, yt_nh)
            X = np.concatenate([probs_map[k] for k in keys], axis=1)
            return clf.predict_proba(X).astype(np.float32)
        raise ValueError(fam)

    hold_rows = {}
    for name in methods:
        try:
            hold_rows[name] = acc(apply_named(name, HP, HL).argmax(1), yt_h)
        except Exception as e:
            print(f"hold fail {name}: {e}", flush=True)
            hold_rows[name] = -1.0

    oof_best = methods[best_name]["oof_acc"]
    hold_best = hold_rows[best_name]
    peek_best = max(hold_rows, key=lambda k: hold_rows[k])
    clear_win = bool(oof_best >= WIN_OOF or hold_best >= WIN_HOLD)
    overwrite = clear_win
    print(
        f"selected {best_name}: nested={oof_best:.4f} hold={hold_best:.4f} win={clear_win}",
        flush=True,
    )
    print(
        f"peek best hold={hold_rows[peek_best]:.4f} {peek_best} (NOT used)",
        flush=True,
    )

    # Resolve test branches
    def resolve_keys(name):
        m = methods[name]
        need = set()
        if m["family"] == "method_blend":
            for mem in m["params"]["members"]:
                need |= resolve_keys(mem)
        elif m["family"] in ("subset_pow", "subset_eq"):
            need |= set(final[name]["subset"])
        else:
            need |= set(m["keys"])
        return need

    raw_need = resolve_keys(best_name)
    resolved = set()
    for k in raw_need:
        if k == "mf_avg":
            resolved.update(mf_keys)
        elif k == "mf12":
            resolved.update(["midfuse", "midfuse2"])
        elif k == "gru_avg":
            resolved.update(gru_keys)
        else:
            resolved.add(k)
    need = sorted(resolved & set(branches.keys()))
    print(f"test branches: {need}", flush=True)
    test_probs, test_logits, paths = test_branch_logits(branches, need, cache, device)
    if set(mf_keys).issubset(test_logits):
        test_logits["mf_avg"] = sum(test_logits[k] for k in mf_keys) / float(len(mf_keys))
        test_probs["mf_avg"] = softmax(test_logits["mf_avg"])
    if "midfuse" in test_logits and "midfuse2" in test_logits:
        test_logits["mf12"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse2"])
        test_probs["mf12"] = softmax(test_logits["mf12"])
    if set(gru_keys).issubset(test_logits) and len(gru_keys) >= 2:
        test_logits["gru_avg"] = sum(test_logits[k] for k in gru_keys) / float(len(gru_keys))
        test_probs["gru_avg"] = softmax(test_logits["gru_avg"])

    mix_te = apply_named(best_name, test_probs, test_logits)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub = ROOT / "submission_v10b.csv"
    df.to_csv(sub, index=False)
    np.savez_compressed(
        ROOT / "submission_v10b_probs.npz",
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
        # also refresh submission_v10.csv
        df.to_csv(ROOT / "submission_v10.csv", index=False)

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
        "track": "v10b_advanced",
        "v9_baseline_oof": V9_OOF,
        "v9_baseline_holdout": V9_HOLD,
        "branches": list(branches.keys()),
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
        "ping_disk_saver": bool(clear_win),
        "holdout_peek_not_used_for_selection": {
            "best_holdout_method_peek": peek_best,
            "best_holdout_acc_peek": hold_rows[peek_best],
            "note": "NOT used for selection or overwrite decision",
        },
        "elapsed_sec": time.time() - t0,
        "finished_at_unix": time.time(),
        "win_criteria": f"nested_OOF>={WIN_OOF} or holdout>={WIN_HOLD}",
        "skipped": ["TTA", "radar_fill"],
        "submission_v10b": str(sub),
    }
    (ROOT / "metrics_v10b.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Also update metrics.json if better nested than previous v10
    prev = ROOT / "metrics.json"
    update_main = True
    if prev.exists():
        old = json.loads(prev.read_text(encoding="utf-8"))
        if old.get("selected_oof_acc_nested", -1) > oof_best and not clear_win:
            update_main = False
    if update_main:
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
                    "elapsed_sec",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
