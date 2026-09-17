"""v8 refine late-fuse / stacker: MidFuse + ST-GCN (+ GRU-attn).

Protocol (no holdout peek for selection):
- Collect GroupKFold OOF logits for each branch.
- Select fusion / train stacker on NON-HOLDOUT OOF only.
  Optimistic in-sample scores are NOT used for selection when a nested
  GroupKFold estimate is available (stackers, per-class, conf-gate).
- Evaluate holdout LAST with holdout-trained checkpoints.
- Overwrite track submission.csv only if OOF >= v7+0.005 or holdout >= 0.55.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

V7 = Path(__file__).resolve().parent.parent / "v7_stgcn"
V2 = Path(__file__).resolve().parent.parent / "skeleton_imu_v2"
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

V7_OOF = 0.5235
V7_HOLD = 0.5426
NUM_CLASSES = 40


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v8", V2 / "model.py")
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


@torch.no_grad()
def predict_ds(model, ds, device, batch=64):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    outs = []
    for xs, xi, y, _u, flag in loader:
        outs.append(model(xs.to(device), xi.to(device), flag.to(device)).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


def collect_oof(branches, X_skel, X_imu, y, users, has_imu, device, n_splits=5):
    n = len(y)
    oof = {k: np.zeros((n, NUM_CLASSES), dtype=np.float32) for k in branches}
    gkf = GroupKFold(n_splits=n_splits)
    for fi, (tr, va) in enumerate(gkf.split(np.arange(n), y, users)):
        ds = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False)
        for name, meta in branches.items():
            m, _ = load_ckpt(
                meta["kind"],
                meta["fold_dir"] / f"best_fold{fi}.pt",
                device,
                meta["v2m"],
                meta["default"],
            )
            oof[name][va] = predict_ds(m, ds, device)
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"OOF fold{fi} done", flush=True)
    return oof


def holdout_logits(branches, X_skel, X_imu, y, users, has_imu, ho_idx, device):
    ds = CachedDualDataset(X_skel, X_imu, y, users, ho_idx, has_imu, augment=False)
    out = {}
    for name, meta in branches.items():
        m, _ = load_ckpt(meta["kind"], meta["holdout_ckpt"], device, meta["v2m"], meta["default"])
        out[name] = predict_ds(m, ds, device)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def fit_global_alpha2(Sm, Ss, y):
    best_a, best_acc = 0.5, -1.0
    rows = []
    for a in [round(x, 2) for x in np.linspace(0.0, 1.0, 51)]:
        a_acc = acc((a * Ss + (1 - a) * Sm).argmax(1), y)
        rows.append({"alpha": a, "oof_acc": a_acc})
        if a_acc > best_acc:
            best_a, best_acc = a, a_acc
    return {"alpha": best_a, "oof_acc": best_acc, "rows": rows}


def apply_alpha2(Sm, Ss, a):
    return a * Ss + (1 - a) * Sm


def fit_global_w3(probs_list, y, grid=13):
    best_w, best_acc = (1 / 3, 1 / 3, 1 / 3), -1.0
    vals = np.linspace(0.0, 1.0, grid)
    n_tried = 0
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
            mix = ww[0] * probs_list[0] + ww[1] * probs_list[1] + ww[2] * probs_list[2]
            a = acc(mix.argmax(1), y)
            n_tried += 1
            if a > best_acc:
                best_w, best_acc = ww, a
    return {"w": best_w, "oof_acc": best_acc, "n_tried": n_tried}


def apply_w3(probs_list, w):
    mix = w[0] * probs_list[0]
    for i in range(1, len(probs_list)):
        mix = mix + w[i] * probs_list[i]
    return mix


def _fit_per_class_alphas(Sm, Ss, y, n_passes=2):
    a = np.full(NUM_CLASSES, 0.5, dtype=np.float64)
    alphas = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
    best_acc = acc((0.5 * Ss + 0.5 * Sm).argmax(1), y)
    for _ in range(n_passes):
        improved = False
        for c in range(NUM_CLASSES):
            local_best, local_acc = a[c], best_acc
            for cand in alphas:
                aa = a.copy()
                aa[c] = cand
                mix = aa[None, :] * Ss + (1.0 - aa[None, :]) * Sm
                ca = acc(mix.argmax(1), y)
                if ca > local_acc + 1e-12:
                    local_acc, local_best = ca, cand
            if local_best != a[c]:
                a[c] = local_best
                best_acc = local_acc
                improved = True
        if not improved:
            break
    return a, best_acc


def apply_per_class_alpha2(Sm, Ss, a):
    aa = np.asarray(a, dtype=np.float64)[None, :]
    return aa * Ss + (1.0 - aa) * Sm


def nested_per_class_oof(Sm, Ss, y, users):
    """Honest nested GroupKFold OOF for per-class alphas."""
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), y, users):
        a, _ = _fit_per_class_alphas(Sm[tr], Ss[tr], y[tr])
        mix = apply_per_class_alpha2(Sm[va], Ss[va], a)
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def _fit_conf_gate(Sm, Ss, y):
    best = {"mode": "mf_gate", "t": 0.5, "alpha": 0.5, "oof_acc": -1.0}
    for mode in ("mf_gate", "st_gate", "max_gate"):
        for t in [round(x, 2) for x in np.linspace(0.3, 0.9, 13)]:
            for a in (0.3, 0.4, 0.5, 0.6):
                mix = apply_conf_gate(Sm, Ss, {"mode": mode, "t": t, "alpha": a})
                ca = acc(mix.argmax(1), y)
                if ca > best["oof_acc"]:
                    best = {"mode": mode, "t": t, "alpha": a, "oof_acc": ca}
    return best


def apply_conf_gate(Sm, Ss, cfg):
    a, t, mode = cfg["alpha"], cfg["t"], cfg["mode"]
    blend = a * Ss + (1 - a) * Sm
    mix = blend.copy()
    if mode == "mf_gate":
        take = Sm.max(1) >= t
        mix[take] = Sm[take]
    elif mode == "st_gate":
        take = Ss.max(1) >= t
        mix[take] = Ss[take]
    else:
        cm, cs = Sm.max(1), Ss.max(1)
        take_m = (cm >= t) & (cm >= cs)
        take_s = (cs >= t) & (cs > cm)
        mix[take_m] = Sm[take_m]
        mix[take_s] = Ss[take_s]
    return mix


def nested_conf_gate_oof(Sm, Ss, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = _fit_conf_gate(Sm[tr], Ss[tr], y[tr])
        mix = apply_conf_gate(Sm[va], Ss[va], cfg)
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def fit_conf_weighted(probs_list, y):
    confs = [p.max(1, keepdims=True) for p in probs_list]
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in (0.25, 0.5, 1.0, 2.0, 4.0):
        mix = apply_conf_weighted(probs_list, temp)
        ca = acc(mix.argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": temp, "oof_acc": ca}
    return best


def apply_conf_weighted(probs_list, temp):
    confs = [p.max(1, keepdims=True) for p in probs_list]
    logits_c = np.concatenate([np.log(np.maximum(c, 1e-8)) / temp for c in confs], axis=1)
    lc = logits_c - logits_c.max(1, keepdims=True)
    w = np.exp(lc)
    w = w / w.sum(1, keepdims=True)
    mix = sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list)))
    return mix


def nested_conf_weighted_oof(probs_list, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)  # K,N,C
    for tr, va in gkf.split(np.arange(n), y, users):
        plist_tr = [stacked[i, tr] for i in range(stacked.shape[0])]
        cfg = fit_conf_weighted(plist_tr, y[tr])
        plist_va = [stacked[i, va] for i in range(stacked.shape[0])]
        mix = apply_conf_weighted(plist_va, cfg["temp"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_w3_oof(probs_list, y, users, grid=9):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    stacked = np.stack(probs_list, axis=0)
    for tr, va in gkf.split(np.arange(n), y, users):
        plist_tr = [stacked[i, tr] for i in range(3)]
        cfg = fit_global_w3(plist_tr, y[tr], grid=grid)
        plist_va = [stacked[i, va] for i in range(3)]
        mix = apply_w3(plist_va, cfg["w"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def nested_alpha2_oof(Sm, Ss, y, users):
    n = len(y)
    pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), y, users):
        cfg = fit_global_alpha2(Sm[tr], Ss[tr], y[tr])
        mix = apply_alpha2(Sm[va], Ss[va], cfg["alpha"])
        pred[va] = mix.argmax(1)
    return acc(pred, y)


def _stack_features(probs_list):
    conf = [p.max(1, keepdims=True) for p in probs_list]
    return np.concatenate(list(probs_list) + conf, axis=1).astype(np.float32)


def make_stacker(kind):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        alpha=1e-3,
        max_iter=400,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )


def nested_stacker_oof(probs_list, y, users, kind="logreg"):
    X = _stack_features(probs_list)
    n = len(y)
    oof_pred = np.zeros(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), y, users):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = make_stacker(kind)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xva).astype(np.float32)
        oof_pred[va] = proba.argmax(1)
    return acc(oof_pred, y)


def fit_final_stacker(probs_list, y, kind="logreg"):
    X = _stack_features(probs_list)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = make_stacker(kind)
    clf.fit(Xs, y)
    return {"scaler": scaler, "clf": clf, "kind": kind}


def apply_stacker(probs_list, pack):
    X = _stack_features(probs_list)
    Xs = pack["scaler"].transform(X)
    return pack["clf"].predict_proba(Xs).astype(np.float32)


def test_branch_probs(branches, names, cache, device, n_folds=5):
    X_te, paths = load_skel_test_cache(cache)
    _, _, X_imu_te, has_imu_te = load_imu_caches(cache)
    has = has_imu_te if has_imu_te is not None else np.ones(len(X_te), bool)
    ds = TestDual(X_te, X_imu_te, has)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    avg = {n: None for n in names}
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
            p = softmax(predict_logits(m, loader, device, True))
            avg[n] = p if avg[n] is None else avg[n] + p
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"test fold{fi} done", flush=True)
    for n in names:
        avg[n] /= float(n_folds)
    return avg, paths


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = V7 / "cache"
    v2m = load_v2_model_mod()

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
    }

    X_skel, y, users, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    n = len(y)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]
    print(f"n={n} nh={len(nh_idx)} ho={len(ho_idx)} device={device}", flush=True)

    oof_path = ROOT / "oof_logits.npz"
    if oof_path.exists():
        z = np.load(oof_path, allow_pickle=False)
        oof = {k: z[k] for k in ("midfuse", "stgcn", "gru")}
        print("Loaded cached OOF logits", flush=True)
    else:
        oof = collect_oof(branches, X_skel, X_imu, y, users, has_imu, device)
        np.savez_compressed(oof_path, **oof, y=y, users=users)
        print(f"Saved {oof_path}", flush=True)

    yt_nh = y[nh_idx]
    users_nh = users[nh_idx]
    Sm = softmax(oof["midfuse"][nh_idx])
    Ss = softmax(oof["stgcn"][nh_idx])
    Sg = softmax(oof["gru"][nh_idx])

    single = {
        "oof_midfuse": acc(oof["midfuse"][nh_idx].argmax(1), yt_nh),
        "oof_stgcn": acc(oof["stgcn"][nh_idx].argmax(1), yt_nh),
        "oof_gru": acc(oof["gru"][nh_idx].argmax(1), yt_nh),
        "agree_mf_st": float((oof["midfuse"][nh_idx].argmax(1) == oof["stgcn"][nh_idx].argmax(1)).mean()),
        "agree_mf_gru": float((oof["midfuse"][nh_idx].argmax(1) == oof["gru"][nh_idx].argmax(1)).mean()),
        "agree_st_gru": float((oof["stgcn"][nh_idx].argmax(1) == oof["gru"][nh_idx].argmax(1)).mean()),
    }
    print(json.dumps(single, indent=2), flush=True)

    methods = {}

    # --- selection scores use nested OOF where hyperparams are fit ---
    a2 = fit_global_alpha2(Sm, Ss, yt_nh)
    a2_nested = nested_alpha2_oof(Sm, Ss, yt_nh, users_nh)
    methods["alpha2_mf_st"] = {
        "oof_acc": a2_nested,
        "oof_acc_in_sample": a2["oof_acc"],
        "params": {"alpha_stgcn": a2["alpha"]},
        "family": "alpha2",
        "branches": ["midfuse", "stgcn"],
    }
    print(f"alpha2_mf_st nested={a2_nested:.4f} in_sample={a2['oof_acc']:.4f} a={a2['alpha']}", flush=True)

    a2g = fit_global_alpha2(Sm, Sg, yt_nh)
    a2g_nested = nested_alpha2_oof(Sm, Sg, yt_nh, users_nh)
    methods["alpha2_mf_gru"] = {
        "oof_acc": a2g_nested,
        "oof_acc_in_sample": a2g["oof_acc"],
        "params": {"alpha_other": a2g["alpha"]},
        "family": "alpha2_mg",
        "branches": ["midfuse", "gru"],
    }
    print(f"alpha2_mf_gru nested={a2g_nested:.4f} in_sample={a2g['oof_acc']:.4f}", flush=True)

    w3 = fit_global_w3([Sm, Ss, Sg], yt_nh, grid=13)
    w3_nested = nested_w3_oof([Sm, Ss, Sg], yt_nh, users_nh, grid=9)
    methods["w3_mf_st_gru"] = {
        "oof_acc": w3_nested,
        "oof_acc_in_sample": w3["oof_acc"],
        "params": {"w_mf_st_gru": list(w3["w"])},
        "family": "w3",
        "branches": ["midfuse", "stgcn", "gru"],
    }
    print(f"w3 nested={w3_nested:.4f} in_sample={w3['oof_acc']:.4f} w={w3['w']}", flush=True)

    pc_a, pc_in = _fit_per_class_alphas(Sm, Ss, yt_nh)
    pc_nested = nested_per_class_oof(Sm, Ss, yt_nh, users_nh)
    methods["per_class_alpha2_mf_st"] = {
        "oof_acc": pc_nested,
        "oof_acc_in_sample": pc_in,
        "params": {"alpha_per_class": pc_a.tolist()},
        "family": "per_class",
        "branches": ["midfuse", "stgcn"],
    }
    print(f"per_class nested={pc_nested:.4f} in_sample={pc_in:.4f}", flush=True)

    cg = _fit_conf_gate(Sm, Ss, yt_nh)
    cg_nested = nested_conf_gate_oof(Sm, Ss, yt_nh, users_nh)
    methods["conf_gate_mf_st"] = {
        "oof_acc": cg_nested,
        "oof_acc_in_sample": cg["oof_acc"],
        "params": {k: cg[k] for k in ("mode", "t", "alpha")},
        "family": "conf_gate",
        "branches": ["midfuse", "stgcn"],
    }
    print(f"conf_gate nested={cg_nested:.4f} in_sample={cg['oof_acc']:.4f}", flush=True)

    cw = fit_conf_weighted([Sm, Ss, Sg], yt_nh)
    cw_nested = nested_conf_weighted_oof([Sm, Ss, Sg], yt_nh, users_nh)
    methods["conf_weighted_3"] = {
        "oof_acc": cw_nested,
        "oof_acc_in_sample": cw["oof_acc"],
        "params": {"temp": cw["temp"]},
        "family": "conf_w",
        "branches": ["midfuse", "stgcn", "gru"],
    }
    print(f"conf_w nested={cw_nested:.4f} in_sample={cw['oof_acc']:.4f}", flush=True)

    for kind, key, plist, blist in [
        ("logreg", "stack_logreg_2", [Sm, Ss], ["midfuse", "stgcn"]),
        ("logreg", "stack_logreg_3", [Sm, Ss, Sg], ["midfuse", "stgcn", "gru"]),
        ("mlp", "stack_mlp_2", [Sm, Ss], ["midfuse", "stgcn"]),
        ("mlp", "stack_mlp_3", [Sm, Ss, Sg], ["midfuse", "stgcn", "gru"]),
    ]:
        oof_a = nested_stacker_oof(plist, yt_nh, users_nh, kind=kind)
        methods[key] = {
            "oof_acc": oof_a,
            "oof_acc_in_sample": None,
            "params": {"kind": kind},
            "family": f"stack_{kind}",
            "branches": blist,
            "nested": True,
        }
        print(f"{key} nested OOF={oof_a:.4f}", flush=True)

    best_name = max(methods, key=lambda k: methods[k]["oof_acc"])
    best = methods[best_name]
    print(f"BEST method={best_name} nested_oof={best['oof_acc']:.4f}", flush=True)

    # Holdout LAST
    H = holdout_logits(branches, X_skel, X_imu, y, users, has_imu, ho_idx, device)
    yt_h = y[ho_idx]
    Hm, Hs, Hg = softmax(H["midfuse"]), softmax(H["stgcn"]), softmax(H["gru"])
    hold_single = {
        "holdout_midfuse": acc(H["midfuse"].argmax(1), yt_h),
        "holdout_stgcn": acc(H["stgcn"].argmax(1), yt_h),
        "holdout_gru": acc(H["gru"].argmax(1), yt_h),
    }

    # Fit final params on full nh; apply to holdout / test
    final_packs = {}
    for name, m in methods.items():
        fam = m["family"]
        if fam.startswith("stack_"):
            bl = m["branches"]
            plist = [{"midfuse": Sm, "stgcn": Ss, "gru": Sg}[b] for b in bl]
            final_packs[name] = fit_final_stacker(plist, yt_nh, kind=m["params"]["kind"])

    def apply_named(name, Pm, Ps, Pg):
        m = methods[name]
        fam, p = m["family"], m["params"]
        if fam == "alpha2":
            return apply_alpha2(Pm, Ps, p["alpha_stgcn"])
        if fam == "alpha2_mg":
            return apply_alpha2(Pm, Pg, p["alpha_other"])
        if fam == "w3":
            return apply_w3([Pm, Ps, Pg], p["w_mf_st_gru"])
        if fam == "per_class":
            return apply_per_class_alpha2(Pm, Ps, p["alpha_per_class"])
        if fam == "conf_gate":
            return apply_conf_gate(Pm, Ps, p)
        if fam == "conf_w":
            return apply_conf_weighted([Pm, Ps, Pg], p["temp"])
        if fam.startswith("stack_"):
            bl = m["branches"]
            plist = [{"midfuse": Pm, "stgcn": Ps, "gru": Pg}[b] for b in bl]
            return apply_stacker(plist, final_packs[name])
        raise ValueError(name)

    hold_rows = {name: acc(apply_named(name, Hm, Hs, Hg).argmax(1), yt_h) for name in methods}
    hold_best = hold_rows[best_name]
    peek_best_name = max(hold_rows, key=lambda k: hold_rows[k])
    peek = {
        "best_holdout_method_peek": peek_best_name,
        "best_holdout_acc_peek": hold_rows[peek_best_name],
        "note": "NOT used for selection or overwrite decision",
    }

    oof_best = best["oof_acc"]
    delta_oof = oof_best - V7_OOF
    overwrite = bool(oof_best >= V7_OOF + 0.005 or hold_best >= 0.55)
    ping = bool((hold_best > 0.543 or oof_best > 0.53) and overwrite)

    # Test submission for selected method
    test_probs, paths = test_branch_probs(branches, ["midfuse", "stgcn", "gru"], cache, device)
    Tm, Ts, Tg = test_probs["midfuse"], test_probs["stgcn"], test_probs["gru"]
    mix_te = apply_named(best_name, Tm, Ts, Tg)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub_v8 = ROOT / "submission_v8.csv"
    df.to_csv(sub_v8, index=False)
    np.savez_compressed(
        ROOT / "submission_v8_probs.npz",
        probs=mix_te,
        paths=np.array(paths, dtype=object),
        method=np.array(best_name),
    )

    overwritten = False
    if overwrite:
        df.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submissions").mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v8.csv", index=False)
        overwritten = True

    if "per_class_alpha2_mf_st" in methods:
        np.save(
            ROOT / "per_class_alphas.npy",
            np.array(methods["per_class_alpha2_mf_st"]["params"]["alpha_per_class"]),
        )

    methods_out = {}
    for k, v in methods.items():
        params = v["params"]
        if k == "per_class_alpha2_mf_st":
            params = {"alpha_per_class_file": "per_class_alphas.npy"}
        methods_out[k] = {
            "oof_acc_nested": v["oof_acc"],
            "oof_acc_in_sample": v.get("oof_acc_in_sample"),
            "holdout_acc": hold_rows[k],
            "family": v["family"],
            "branches": v["branches"],
            "params": params,
        }

    metrics = {
        "track": "v8_refine_fuse",
        "v7_baseline_oof": V7_OOF,
        "v7_baseline_holdout": V7_HOLD,
        "single_oof": single,
        "single_holdout": hold_single,
        "methods": methods_out,
        "selected_method": best_name,
        "selected_oof_acc_nested": oof_best,
        "selected_holdout_acc": hold_best,
        "delta_oof_vs_v7": delta_oof,
        "delta_holdout_vs_v7": hold_best - V7_HOLD,
        "overwrite_submission": overwrite,
        "submission_overwrote_track": overwritten,
        "ping_disk_saver": ping,
        "holdout_peek_not_used_for_selection": peek,
        "elapsed_sec": time.time() - t0,
        "finished_at_unix": time.time(),
        "protocol": "GroupKFold OOF; nested hyperparam fit on non-holdout; holdout last",
        "submission_v8": str(sub_v8),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Wrote {sub_v8}; overwrite={overwritten}; ping={ping}", flush=True)


if __name__ == "__main__":
    main()
