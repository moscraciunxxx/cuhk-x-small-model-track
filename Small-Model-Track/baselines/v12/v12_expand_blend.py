"""v12: expand ABG weights + temp-scale + deepconv-diverse blends (nested discipline).

Baselines: v11 wABG_4_2_3 nested OOF 0.5614 / holdout 0.5663 (track submission).
Clear-win overwrite/ping only if:
  holdout >= 0.576 OR holdout >= 0.5713 (0.5663+0.005)
  AND nested OOF >= 0.558
Prefer (for reporting): holdout > 0.5663 with OOF >= 0.5614.
Selection: max nested OOF only; holdout last. No TTA, no sklearn stacker.
"""
from __future__ import annotations

import json
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

V7 = Path(__file__).resolve().parent.parent / "v7_stgcn"
V2 = Path(__file__).resolve().parent.parent / "skeleton_imu_v2"
V10 = Path(__file__).resolve().parent.parent / "v10"
V11 = Path(__file__).resolve().parent.parent / "v11"
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

NUM_CLASSES = 40
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
TEMPS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
BRANCH_TEMPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)

V11_OOF = 0.5614179719703215
V11_HOLD = 0.5663366336633663
V10_OOF = 0.5585325638911789
V10_HOLD = 0.5623762376237624
V9_OOF = 0.5486397361912614
V9_HOLD = 0.5663366336633663

# Clear-win thresholds for overwrite / ping_disk_saver
WIN_OOF_FLOOR = 0.558
WIN_HOLD_STRICT = 0.576
WIN_HOLD_DELTA = 0.005  # clearly > v11 hold by >= 0.005 => 0.5713
WIN_HOLD_CLEAR = V11_HOLD + WIN_HOLD_DELTA
PREF_OOF = 0.5614
PREF_HOLD = V11_HOLD  # prefer holdout > this


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v12", V2 / "model.py")
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


def softmax_temp(logits, temp):
    return softmax(logits / float(temp))


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


class ArrayDual(Dataset):
    def __init__(self, xs, xi, flag):
        self.xs = torch.from_numpy(np.asarray(xs, np.float32))
        self.xi = torch.from_numpy(np.asarray(xi, np.float32))
        self.flag = torch.from_numpy(np.asarray(flag, np.float32))

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
    return np.concatenate(outs, 0)


def apply_power_mean(probs_list, p):
    stacked = np.stack(probs_list, 0)
    if abs(p) < 1e-8:
        out = np.exp(np.mean(np.log(np.clip(stacked, 1e-12, 1)), 0))
    else:
        out = np.mean(stacked ** p, 0) ** (1.0 / p)
    out = out / np.maximum(out.sum(1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def fit_power_mean(probs_list, y):
    best = None
    for p in POWERS:
        pr = apply_power_mean(probs_list, p)
        a = acc(pr.argmax(1), y)
        if best is None or a > best[0]:
            best = (a, {"p": float(p)})
    return best[1]


def apply_equal(probs_list):
    return (sum(probs_list) / float(len(probs_list))).astype(np.float32)


def apply_conf(probs_list, temp):
    confs = [np.exp(pr.max(1, keepdims=True) / temp) for pr in probs_list]
    w = np.concatenate(confs, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list))).astype(np.float32)


def fit_conf(probs_list, y):
    best = None
    for t in TEMPS:
        pr = apply_conf(probs_list, t)
        a = acc(pr.argmax(1), y)
        if best is None or a > best[0]:
            best = (a, {"temp": float(t)})
    return best[1]


def fit_branch_temp(logits, y):
    best = None
    for t in BRANCH_TEMPS:
        a = acc(softmax_temp(logits, t).argmax(1), y)
        if best is None or a > best[0]:
            best = (a, float(t))
    return best[1]


def nested_branch_temp(logits, y, users):
    """Nested GroupKFold temperature per sample for one branch."""
    n = len(y)
    out = np.zeros((n, NUM_CLASSES), np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), y, users):
        t = fit_branch_temp(logits[tr], y[tr])
        out[va] = softmax_temp(logits[va], t)
    return out


def nested_family(keys, P, yt, us, family):
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), yt, us):
        if family == "eq":
            out[va] = apply_equal([P[k][va] for k in keys])
        elif family == "pow":
            cfg = fit_power_mean([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_power_mean([P[k][va] for k in keys], cfg["p"])
        elif family == "conf":
            cfg = fit_conf([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_conf([P[k][va] for k in keys], cfg["temp"])
        else:
            raise ValueError(family)
    return out


def folds_complete(fold_dir, n=5):
    return all((fold_dir / f"best_fold{i}.pt").exists() for i in range(n)) and (
        fold_dir / "best_holdout.pt"
    ).exists()


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


def weighted_sum(parts, weights):
    s = float(sum(weights))
    return sum(w * p for w, p in zip(weights, parts)) / s


def is_clear_win(oof_a, hold_a):
    hold_ok = (hold_a >= WIN_HOLD_STRICT) or (hold_a >= WIN_HOLD_CLEAR)
    return bool(hold_ok and oof_a >= WIN_OOF_FLOOR)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2m = load_v2_model_mod()
    cache = V7 / "cache"
    branches = {
        "midfuse": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_midfuse_v2b",
            "holdout_ckpt": V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt",
            "v2m": v2m,
            "default": "mid_fusion_v2",
        },
        "midfuse2": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_midfuse_s123",
            "holdout_ckpt": V2 / "checkpoints_midfuse_s123" / "best_holdout.pt",
            "v2m": v2m,
            "default": "mid_fusion_v2",
        },
        "midfuse3": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_midfuse_s7",
            "holdout_ckpt": V2 / "checkpoints_midfuse_s7" / "best_holdout.pt",
            "v2m": v2m,
            "default": "mid_fusion_v2",
        },
        "stgcn": {
            "kind": "stgcn",
            "fold_dir": V7 / "checkpoints_cv",
            "holdout_ckpt": V7 / "checkpoints_v2" / "best_holdout.pt",
            "v2m": v2m,
            "default": "stgcn_fuse",
        },
        "gru": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_gru",
            "holdout_ckpt": V2 / "checkpoints_gru" / "best_holdout.pt",
            "v2m": v2m,
            "default": "gru_imu",
        },
        "gru2": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_gru_s7",
            "holdout_ckpt": V2 / "checkpoints_gru_s7" / "best_holdout.pt",
            "v2m": v2m,
            "default": "gru_imu",
        },
        "deepconv": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_deepconv",
            "holdout_ckpt": V2 / "checkpoints_deepconv" / "best_holdout.pt",
            "v2m": v2m,
            "default": "deepconv_imu",
        },
        "cfuse": {
            "kind": "v2",
            "fold_dir": V2 / "checkpoints_compact_fuse",
            "holdout_ckpt": V2 / "checkpoints_compact_fuse" / "best_holdout.pt",
            "v2m": v2m,
            "default": "compact_fuse",
        },
    }

    z = np.load(V10 / "oof_logits_v10.npz", allow_pickle=True)
    y = z["y"]
    users = z["users"]
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]
    yt_nh = y[nh_idx]
    users_nh = users[nh_idx]
    yt_h = y[ho_idx]
    print(f"n={len(y)} nh={len(nh_idx)} ho={len(ho_idx)} device={device}", flush=True)

    raw_branch_keys = list(branches.keys())
    L = {k: z[k][nh_idx] for k in raw_branch_keys}
    mf_keys = ["midfuse", "midfuse2", "midfuse3"]
    gru_keys = ["gru", "gru2"]
    L["mf_avg"] = sum(L[k] for k in mf_keys) / 3.0
    L["mf12"] = 0.5 * (L["midfuse"] + L["midfuse2"])
    L["gru_avg"] = 0.5 * (L["gru"] + L["gru2"])

    # Holdout logits cache (prefer v12 copy, else v11)
    ho_path = ROOT / "holdout_logits.npz"
    if not ho_path.exists() and (V11 / "holdout_logits.npz").exists():
        import shutil

        shutil.copy2(V11 / "holdout_logits.npz", ho_path)
    X_skel, _, _, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    if ho_path.exists():
        hz = np.load(ho_path)
        H = {k: hz[k] for k in raw_branch_keys}
        print("loaded holdout cache", flush=True)
    else:
        H = {}
        for name, meta in branches.items():
            print("holdout", name, flush=True)
            m, _ = load_ckpt(meta["kind"], meta["holdout_ckpt"], device, meta["v2m"], meta["default"])
            H[name] = predict_logits_arr(
                m, X_skel[ho_idx], X_imu[ho_idx], has_imu[ho_idx].astype(np.float32), device
            )
            del m
            torch.cuda.empty_cache()
        np.savez_compressed(ho_path, **H)

    HL = dict(H)
    HL["mf_avg"] = sum(H[k] for k in mf_keys) / 3.0
    HL["mf12"] = 0.5 * (H["midfuse"] + H["midfuse2"])
    HL["gru_avg"] = 0.5 * (H["gru"] + H["gru2"])

    # ---- Idea 3: OOF-only temperature scaling per branch ----
    derived_keys = ["mf_avg", "mf12", "gru_avg"]
    all_logit_keys = raw_branch_keys + derived_keys

    P_raw = {k: softmax(L[k]) for k in all_logit_keys}
    HP_raw = {k: softmax(HL[k]) for k in all_logit_keys}

    P_ts = {}
    branch_temp_nested_note = {}
    for k in all_logit_keys:
        P_ts[k] = nested_branch_temp(L[k], yt_nh, users_nh)
        t_final = fit_branch_temp(L[k], yt_nh)
        branch_temp_nested_note[k] = {
            "final_T_fit_on_full_nh": t_final,
            "raw_acc": acc(P_raw[k].argmax(1), yt_nh),
            "ts_nested_acc": acc(P_ts[k].argmax(1), yt_nh),
        }
        print(
            f"temp {k}: raw={branch_temp_nested_note[k]['raw_acc']:.4f} "
            f"ts_nested={branch_temp_nested_note[k]['ts_nested_acc']:.4f} T*={t_final}",
            flush=True,
        )

    HP_ts = {k: softmax_temp(HL[k], branch_temp_nested_note[k]["final_T_fit_on_full_nh"]) for k in all_logit_keys}

    # Evaluate both raw and temp-scaled probability maps as blend substrates
    substrates = {
        "raw": (P_raw, HP_raw),
        "ts": (P_ts, HP_ts),
    }

    # Base fuse recipes (v11 + deepconv-diverse)
    bases_spec = {
        "A": (["mf_avg", "stgcn", "gru", "cfuse"], "pow"),
        "B": (["mf_avg", "stgcn", "gru_avg"], "eq"),
        "G": (["mf_avg", "stgcn", "gru_avg"], "pow"),
        "E": (["mf_avg", "stgcn", "gru_avg", "cfuse"], "pow"),
        "C": (["mf_avg", "stgcn", "gru", "cfuse"], "eq"),
        "D": (["mf_avg", "stgcn", "gru", "cfuse"], "conf"),
        "F": (["mf12", "stgcn", "gru_avg", "cfuse"], "conf"),
        "I": (["mf_avg", "stgcn", "gru2", "cfuse"], "pow"),
        "M": (["midfuse", "midfuse2", "midfuse3", "stgcn", "gru", "cfuse"], "pow"),
        # Idea 2: diverse deepconv-complete branch recipes (already OOF-complete)
        "H": (["mf_avg", "stgcn", "gru_avg", "deepconv"], "pow"),
        "J": (["mf_avg", "stgcn", "deepconv", "cfuse"], "eq"),
        "K": (["mf_avg", "stgcn", "gru", "deepconv", "cfuse"], "pow"),
        "N": (["mf_avg", "stgcn", "gru_avg", "deepconv", "cfuse"], "pow"),
        "O": (["mf_avg", "stgcn", "deepconv"], "pow"),
        "P": (["mf_avg", "stgcn", "gru_avg", "deepconv"], "eq"),
        "pow_all5": (["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "pow"),
        "eq_mf12": (["mf12", "stgcn", "gru"], "eq"),
        "pow_mf12": (["midfuse", "midfuse2", "stgcn", "gru"], "pow"),
        "eq_mfavg_gruavg_cf": (["mf_avg", "stgcn", "gru_avg", "cfuse"], "eq"),
    }

    methods = {}
    nested_probs = {}
    hold_probs = {}
    final_cfg = {}  # per substrate prefix

    def register(name, family, oof_p, hold_p, params=None):
        nested_probs[name] = oof_p
        hold_probs[name] = hold_p
        methods[name] = {
            "oof_acc_nested": acc(oof_p.argmax(1), yt_nh),
            "holdout_acc": acc(hold_p.argmax(1), yt_h),
            "family": family,
            "params": params or {},
        }

    # Build bases for each substrate
    nested_base = {}  # name -> probs (with substrate prefix in name for ts)
    hold_base = {}
    cfg_store = {}

    for sub_name, (Pmap, HPmap) in substrates.items():
        prefix = "" if sub_name == "raw" else "ts_"
        for name, (keys, fam) in bases_spec.items():
            full = prefix + name
            nested_base[full] = nested_family(keys, Pmap, yt_nh, users_nh, fam)
            if fam == "eq":
                cfg = {}
            elif fam == "pow":
                cfg = fit_power_mean([Pmap[k] for k in keys], yt_nh)
            else:
                cfg = fit_conf([Pmap[k] for k in keys], yt_nh)
            cfg_store[full] = {"keys": keys, "family": fam, "params": cfg, "substrate": sub_name}
            # holdout uses final params fit on full nh
            if fam == "eq":
                hold_base[full] = apply_equal([HPmap[k] for k in keys])
            elif fam == "pow":
                hold_base[full] = apply_power_mean([HPmap[k] for k in keys], cfg["p"])
            else:
                hold_base[full] = apply_conf([HPmap[k] for k in keys], cfg["temp"])
            register(full, fam, nested_base[full], hold_base[full], cfg_store[full])
            print(
                f"base {full} nested={methods[full]['oof_acc_nested']:.4f}",
                flush=True,
            )

        # v9 / v10 composites
        for label, members, wmode in [
            ("v9", ["pow_all5", "eq_mf12", "pow_mf12"], "equal"),
            ("v10", ["A", "B"], "equal"),
            ("v11ref", ["A", "B", "G"], "w423"),  # reference recipe weights
        ]:
            full_members = [prefix + m for m in members]
            full = prefix + label
            if wmode == "equal":
                nested_base[full] = sum(nested_base[m] for m in full_members) / len(full_members)
                hold_base[full] = sum(hold_base[m] for m in full_members) / len(full_members)
                params = {"members": members, "w": "equal", "substrate": sub_name}
            else:
                ws = [4, 2, 3]
                nested_base[full] = weighted_sum([nested_base[m] for m in full_members], ws)
                hold_base[full] = weighted_sum([hold_base[m] for m in full_members], ws)
                params = {"members": members, "w": ws, "substrate": sub_name}
            cfg_store[full] = {"family": "method_blend", "params": params}
            register(full, "method_blend", nested_base[full], hold_base[full], params)

    # ---- Idea 1: expanded integer-weight search around A/B/G (+ nearby both-hit recipes) ----
    # Raw and ts substrates; ABG grids expanded; also ABG+H/E/I/N light mixes
    abg_centers = [
        (4, 2, 3),
        (7, 4, 5),
        (3, 2, 2),
        (3, 1, 3),
        (6, 3, 5),
        (6, 4, 4),
        (7, 3, 5),
        (4, 3, 2),
        (5, 2, 3),
        (5, 3, 3),
        (4, 2, 4),
        (4, 1, 3),
        (8, 4, 6),
        (2, 1, 2),
    ]

    for sub_name in ("raw", "ts"):
        prefix = "" if sub_name == "raw" else "ts_"
        A, B, G = prefix + "A", prefix + "B", prefix + "G"

        # Expanded rectangular grid (wider than v11)
        for wa in range(1, 11):
            for wb in range(0, 8):
                for wg in range(0, 8):
                    if wa + wb + wg == 0:
                        continue
                    # skip pure singles already as bases when two weights 0
                    name = f"{prefix}wABG_{wa}_{wb}_{wg}"
                    if name in methods:
                        continue
                    o = weighted_sum([nested_base[A], nested_base[B], nested_base[G]], [wa, wb, wg])
                    h = weighted_sum([hold_base[A], hold_base[B], hold_base[G]], [wa, wb, wg])
                    register(
                        name,
                        "abg_weight",
                        o,
                        h,
                        {"wa": wa, "wb": wb, "wg": wg, "members": ["A", "B", "G"], "substrate": sub_name},
                    )

        # Neighborhood fine grid around centers (already covered by rect grid mostly)
        # 4-way: ABG + diverse (H/E/I/N/M/v9) with small integer weights
        extras = ["H", "E", "I", "N", "M", "J", "K", "P", "v9"]
        for wa, wb, wg in abg_centers:
            base_o = weighted_sum([nested_base[A], nested_base[B], nested_base[G]], [wa, wb, wg])
            base_h = weighted_sum([hold_base[A], hold_base[B], hold_base[G]], [wa, wb, wg])
            for ex in extras:
                ex_full = prefix + ex
                if ex_full not in nested_base:
                    continue
                for we in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
                    name = f"{prefix}wABG{wa}{wb}{wg}_{ex}x{we}"
                    denom = wa + wb + wg + we
                    o = ((wa + wb + wg) * base_o + we * nested_base[ex_full]) / denom
                    h = ((wa + wb + wg) * base_h + we * hold_base[ex_full]) / denom
                    register(
                        name,
                        "abg_plus",
                        o,
                        h,
                        {
                            "wa": wa,
                            "wb": wb,
                            "wg": wg,
                            "extra": ex,
                            "we": we,
                            "substrate": sub_name,
                        },
                    )

        # Integer 4-way ABGH / ABGE around both-hit region
        for wa in range(2, 8):
            for wb in range(1, 5):
                for wg in range(1, 5):
                    for wh in range(0, 4):
                        if wh == 0:
                            continue
                        for ex in ["H", "E", "N"]:
                            name = f"{prefix}wABG{ex}_{wa}_{wb}_{wg}_{wh}"
                            parts_n = [nested_base[A], nested_base[B], nested_base[G], nested_base[prefix + ex]]
                            parts_h = [hold_base[A], hold_base[B], hold_base[G], hold_base[prefix + ex]]
                            ws = [wa, wb, wg, wh]
                            register(
                                name,
                                "abgX_weight",
                                weighted_sum(parts_n, ws),
                                weighted_sum(parts_h, ws),
                                {
                                    "wa": wa,
                                    "wb": wb,
                                    "wg": wg,
                                    "wx": wh,
                                    "extra": ex,
                                    "substrate": sub_name,
                                },
                            )

        # Equal blends of notable sets
        for comb in [
            ("A", "B", "G"),
            ("A", "B", "G", "H"),
            ("A", "B", "H"),
            ("v11ref", "H"),
            ("v11ref", "N"),
            ("v10", "H"),
            ("A", "B", "G", "N"),
            ("H", "B", "G"),
            ("K", "B", "G"),
            ("v11ref", "v9"),
            ("v11ref", "H", "v9"),
        ]:
            fulls = [prefix + c for c in comb]
            if any(f not in nested_base for f in fulls):
                continue
            name = prefix + "eq_" + "+".join(comb)
            o = sum(nested_base[f] for f in fulls) / len(fulls)
            h = sum(hold_base[f] for f in fulls) / len(fulls)
            register(name, "method_blend", o, h, {"members": list(comb), "w": "equal", "substrate": sub_name})

    print(f"registered methods: {len(methods)}", flush=True)

    # SELECT by nested OOF only
    best_name = max(methods, key=lambda k: methods[k]["oof_acc_nested"])
    oof_best = methods[best_name]["oof_acc_nested"]
    hold_best = methods[best_name]["holdout_acc"]
    clear_win = is_clear_win(oof_best, hold_best)
    prefer_hit = bool(hold_best > PREF_HOLD and oof_best >= PREF_OOF)
    peek_best = max(methods, key=lambda k: methods[k]["holdout_acc"])

    # Also find best under prefer / clear constraints (report only; selection stays nested-max)
    both_pref = {
        k: v
        for k, v in methods.items()
        if v["oof_acc_nested"] >= PREF_OOF and v["holdout_acc"] > PREF_HOLD
    }
    clear_hits = {
        k: v
        for k, v in methods.items()
        if is_clear_win(v["oof_acc_nested"], v["holdout_acc"])
    }

    print(
        f"SELECTED={best_name} nested={oof_best:.4f} hold={hold_best:.4f} "
        f"clear_win={clear_win} prefer_hit={prefer_hit}",
        flush=True,
    )
    print(
        f"peek best hold={methods[peek_best]['holdout_acc']:.4f} {peek_best} "
        f"oof={methods[peek_best]['oof_acc_nested']:.4f} (NOT used)",
        flush=True,
    )
    print(f"prefer_hits={len(both_pref)} clear_hits={len(clear_hits)}", flush=True)

    def apply_base_on(name, probs_map, substrate):
        """Apply a base fuse (without prefix) using cfg_store entry."""
        prefix = "" if substrate == "raw" else "ts_"
        full = name if name.startswith("ts_") or name in cfg_store else prefix + name
        # normalize
        if full not in cfg_store and prefix + name in cfg_store:
            full = prefix + name
        cfg = cfg_store[full]
        if cfg.get("family") == "method_blend" or ("keys" not in cfg):
            # method blend of bases
            params = cfg["params"] if "params" in cfg else cfg
            members = params["members"]
            w = params.get("w", "equal")
            mixes = [apply_base_on(m, probs_map, substrate) for m in members]
            if w == "equal":
                return sum(mixes) / float(len(mixes))
            return weighted_sum(mixes, w)
        keys = cfg["keys"]
        fam = cfg["family"]
        params = cfg["params"]
        if fam == "eq":
            return apply_equal([probs_map[k] for k in keys])
        if fam == "pow":
            return apply_power_mean([probs_map[k] for k in keys], params["p"])
        if fam == "conf":
            return apply_conf([probs_map[k] for k in keys], params["temp"])
        raise ValueError((full, fam))

    def apply_named(name, probs_map_raw, probs_map_ts):
        m = methods[name]
        fam = m["family"]
        params = m["params"]
        substrate = params.get("substrate", "raw" if not name.startswith("ts_") else "ts")
        Pmap = probs_map_raw if substrate == "raw" else probs_map_ts

        if fam in ("pow", "eq", "conf"):
            return apply_base_on(name, Pmap, substrate)

        if fam == "method_blend":
            members = params["members"]
            mixes = [apply_base_on(mem, Pmap, substrate) for mem in members]
            w = params.get("w", "equal")
            if w == "equal":
                return sum(mixes) / float(len(mixes))
            return weighted_sum(mixes, w)

        if fam == "abg_weight":
            wa, wb, wg = params["wa"], params["wb"], params["wg"]
            return weighted_sum(
                [
                    apply_base_on("A", Pmap, substrate),
                    apply_base_on("B", Pmap, substrate),
                    apply_base_on("G", Pmap, substrate),
                ],
                [wa, wb, wg],
            )

        if fam == "abg_plus":
            wa, wb, wg = params["wa"], params["wb"], params["wg"]
            we = params["we"]
            extra = params["extra"]
            base = weighted_sum(
                [
                    apply_base_on("A", Pmap, substrate),
                    apply_base_on("B", Pmap, substrate),
                    apply_base_on("G", Pmap, substrate),
                ],
                [wa, wb, wg],
            )
            return ((wa + wb + wg) * base + we * apply_base_on(extra, Pmap, substrate)) / float(
                wa + wb + wg + we
            )

        if fam == "abgX_weight":
            wa, wb, wg, wx = params["wa"], params["wb"], params["wg"], params["wx"]
            extra = params["extra"]
            return weighted_sum(
                [
                    apply_base_on("A", Pmap, substrate),
                    apply_base_on("B", Pmap, substrate),
                    apply_base_on("G", Pmap, substrate),
                    apply_base_on(extra, Pmap, substrate),
                ],
                [wa, wb, wg, wx],
            )

        raise ValueError((name, fam))

    # Resolve branches needed for selected method
    sel_params = methods[best_name]["params"]
    substrate = sel_params.get("substrate", "raw" if not best_name.startswith("ts_") else "ts")
    need = set()

    def add_keys_for_member(mem):
        prefix = "" if substrate == "raw" else "ts_"
        full = prefix + mem if (prefix + mem) in cfg_store else mem
        if full in cfg_store and "keys" in cfg_store[full]:
            need.update(cfg_store[full]["keys"])
        elif mem == "v9":
            need.update(["midfuse", "midfuse2", "stgcn", "gru", "deepconv", "mf12"])
        elif mem in ("v10", "v11ref"):
            need.update(["mf_avg", "stgcn", "gru", "gru_avg", "cfuse"])
        else:
            need.update(["mf_avg", "stgcn", "gru", "gru_avg", "cfuse"])

    fam = methods[best_name]["family"]
    if fam in ("abg_weight", "abg_plus", "abgX_weight"):
        need |= {"mf_avg", "stgcn", "gru", "gru_avg", "cfuse"}
        if fam in ("abg_plus", "abgX_weight"):
            add_keys_for_member(sel_params["extra"])
    elif fam == "method_blend":
        for mem in sel_params.get("members", []):
            add_keys_for_member(mem)
    elif best_name in cfg_store and "keys" in cfg_store[best_name]:
        need |= set(cfg_store[best_name]["keys"])
    else:
        # strip ts_
        bare = best_name[3:] if best_name.startswith("ts_") else best_name
        key = ("" if substrate == "raw" else "ts_") + bare
        if key in cfg_store and "keys" in cfg_store[key]:
            need |= set(cfg_store[key]["keys"])
        else:
            need |= {"mf_avg", "stgcn", "gru", "gru_avg", "cfuse"}

    resolved = set()
    for k in need:
        if k == "mf_avg":
            resolved.update(mf_keys)
        elif k == "mf12":
            resolved.update(["midfuse", "midfuse2"])
        elif k == "gru_avg":
            resolved.update(gru_keys)
        else:
            resolved.add(k)
    need_names = sorted(resolved & set(branches.keys()))
    print(f"test branches: {need_names}", flush=True)

    for nm in need_names:
        assert folds_complete(branches[nm]["fold_dir"]), nm

    # Always build candidate submission for selected nested-OOF method
    test_probs_raw, test_logits, paths = test_branch_logits(branches, need_names, cache, device)
    if set(mf_keys).issubset(test_logits):
        test_logits["mf_avg"] = sum(test_logits[k] for k in mf_keys) / float(len(mf_keys))
        test_probs_raw["mf_avg"] = softmax(test_logits["mf_avg"])
    if "midfuse" in test_logits and "midfuse2" in test_logits:
        test_logits["mf12"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse2"])
        test_probs_raw["mf12"] = softmax(test_logits["mf12"])
    if set(gru_keys).issubset(test_logits):
        test_logits["gru_avg"] = sum(test_logits[k] for k in gru_keys) / float(len(gru_keys))
        test_probs_raw["gru_avg"] = softmax(test_logits["gru_avg"])

    # For missing keys that might be needed, fill from available
    for k in all_logit_keys:
        if k not in test_logits and k in raw_branch_keys and k in test_logits:
            pass
        if k in raw_branch_keys and k not in test_probs_raw and k in test_logits:
            test_probs_raw[k] = softmax(test_logits[k])

    # Temperature-scaled test probs using T* fit on full nh
    test_probs_ts = {}
    for k in list(test_logits.keys()):
        if k in branch_temp_nested_note:
            test_probs_ts[k] = softmax_temp(test_logits[k], branch_temp_nested_note[k]["final_T_fit_on_full_nh"])
        else:
            test_probs_ts[k] = test_probs_raw[k]
    # also for derived already in test_probs_raw
    for k in test_probs_raw:
        if k not in test_probs_ts:
            if k in branch_temp_nested_note and k in test_logits:
                test_probs_ts[k] = softmax_temp(test_logits[k], branch_temp_nested_note[k]["final_T_fit_on_full_nh"])
            else:
                test_probs_ts[k] = test_probs_raw[k]

    mix_te = apply_named(best_name, test_probs_raw, test_probs_ts)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})

    cand_path = ROOT / "submission_v12_candidate.csv"
    df.to_csv(cand_path, index=False)
    np.savez_compressed(
        ROOT / "submission_v12_probs.npz",
        probs=mix_te,
        paths=np.array(paths, dtype=object),
        method=np.array(best_name),
    )
    df.to_csv(ROOT / "submission_v12.csv", index=False)

    overwritten = False
    ping = False
    if clear_win:
        df.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submissions").mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v12.csv", index=False)
        overwritten = True
        ping = True
        (TRACK / "submission_README.txt").write_text(
            (
                f"v12 clear-win: nested OOF {oof_best:.4f} (>=0.558) and holdout {hold_best:.4f} "
                f"(>=0.576 or >=0.5713)\n"
                f"method={best_name} params={json.dumps(methods[best_name]['params'])}\n"
                f"No TTA. No sklearn stacker. ping_disk_saver=true.\n"
                f"Prior track was v11 wABG_4_2_3 OOF 0.5614 / hold 0.5663.\n"
            ),
            encoding="utf-8",
        )
    else:
        print("NO clear win; left track submission.csv as v11", flush=True)

    methods_out = {
        k: {
            "oof_acc_nested": v["oof_acc_nested"],
            "holdout_acc": v["holdout_acc"],
            "family": v["family"],
            "params": v["params"],
        }
        for k, v in methods.items()
    }

    top = sorted(methods_out.items(), key=lambda kv: kv[1]["oof_acc_nested"], reverse=True)[:50]
    both_vs_v11 = {
        k: v
        for k, v in methods_out.items()
        if v["oof_acc_nested"] >= V11_OOF - 1e-12 and v["holdout_acc"] > V11_HOLD + 1e-12
    }
    # gate hits at v11 bar (not clear-win bar)
    v11_bar_hits = {
        k: v
        for k, v in methods_out.items()
        if v["oof_acc_nested"] >= 0.558 and v["holdout_acc"] >= V11_HOLD - 1e-9
    }

    # Honest table rows
    table_rows = [
        ("v9_blend_ABD", methods_out.get("v9", methods_out.get("ts_v9"))),
        ("v10_selected", methods_out.get("v10", methods_out.get("ts_v10"))),
        ("v11_wABG_4_2_3", methods_out.get("wABG_4_2_3")),
        ("v11ref_raw", methods_out.get("v11ref")),
        (f"v12_selected ({best_name})", methods_out[best_name]),
        (
            f"peek_best_hold ({peek_best})",
            methods_out[peek_best],
        ),
    ]

    metrics = {
        "track": "v12_expand_temp_deepconv",
        "v11_baseline_oof": V11_OOF,
        "v11_baseline_holdout": V11_HOLD,
        "v10_baseline_oof": V10_OOF,
        "v10_baseline_holdout": V10_HOLD,
        "v9_baseline_oof": V9_OOF,
        "v9_baseline_holdout": V9_HOLD,
        "selected_method": best_name,
        "selected_oof_acc_nested": oof_best,
        "selected_holdout_acc": hold_best,
        "delta_oof_vs_v11": oof_best - V11_OOF,
        "delta_holdout_vs_v11": hold_best - V11_HOLD,
        "clear_win": clear_win,
        "prefer_hit": prefer_hit,
        "overwrite_submission": overwritten,
        "ping_disk_saver": ping,
        "win_criteria": (
            f"holdout>={WIN_HOLD_STRICT} OR holdout>={WIN_HOLD_CLEAR:.4f} "
            f"AND nested_OOF>={WIN_OOF_FLOOR}; prefer hold>{PREF_HOLD:.4f} & OOF>={PREF_OOF}"
        ),
        "holdout_peek_not_used_for_selection": {
            "best_holdout_method_peek": peek_best,
            "best_holdout_acc_peek": methods[peek_best]["holdout_acc"],
            "best_holdout_oof_peek": methods[peek_best]["oof_acc_nested"],
            "note": "NOT used for selection or overwrite decision",
        },
        "branch_temperature_scaling": branch_temp_nested_note,
        "n_methods": len(methods_out),
        "prefer_hits_count": len(both_pref),
        "clear_hits_count": len(clear_hits),
        "v11_bar_hits_count": len(v11_bar_hits),
        "prefer_hits_top": dict(
            list(sorted(both_pref.items(), key=lambda kv: -kv[1]["oof_acc_nested"]))[:15]
        ),
        "clear_hits_top": dict(
            list(sorted(clear_hits.items(), key=lambda kv: -kv[1]["oof_acc_nested"]))[:15]
        ),
        "top_by_nested_oof": [{"name": k, **v} for k, v in top],
        "honest_table": [
            {
                "label": lab,
                "oof": (None if row is None else row["oof_acc_nested"]),
                "holdout": (None if row is None else row["holdout_acc"]),
            }
            for lab, row in table_rows
        ],
        "selected_params": methods[best_name]["params"],
        "protocol": (
            "GroupKFold nested OOF; per-branch OOF-only temp scale; expanded ABG + deepconv-diverse "
            "H/J/K/N/P; holdout last; clear-win gate for overwrite"
        ),
        "ideas": [
            "expanded integer ABG weights + ABG+extra + ABGH/E/N 4-way",
            "deepconv-diverse OOF-complete recipes H/J/K/N/O/P (no new training)",
            "OOF-only temperature scaling per branch before blend (raw vs ts substrates)",
        ],
        "elapsed_sec": time.time() - t0,
        "finished_at_unix": time.time(),
        "methods": methods_out,
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def fmt(x):
        return "n/a" if x is None else f"{x:.4f}"

    # README is written by a safe helper to avoid f-string dict-literal issues
    def _f4(x):
        return "n/a" if x is None else f"{x:.4f}"

    v9o = methods_out.get("v9", {})
    v10o = methods_out.get("v10", {})
    v11o = methods_out.get("wABG_4_2_3", {})
    readme = "\n".join([
        "# Small-Model-Track v12 — expand ABG + temp-scale + deepconv-diverse",
        "",
        "## Summary vs v11",
        "",
        f"| Nested OOF | v11 0.5614 | v12 selected ({best_name}) {oof_best:.4f} |",
        f"| Holdout | v11 0.5663 | v12 {hold_best:.4f} |",
        "",
        f"clear_win={clear_win}. overwrite_submission={overwritten}. ping_disk_saver={ping}.",
        f"prefer_hits={len(both_pref)}. clear_hits={len(clear_hits)}. n_methods={len(methods_out)}.",
        "",
        f"Selected: `{best_name}` params=`{json.dumps(methods[best_name]['params'])}`",
        "",
        "See metrics.json honest_table. Track submission.csv left as v11 unless clear_win.",
        "",
        "## Reproduce",
        "",
        "```text",
        "baselines\\v7_stgcn\\.venv\\Scripts\\python.exe baselines\\v12\\v12_expand_blend.py",
        "```",
        "",
    ])
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "selected_method": best_name,
        "selected_oof_acc_nested": oof_best,
        "selected_holdout_acc": hold_best,
        "clear_win": clear_win,
        "prefer_hit": prefer_hit,
        "overwrite_submission": overwritten,
        "ping_disk_saver": ping,
        "delta_oof_vs_v11": oof_best - V11_OOF,
        "delta_holdout_vs_v11": hold_best - V11_HOLD,
        "prefer_hits_count": len(both_pref),
        "clear_hits_count": len(clear_hits),
        "n_methods": len(methods_out),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
