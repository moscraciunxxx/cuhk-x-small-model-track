"""v11: ABG reweight + v9/v10 blends under nested GroupKFold discipline.

Clear-win gate (AND): nested OOF >= 0.5585 AND holdout >= 0.5663.
Selection uses nested OOF only; holdout evaluated last.
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
V9_OOF = 0.5486397361912614
V9_HOLD = 0.5663366336633663
V10_OOF = 0.5585325638911789
V10_HOLD = 0.5623762376237624
WIN_OOF = 0.5585
WIN_HOLD = 0.5663


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v11", V2 / "model.py")
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

    L = {k: z[k][nh_idx] for k in branches}
    P = {k: softmax(L[k]) for k in branches}
    mf_keys = ["midfuse", "midfuse2", "midfuse3"]
    gru_keys = ["gru", "gru2"]
    L["mf_avg"] = sum(L[k] for k in mf_keys) / 3.0
    P["mf_avg"] = softmax(L["mf_avg"])
    L["mf12"] = 0.5 * (L["midfuse"] + L["midfuse2"])
    P["mf12"] = softmax(L["mf12"])
    L["gru_avg"] = 0.5 * (L["gru"] + L["gru2"])
    P["gru_avg"] = softmax(L["gru_avg"])

    # Holdout logits (cache)
    ho_path = ROOT / "holdout_logits.npz"
    X_skel, _, _, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    if ho_path.exists():
        hz = np.load(ho_path)
        H = {k: hz[k] for k in branches}
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
    HP = {k: softmax(HL[k]) for k in HL}

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
        "pow_all5": (["midfuse", "midfuse2", "stgcn", "gru", "deepconv"], "pow"),
        "eq_mf12": (["mf12", "stgcn", "gru"], "eq"),
        "pow_mf12": (["midfuse", "midfuse2", "stgcn", "gru"], "pow"),
        "eq_mfavg_gruavg_cf": (["mf_avg", "stgcn", "gru_avg", "cfuse"], "eq"),
    }

    nested_base = {}
    final_cfg = {}
    for name, (keys, fam) in bases_spec.items():
        nested_base[name] = nested_family(keys, P, yt_nh, users_nh, fam)
        if fam == "eq":
            cfg = {}
        elif fam == "pow":
            cfg = fit_power_mean([P[k] for k in keys], yt_nh)
        else:
            cfg = fit_conf([P[k] for k in keys], yt_nh)
        final_cfg[name] = {"keys": keys, "family": fam, "params": cfg}
        print(
            f"base {name} nested={acc(nested_base[name].argmax(1), yt_nh):.4f} cfg={cfg}",
            flush=True,
        )

    nested_base["v9"] = (nested_base["pow_all5"] + nested_base["eq_mf12"] + nested_base["pow_mf12"]) / 3.0
    nested_base["v10"] = 0.5 * (nested_base["A"] + nested_base["B"])
    final_cfg["v9"] = {
        "family": "method_blend",
        "params": {"members": ["pow_all5", "eq_mf12", "pow_mf12"], "w": "equal"},
    }
    final_cfg["v10"] = {
        "family": "method_blend",
        "params": {"members": ["A", "B"], "w": "equal"},
    }

    def apply_base(name, probs_map):
        cfg = final_cfg[name]
        if cfg["family"] == "method_blend":
            members = cfg["params"]["members"]
            mixes = [apply_base(m, probs_map) for m in members]
            w = cfg["params"].get("w", "equal")
            if w == "equal":
                return sum(mixes) / float(len(mixes))
            s = float(sum(w))
            return sum(w[i] * mixes[i] for i in range(len(mixes))) / s
        keys = cfg["keys"]
        fam = cfg["family"]
        params = cfg["params"]
        if fam == "eq":
            return apply_equal([probs_map[k] for k in keys])
        if fam == "pow":
            return apply_power_mean([probs_map[k] for k in keys], params["p"])
        if fam == "conf":
            return apply_conf([probs_map[k] for k in keys], params["temp"])
        raise ValueError(fam)

    hold_base = {n: apply_base(n, HP) for n in list(bases_spec) + ["v9", "v10"]}

    methods = {}
    nested_probs = {}
    hold_probs = {}

    def register(name, family, oof_p, hold_p, params=None):
        nested_probs[name] = oof_p
        hold_probs[name] = hold_p
        methods[name] = {
            "oof_acc_nested": acc(oof_p.argmax(1), yt_nh),
            "holdout_acc": acc(hold_p.argmax(1), yt_h),
            "family": family,
            "params": params or {},
        }
        print(
            f"{name} nested={methods[name]['oof_acc_nested']:.4f} "
            f"(hold deferred in print until end table)",
            flush=True,
        )

    # Base recipes
    for name in bases_spec:
        register(name, bases_spec[name][1], nested_base[name], hold_base[name], final_cfg[name])
    register("v9_blend_ABD", "method_blend", nested_base["v9"], hold_base["v9"], final_cfg["v9"])
    register("v10_selected", "method_blend", nested_base["v10"], hold_base["v10"], final_cfg["v10"])

    # Integer ABG weight grid (predefined; select by nested OOF only)
    for wa in range(1, 8):
        for wb in range(0, 6):
            for wg in range(0, 6):
                if wa + wb + wg == 0:
                    continue
                # skip pure duplicates of A/B/G already registered as bases
                name = f"wABG_{wa}_{wb}_{wg}"
                o = (wa * nested_base["A"] + wb * nested_base["B"] + wg * nested_base["G"]) / (
                    wa + wb + wg
                )
                h = (wa * hold_base["A"] + wb * hold_base["B"] + wg * hold_base["G"]) / (wa + wb + wg)
                register(
                    name,
                    "abg_weight",
                    o,
                    h,
                    {"wa": wa, "wb": wb, "wg": wg, "members": ["A", "B", "G"]},
                )

    # v9/v10 weight grid
    for w10 in np.linspace(0.0, 1.0, 21):
        name = f"v9v10_w{w10:.2f}"
        o = (1.0 - w10) * nested_base["v9"] + w10 * nested_base["v10"]
        h = (1.0 - w10) * hold_base["v9"] + w10 * hold_base["v10"]
        register(name, "v9v10_blend", o, h, {"w10": float(w10)})

    # Agreement / margin (fixed thr grid; pick by nested OOF)
    m10 = nested_base["v10"].max(1) - np.partition(nested_base["v10"], -2, axis=1)[:, -2]
    mh = hold_base["v10"].max(1) - np.partition(hold_base["v10"], -2, axis=1)[:, -2]
    for thr in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for fb_name in ["v9", "C", "F"]:
            o = np.where((m10 >= thr)[:, None], nested_base["v10"], nested_base[fb_name])
            h = np.where((mh >= thr)[:, None], hold_base["v10"], hold_base[fb_name])
            register(
                f"margin_v10_else_{fb_name}_t{thr}",
                "margin_gate",
                o,
                h,
                {"thr": thr, "fallback": fb_name},
            )

    # Light mixes of top ABG with diversity members (predefined small set)
    for wa, wb, wg in [(4, 2, 3), (3, 2, 2), (7, 4, 5), (2, 1, 1), (4, 3, 2)]:
        base_o = (wa * nested_base["A"] + wb * nested_base["B"] + wg * nested_base["G"]) / (
            wa + wb + wg
        )
        base_h = (wa * hold_base["A"] + wb * hold_base["B"] + wg * hold_base["G"]) / (wa + wb + wg)
        for extra in ["I", "E", "M", "C", "F", "v9"]:
            for we in [0.25, 0.5, 0.75, 1.0]:
                name = f"wABG{wa}{wb}{wg}_{extra}x{we}"
                o = ((wa + wb + wg) * base_o + we * nested_base[extra]) / (wa + wb + wg + we)
                h = ((wa + wb + wg) * base_h + we * hold_base[extra]) / (wa + wb + wg + we)
                register(
                    name,
                    "abg_plus",
                    o,
                    h,
                    {"wa": wa, "wb": wb, "wg": wg, "extra": extra, "we": we},
                )

    # Equal blends of notable pairs/triples
    for comb in [
        ("A", "B"),
        ("A", "G"),
        ("A", "B", "G"),
        ("v10", "v9"),
        ("v10", "F"),
        ("v10", "C"),
        ("I", "B", "C"),
        ("A", "B", "F"),
        ("F", "v10", "C"),
        ("eq_mfavg_gruavg_cf", "B"),
        ("eq_mfavg_gruavg_cf", "v10"),
    ]:
        o = sum(nested_base[x] for x in comb) / len(comb)
        h = sum(hold_base[x] for x in comb) / len(comb)
        register("eq_" + "+".join(comb), "method_blend", o, h, {"members": list(comb), "w": "equal"})

    # SELECT by nested OOF only
    best_name = max(methods, key=lambda k: methods[k]["oof_acc_nested"])
    oof_best = methods[best_name]["oof_acc_nested"]
    hold_best = methods[best_name]["holdout_acc"]
    clear_win = bool(oof_best >= WIN_OOF and hold_best >= WIN_HOLD)
    peek_best = max(methods, key=lambda k: methods[k]["holdout_acc"])
    print(
        f"SELECTED={best_name} nested={oof_best:.4f} hold={hold_best:.4f} clear_and={clear_win}",
        flush=True,
    )
    print(
        f"peek best hold={methods[peek_best]['holdout_acc']:.4f} {peek_best} "
        f"oof={methods[peek_best]['oof_acc_nested']:.4f} (NOT used)",
        flush=True,
    )

    # Build selected hold/test applicator
    def apply_named(name, probs_map):
        m = methods[name]
        fam = m["family"]
        params = m["params"]
        if fam in ("pow", "eq", "conf"):
            return apply_base(name, probs_map)
        if fam == "method_blend":
            if name in ("v9_blend_ABD", "v10_selected") or name.startswith("eq_"):
                members = params["members"]
                mixes = []
                for mem in members:
                    if mem in bases_spec or mem in ("v9", "v10"):
                        mixes.append(apply_base(mem, probs_map) if mem in final_cfg else apply_named(mem, probs_map))
                    else:
                        mixes.append(apply_base(mem, probs_map))
                return sum(mixes) / float(len(mixes))
        if fam == "abg_weight":
            wa, wb, wg = params["wa"], params["wb"], params["wg"]
            return (
                wa * apply_base("A", probs_map)
                + wb * apply_base("B", probs_map)
                + wg * apply_base("G", probs_map)
            ) / float(wa + wb + wg)
        if fam == "v9v10_blend":
            w10 = params["w10"]
            return (1.0 - w10) * apply_base("v9", probs_map) + w10 * apply_base("v10", probs_map)
        if fam == "margin_gate":
            primary = apply_base("v10", probs_map)
            fb = apply_base(params["fallback"], probs_map)
            thr = params["thr"]
            margin = primary.max(1) - np.partition(primary, -2, axis=1)[:, -2]
            return np.where((margin >= thr)[:, None], primary, fb)
        if fam == "abg_plus":
            wa, wb, wg = params["wa"], params["wb"], params["wg"]
            we = params["we"]
            extra = params["extra"]
            base = (
                wa * apply_base("A", probs_map)
                + wb * apply_base("B", probs_map)
                + wg * apply_base("G", probs_map)
            ) / float(wa + wb + wg)
            return ((wa + wb + wg) * base + we * apply_base(extra, probs_map)) / float(
                wa + wb + wg + we
            )
        raise ValueError((name, fam))

    # Test predictions if clear win OR always write candidate
    need_branches = sorted(set(branches.keys()) & {"midfuse", "midfuse2", "midfuse3", "stgcn", "gru", "gru2", "cfuse", "deepconv"})
    # resolve for selected
    sel_params = methods[best_name]["params"]
    need = set()
    fam = methods[best_name]["family"]
    if fam == "abg_weight" or fam == "abg_plus":
        need |= {"mf_avg", "stgcn", "gru", "gru_avg", "cfuse"}
        if fam == "abg_plus":
            extra = sel_params["extra"]
            if extra in final_cfg and "keys" in final_cfg[extra]:
                need |= set(final_cfg[extra]["keys"])
            elif extra == "v9":
                need |= {"midfuse", "midfuse2", "stgcn", "gru", "deepconv", "mf12"}
    elif fam == "v9v10_blend":
        need |= {"mf_avg", "stgcn", "gru", "gru_avg", "cfuse", "midfuse", "midfuse2", "deepconv", "mf12"}
    elif best_name in final_cfg and "keys" in final_cfg[best_name]:
        need |= set(final_cfg[best_name]["keys"])
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

    test_probs, test_logits, paths = test_branch_logits(branches, need_names, cache, device)
    if set(mf_keys).issubset(test_logits):
        test_logits["mf_avg"] = sum(test_logits[k] for k in mf_keys) / float(len(mf_keys))
        test_probs["mf_avg"] = softmax(test_logits["mf_avg"])
    if "midfuse" in test_logits and "midfuse2" in test_logits:
        test_logits["mf12"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse2"])
        test_probs["mf12"] = softmax(test_logits["mf12"])
    if set(gru_keys).issubset(test_logits):
        test_logits["gru_avg"] = sum(test_logits[k] for k in gru_keys) / float(len(gru_keys))
        test_probs["gru_avg"] = softmax(test_logits["gru_avg"])

    mix_te = apply_named(best_name, test_probs)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})

    cand_path = ROOT / "submission_v11_candidate.csv"
    df.to_csv(cand_path, index=False)
    np.savez_compressed(
        ROOT / "submission_v11_probs.npz",
        probs=mix_te,
        paths=np.array(paths, dtype=object),
        method=np.array(best_name),
    )

    overwritten = False
    ping = False
    if clear_win:
        df.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submissions").mkdir(exist_ok=True)
        df.to_csv(TRACK / "submissions" / "submission_v11.csv", index=False)
        df.to_csv(ROOT / "submission_v11.csv", index=False)
        overwritten = True
        ping = True
        (TRACK / "submission_README.txt").write_text(
            (
                f"v11 clear-win AND: nested OOF {oof_best:.4f} (>=0.5585) and holdout {hold_best:.4f} (>=0.5663)\n"
                f"method={best_name} params={json.dumps(methods[best_name]['params'])}\n"
                f"A=pow(mf_avg3,ST-GCN,GRU,cfuse) B=eq(mf_avg3,ST-GCN,gru_avg2) G=pow(mf_avg3,ST-GCN,gru_avg2)\n"
                f"No TTA. No sklearn stacker. ping_disk_saver=true.\n"
            ),
            encoding="utf-8",
        )
    else:
        # do not overwrite submission.csv
        df.to_csv(ROOT / "submission_v11.csv", index=False)
        print("NO clear AND-win; left track submission.csv untouched", flush=True)

    methods_out = {
        k: {
            "oof_acc_nested": v["oof_acc_nested"],
            "holdout_acc": v["holdout_acc"],
            "family": v["family"],
            "params": v["params"],
        }
        for k, v in methods.items()
    }

    # compact comparison table vs v9/v10
    top = sorted(methods_out.items(), key=lambda kv: kv[1]["oof_acc_nested"], reverse=True)[:40]
    both_hits = {
        k: v
        for k, v in methods_out.items()
        if v["oof_acc_nested"] >= WIN_OOF and v["holdout_acc"] >= WIN_HOLD
    }

    metrics = {
        "track": "v11_abg_blend",
        "v9_baseline_oof": V9_OOF,
        "v9_baseline_holdout": V9_HOLD,
        "v10_baseline_oof": V10_OOF,
        "v10_baseline_holdout": V10_HOLD,
        "selected_method": best_name,
        "selected_oof_acc_nested": oof_best,
        "selected_holdout_acc": hold_best,
        "delta_oof_vs_v10": oof_best - V10_OOF,
        "delta_holdout_vs_v9": hold_best - V9_HOLD,
        "delta_holdout_vs_v10": hold_best - V10_HOLD,
        "clear_win_and": clear_win,
        "overwrite_submission": overwritten,
        "ping_disk_saver": ping,
        "win_criteria": f"nested_OOF>={WIN_OOF} AND holdout>={WIN_HOLD}",
        "holdout_peek_not_used_for_selection": {
            "best_holdout_method_peek": peek_best,
            "best_holdout_acc_peek": methods[peek_best]["holdout_acc"],
            "note": "NOT used for selection or overwrite decision",
        },
        "selected_recipe_human": {
            "name": best_name,
            "params": methods[best_name]["params"],
            "A": "nested power-mean(mf_avg3, ST-GCN, GRU, compact_fuse)",
            "B": "equal(mf_avg3, ST-GCN, gru_avg2)",
            "G": "nested power-mean(mf_avg3, ST-GCN, gru_avg2)",
        },
        "both_gate_hits_count": len(both_hits),
        "both_gate_hits_top": dict(list(sorted(both_hits.items(), key=lambda kv: -kv[1]["oof_acc_nested"]))[:15]),
        "top_by_nested_oof": [{ "name": k, **v} for k, v in top],
        "methods": methods_out,
        "protocol": "GroupKFold nested OOF; ABG weight grid + v9/v10 blends; holdout last; AND gate",
        "elapsed_sec": time.time() - t0,
        "finished_at_unix": time.time(),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # README
    readme = f"""# Small-Model-Track v11 — ABG reweight + v9/v10 blends

## Summary vs v9 / v10

| | v9 blend_ABD | v10 selected | **v11 selected** (`{best_name}`) |
|---|---|---|---|
| Nested OOF | 0.5486 | 0.5585 | **{oof_best:.4f}** |
| Holdout | 0.5663 | 0.5624 | **{hold_best:.4f}** |

Win criteria (AND): nested OOF ≥ **0.5585** AND holdout ≥ **0.5663**.  
clear_win_and={clear_win}. overwrite_submission={overwritten}. ping_disk_saver={ping}.

Selection used **nested GroupKFold on non-holdout OOF only**; holdout evaluated last.

## Selected method

`{best_name}` params=`{json.dumps(methods[best_name]['params'])}`

Weighted blend of three nested fuses:

1. **A** `pow_mfavg_st_gru_cf` — nested power-mean over (mf_avg3, ST-GCN, GRU, compact_fuse)
2. **B** `eq_mfavg_st_gruavg` — equal softmax of (mf_avg3, ST-GCN, gru_avg2)
3. **G** `pow_mfavg_st_gruavg` — nested power-mean over (mf_avg3, ST-GCN, gru_avg2)

v10 was equal(A,B). v11 searches integer weights (wa,wb,wg) on nested OOF.

## Protocol

1. Reuse `baselines/v10/oof_logits_v10.npz` branch OOF logits
2. Nested OOF for base fuses A/B/G (+ others); ABG weight grid; v9/v10 blends; margin gates
3. Select max nested OOF
4. Holdout last (AND gate for overwrite)
5. Test: 5-fold avg softmax → same recipe → submission

## Key ablations (nested → holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| v9_blend_ABD | {methods['v9_blend_ABD']['oof_acc_nested']:.4f} | {methods['v9_blend_ABD']['holdout_acc']:.4f} |
| v10_selected | {methods['v10_selected']['oof_acc_nested']:.4f} | {methods['v10_selected']['holdout_acc']:.4f} |
| **{best_name} (selected)** | **{oof_best:.4f}** | **{hold_best:.4f}** |
| peek best hold (NOT used) | {methods[peek_best]['oof_acc_nested']:.4f} | {methods[peek_best]['holdout_acc']:.4f} |

Both-gate hits in search: {len(both_hits)}.

## Files

- `v11_abg_blend.py` — pipeline
- `holdout_logits.npz` — cached holdout branch logits
- `metrics.json` — full comparison
- `submission_v11.csv` / `submission_v11_candidate.csv` / `submission_v11_probs.npz`

## Reproduce

```text
baselines\\v7_stgcn\\.venv\\Scripts\\python.exe baselines\\v11\\v11_abg_blend.py
```

No TTA. No sklearn stacker. ≤100MB (logits/probs only).
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in [
        "selected_method", "selected_oof_acc_nested", "selected_holdout_acc",
        "clear_win_and", "overwrite_submission", "ping_disk_saver",
        "delta_oof_vs_v10", "delta_holdout_vs_v9", "both_gate_hits_count",
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
