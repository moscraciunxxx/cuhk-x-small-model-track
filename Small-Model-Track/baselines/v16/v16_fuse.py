"""v16 nested late-fuse: MidFusePlus (velocity) + s2s/ms2s + v11 branches.

Clear-win overwrite+ping: holdout >= 0.598 OR (holdout >= 0.593 AND nested OOF >= 0.585).
Selection = max nested OOF; holdout last. No TTA / no leaky stackers.
Branches: stronger KD + optional KD seed2 + MidFusePlus + v15 branches.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V7))  # V7 must win over v13/model.py

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    load_imu_caches,
    load_skel_train_cache,
    load_skel_test_cache,
)
# build_stgcn loaded via importlib to avoid v13/model.py shadowing
def _load_build_stgcn():
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("v7_model_fuse", V7 / "model.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.build_model
build_stgcn = None  # set in main


NUM_CLASSES = 40
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
TEMPS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
V15_OOF = 0.5791
V15_HOLD = 0.5881
V13_OOF = 0.5631
V13_HOLD = 0.5703
WIN_OOF_STRICT = 0.585
WIN_HOLD_STRICT = 0.593
WIN_HOLD_ALONE = 0.598
WIN_OOF_SOFT = 0.585
WIN_HOLD_SOFT = 0.593



def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v13", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
def predict_logits_arr(model, xs, xi, flag, device, batch=32):
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


def load_ckpt(kind, ckpt_path, device, v2m, mfwm, ms2sm):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ck.get("model_name", kind)
    ncls = ck.get("num_classes", NUM_CLASSES)
    if kind in ("stgcn", "stgcn_2s"):
        # ensure v7 package wins for model_2s relative imports
        if str(V7) in sys.path:
            sys.path.remove(str(V7))
        sys.path.insert(0, str(V7))
        m = build_stgcn(name if name else kind, num_classes=ncls)
    elif kind == "mfw":
        m = mfwm.build_model("midfuse_plus", num_classes=ncls)
    elif kind == "kd":
        # compact student (default) or midfuse
        sname = name if name in ("compact", "compact_fuse", "midfuse", "mfp") else ck.get("model_name", "compact")
        if sname in ("compact", "compact_fuse"):
            m = v2m.CompactMidFuse(num_classes=ncls)
        elif sname in ("mfp", "midfuse_plus"):
            m = mfwm.build_model("midfuse_plus", num_classes=ncls)
        else:
            m = v2m.build_model("midfuse", num_classes=ncls)
    elif kind == "ms2s":
        m = ms2sm.build_model("ms_stgcn_2s", num_classes=ncls)
    else:
        m = v2m.build_model(name, num_classes=ncls)
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    global build_stgcn
    build_stgcn = _load_build_stgcn()
    v2m = load_v2_model_mod()
    mfwm = load_mod(ROOT / "model.py", "v13_mfw") if (ROOT / "model.py").exists() else None
    ms2sm = None  # lazy

    X_skel, y, users, _meta = load_skel_train_cache(V2 / "cache")
    X_imu, has_imu, _, _ = load_imu_caches(V2 / "cache")
    y = np.asarray(y)
    users = np.asarray(users)
    hold_set = set(DEFAULT_HOLD_OUT_USERS)
    nh_mask = np.array([int(u) not in hold_set for u in users])
    h_mask = ~nh_mask
    nh_idx = np.where(nh_mask)[0]
    h_idx = np.where(h_mask)[0]
    yt_nh, users_nh, yt_h = y[nh_idx], users[nh_idx], y[h_idx]

    z = np.load(V10 / "oof_logits_v10.npz", allow_pickle=True)
    old = ["midfuse", "midfuse2", "midfuse3", "stgcn", "gru", "gru2", "deepconv", "cfuse"]
    L_full = {k: z[k] for k in old}

    new_branches = {}
    V13 = ROOT.parent / "v13"
    # discover new OOF files
    candidates = [
        ("s2s", ROOT / "oof_stgcn2s.npz", ["stgcn_2s", "s2s"], "stgcn_2s"),
        ("s2s", V13 / "oof_stgcn2s.npz", ["stgcn_2s", "s2s"], "stgcn_2s"),
        ("ms2s", ROOT / "oof_ms2s.npz", ["ms_stgcn_2s", "ms2s"], "ms2s"),
        ("ms2s", V13 / "oof_ms2s.npz", ["ms_stgcn_2s", "ms2s"], "ms2s"),
        ("mfw", ROOT / "oof_mfw.npz", ["midfuse_wide", "midfuse_plus", "mfw"], "mfw"),
        ("mfw", ROOT / "oof_mfp.npz", ["midfuse_wide", "midfuse_plus", "mfw"], "mfw"),
        ("kd", ROOT / "oof_kd.npz", ["kd", "student", "compact"], "kd"),
        ("kd2", ROOT / "oof_kd2.npz", ["kd2", "kd", "student", "compact"], "kd2"),
        ("kd_c", ROOT / "oof_kd_c.npz", ["kd_c", "kd", "student", "compact"], "kd_c"),
        ("kd_alt", ROOT / "oof_kd_alt.npz", ["kd_alt", "kd", "student", "compact"], "kd_alt"),
        ("kdv15", ROOT / "oof_kd_v15.npz", ["kd", "student", "compact"], "kdv15"),
    ]
    for alias, path, keys, _ in candidates:
        if alias in new_branches:
            continue
        if not path.exists():
            continue
        d = np.load(path)
        for k in keys:
            if k in d.files:
                L_full[alias] = d[k]
                new_branches[alias] = path
                print(f"loaded {alias} OOF from {path.name} key={k} shape={d[k].shape}", flush=True)
                break

    L = {k: L_full[k][nh_idx] for k in L_full}
    P = {k: softmax(L[k]) for k in L}
    mf_keys = ["midfuse", "midfuse2", "midfuse3"]
    L["mf_avg"] = sum(L[k] for k in mf_keys) / 3.0
    P["mf_avg"] = softmax(L["mf_avg"])
    L["gru_avg"] = 0.5 * (L["gru"] + L["gru2"])
    P["gru_avg"] = softmax(L["gru_avg"])
    if "mfw" in L:
        L["mf4"] = (L["midfuse"] + L["midfuse2"] + L["midfuse3"] + L["mfw"]) / 4.0
        P["mf4"] = softmax(L["mf4"])
        L["mf_mfw"] = 0.5 * (L["mf_avg"] + L["mfw"])
        P["mf_mfw"] = softmax(L["mf_mfw"])

    # holdout
    hz = np.load(V11 / "holdout_logits.npz")
    H = {k: hz[k] for k in old}
    hold_files = [
        ("s2s", ROOT / "holdout_stgcn2s.npz", ["stgcn_2s", "s2s"]),
        ("s2s", V13 / "holdout_stgcn2s.npz", ["stgcn_2s", "s2s"]),
        ("ms2s", ROOT / "holdout_ms2s.npz", ["ms_stgcn_2s", "ms2s"]),
        ("ms2s", V13 / "holdout_ms2s.npz", ["ms_stgcn_2s", "ms2s"]),
        ("mfw", ROOT / "holdout_mfw.npz", ["midfuse_wide", "midfuse_plus", "mfw"]),
        ("mfw", ROOT / "holdout_mfp.npz", ["midfuse_wide", "midfuse_plus", "mfw"]),
        ("kd", ROOT / "holdout_kd.npz", ["kd", "student", "compact"]),
        ("kd2", ROOT / "holdout_kd2.npz", ["kd2", "kd", "student", "compact"]),
        ("kd_c", ROOT / "holdout_kd_c.npz", ["kd_c", "kd", "student", "compact"]),
        ("kd_alt", ROOT / "holdout_kd_alt.npz", ["kd_alt", "kd", "student", "compact"]),
        ("kdv15", ROOT / "holdout_kd_v15.npz", ["kd", "student", "compact"]),
    ]
    for alias, path, keys in hold_files:
        if alias in H:
            continue
        if not path.exists():
            continue
        d = np.load(path)
        for k in keys:
            if k in d.files:
                H[alias] = d[k]
                print(f"loaded holdout {alias}", flush=True)
                break

    HL = dict(H)
    HP = {k: softmax(HL[k]) for k in HL}
    HL["mf_avg"] = sum(H[k] for k in mf_keys) / 3.0
    HP["mf_avg"] = softmax(HL["mf_avg"])
    HL["gru_avg"] = 0.5 * (H["gru"] + H["gru2"])
    HP["gru_avg"] = softmax(HL["gru_avg"])
    if "mfw" in H:
        HL["mf4"] = (H["midfuse"] + H["midfuse2"] + H["midfuse3"] + H["mfw"]) / 4.0
        HP["mf4"] = softmax(HL["mf4"])
        HL["mf_mfw"] = 0.5 * (HL["mf_avg"] + H["mfw"])
        HP["mf_mfw"] = softmax(HL["mf_mfw"])

    for k in sorted(P):
        print(
            f"solo {k}: oof={acc(P[k].argmax(1), yt_nh):.4f} hold={acc(HP[k].argmax(1), yt_h):.4f}",
            flush=True,
        )

    # graph branch preference: s2s replaces stgcn when present for some recipes
    g_old = "stgcn"
    g_new = "s2s" if "s2s" in P else ("ms2s" if "ms2s" in P else "stgcn")

    bases_spec = {
        "A": (["mf_avg", "stgcn", "gru", "cfuse"], "pow"),
        "B": (["mf_avg", "stgcn", "gru_avg"], "eq"),
        "G": (["mf_avg", "stgcn", "gru_avg"], "pow"),
        "E": (["mf_avg", "stgcn", "gru_avg", "cfuse"], "pow"),
        "C": (["mf_avg", "stgcn", "gru", "cfuse"], "eq"),
    }
    if "s2s" in P:
        bases_spec.update(
            {
                "As": (["mf_avg", "s2s", "gru", "cfuse"], "pow"),
                "Bs": (["mf_avg", "s2s", "gru_avg"], "eq"),
                "Gs": (["mf_avg", "s2s", "gru_avg"], "pow"),
                "Es": (["mf_avg", "s2s", "gru_avg", "cfuse"], "pow"),
                "Cs": (["mf_avg", "s2s", "gru", "cfuse"], "eq"),
                "pow_mf_s2s_gru": (["mf_avg", "s2s", "gru"], "pow"),
                "eq_mf_st_s2s": (["mf_avg", "stgcn", "s2s"], "eq"),
                "pow_mf_st_s2s_gru": (["mf_avg", "stgcn", "s2s", "gru_avg"], "pow"),
                "conf_mf_s2s_gruavg": (["mf_avg", "s2s", "gru_avg"], "conf"),
                "eq_st_s2s": (["stgcn", "s2s"], "eq"),
            }
        )
    if "ms2s" in P:
        bases_spec.update(
            {
                "Am": (["mf_avg", "ms2s", "gru", "cfuse"], "pow"),
                "Bm": (["mf_avg", "ms2s", "gru_avg"], "eq"),
                "Gm": (["mf_avg", "ms2s", "gru_avg"], "pow"),
                "Em": (["mf_avg", "ms2s", "gru_avg", "cfuse"], "pow"),
                "pow_mf_ms_gru": (["mf_avg", "ms2s", "gru"], "pow"),
                "eq_mf_st_ms": (["mf_avg", "stgcn", "ms2s"], "eq"),
                "pow_mf_st_ms_gru": (["mf_avg", "stgcn", "ms2s", "gru_avg"], "pow"),
                "eq_s2s_ms": (["s2s", "ms2s"], "eq") if "s2s" in P else (["stgcn", "ms2s"], "eq"),
            }
        )
        if "s2s" in P:
            bases_spec["eq_mf_s2s_ms"] = (["mf_avg", "s2s", "ms2s"], "eq")
            bases_spec["pow_mf_s2s_ms_gru"] = (["mf_avg", "s2s", "ms2s", "gru_avg"], "pow")
            bases_spec["eq_mf_st_s2s_ms"] = (["mf_avg", "stgcn", "s2s", "ms2s"], "eq")
            bases_spec["Asm"] = (["mf_avg", "s2s", "ms2s", "gru", "cfuse"], "pow")
            bases_spec["Bsm"] = (["mf_avg", "s2s", "ms2s", "gru_avg"], "eq")
            bases_spec["Gsm"] = (["mf_avg", "s2s", "ms2s", "gru_avg"], "pow")
    def add_kd_family(alias):
        if alias not in P:
            return
        bases_spec.update(
            {
                f"eq_mf_{alias}": (["mf_avg", alias], "eq"),
                f"pow_mf_{alias}_gru": (["mf_avg", alias, "gru_avg"], "pow"),
                f"eq_mf_st_{alias}": (["mf_avg", "stgcn", alias], "eq"),
                f"eq_{alias}_s2s": ([alias, "s2s"], "eq") if "s2s" in P else ([alias, "stgcn"], "eq"),
            }
        )
        if "s2s" in P:
            bases_spec[f"eq_mf_s2s_{alias}"] = (["mf_avg", "s2s", alias], "eq")
            bases_spec[f"eq_mf_st_s2s_{alias}"] = (["mf_avg", "stgcn", "s2s", alias], "eq")
            bases_spec[f"pow_mf_s2s_{alias}_gru"] = (["mf_avg", "s2s", alias, "gru_avg"], "pow")
        if "ms2s" in P:
            bases_spec[f"eq_{alias}_ms"] = ([alias, "ms2s"], "eq")
            bases_spec[f"eq_mf_{alias}_ms"] = (["mf_avg", alias, "ms2s"], "eq")
            if "s2s" in P:
                bases_spec[f"eq_mf_s2s_{alias}_ms"] = (["mf_avg", "s2s", alias, "ms2s"], "eq")
        if "mfw" in P:
            bases_spec[f"eq_{alias}_mfp"] = ([alias, "mfw"], "eq")
            bases_spec[f"eq_mf_{alias}_mfp"] = (["mf_avg", alias, "mfw"], "eq")
            if "s2s" in P:
                bases_spec[f"eq_{alias}_s2s_mfp"] = ([alias, "s2s", "mfw"], "eq")
                bases_spec[f"eq_mf_s2s_{alias}_mfp"] = (["mf_avg", "s2s", alias, "mfw"], "eq")

    for _alias in ("kd", "kd2", "kd_c", "kd_alt", "kdv15"):
        add_kd_family(_alias)
    if "kd" in P and "kd2" in P:
        bases_spec["eq_kd_kd2"] = (["kd", "kd2"], "eq")
        bases_spec["pow_kd_kd2"] = (["kd", "kd2"], "pow")
        if "s2s" in P:
            bases_spec["eq_kd_kd2_s2s"] = (["kd", "kd2", "s2s"], "eq")
            bases_spec["eq_mf_kd_kd2_s2s"] = (["mf_avg", "kd", "kd2", "s2s"], "eq")
        if "mfw" in P:
            bases_spec["eq_kd_kd2_mfp"] = (["kd", "kd2", "mfw"], "eq")
            if "s2s" in P:
                bases_spec["eq_kd_kd2_s2s_mfp"] = (["kd", "kd2", "s2s", "mfw"], "eq")
        # MidFusePlus (v15) may be stored under alias mfw from oof_mfp.npz
    if "mfw" in P:
        bases_spec.update(
            {
                "eq_mf_mfp": (["mf_avg", "mfw"], "eq"),
                "pow_mf_mfp_gru": (["mf_avg", "mfw", "gru_avg"], "pow"),
                "eq_mf_st_mfp": (["mf_avg", "stgcn", "mfw"], "eq"),
            }
        )
        if "s2s" in P:
            bases_spec["eq_mf_s2s_mfp"] = (["mf_avg", "s2s", "mfw"], "eq")
            bases_spec["eq_mf_st_s2s_mfp"] = (["mf_avg", "stgcn", "s2s", "mfw"], "eq")
            bases_spec["pow_mf_s2s_mfp_gru"] = (["mf_avg", "s2s", "mfw", "gru_avg"], "pow")
        if "ms2s" in P:
            bases_spec["eq_mfp_ms"] = (["mfw", "ms2s"], "eq")
            bases_spec["eq_mf_mfp_ms"] = (["mf_avg", "mfw", "ms2s"], "eq")
    if "mfw" in P:
        bases_spec.update(
            {
                "Aw": (["mf_mfw", "stgcn", "gru", "cfuse"], "pow"),
                "Bw": (["mf_mfw", "stgcn", "gru_avg"], "eq"),
                "Gw": (["mf_mfw", "stgcn", "gru_avg"], "pow"),
                "A4": (["mf4", g_new, "gru", "cfuse"], "pow"),
                "B4": (["mf4", g_new, "gru_avg"], "eq"),
                "G4": (["mf4", g_new, "gru_avg"], "pow"),
                "pow_mfw_st_gru": (["mfw", g_new, "gru"], "pow"),
            }
        )
        if "s2s" in P:
            bases_spec["Aws"] = (["mf_mfw", "s2s", "gru", "cfuse"], "pow")
            bases_spec["Bws"] = (["mf_mfw", "s2s", "gru_avg"], "eq")

    # Solo KD / MFP branches as first-class methods (strong students)
    for solo in ("kd", "kd2", "kd_c", "kd_alt", "kdv15", "mfw", "s2s", "ms2s"):
        if solo in P:
            bases_spec[f"solo_{solo}"] = ([solo], "eq")

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
        print(f"base {name} nested={acc(nested_base[name].argmax(1), yt_nh):.4f} cfg={cfg}", flush=True)

    nested_base["v11"] = (4 * nested_base["A"] + 2 * nested_base["B"] + 3 * nested_base["G"]) / 9.0
    final_cfg["v11"] = {"family": "abg_weight", "params": {"wa": 4, "wb": 2, "wg": 3, "members": ["A", "B", "G"]}}

    def apply_base(name, probs_map):
        if name == "v11":
            return (4 * apply_base("A", probs_map) + 2 * apply_base("B", probs_map) + 3 * apply_base("G", probs_map)) / 9.0
        cfg = final_cfg[name]
        keys, fam, params = cfg["keys"], cfg["family"], cfg["params"]
        if fam == "eq":
            return apply_equal([probs_map[k] for k in keys])
        if fam == "pow":
            return apply_power_mean([probs_map[k] for k in keys], params["p"])
        if fam == "conf":
            return apply_conf([probs_map[k] for k in keys], params["temp"])
        raise ValueError(fam)

    hold_base = {n: apply_base(n, HP) for n in list(bases_spec) + ["v11"]}

    methods = {}

    def register(name, family, oof_p, hold_p, params=None):
        methods[name] = {
            "oof_acc_nested": acc(oof_p.argmax(1), yt_nh),
            "holdout_acc": acc(hold_p.argmax(1), yt_h),
            "family": family,
            "params": params or {},
            "_oof": oof_p,
            "_hold": hold_p,
        }

    for name in bases_spec:
        register(name, bases_spec[name][1], nested_base[name], hold_base[name], final_cfg[name])
    register("v11ref", "abg_weight", nested_base["v11"], hold_base["v11"], final_cfg["v11"]["params"])

    # classic ABG grid
    for wa in range(1, 8):
        for wb in range(0, 6):
            for wg in range(0, 6):
                if wa + wb + wg == 0:
                    continue
                o = (wa * nested_base["A"] + wb * nested_base["B"] + wg * nested_base["G"]) / (wa + wb + wg)
                h = (wa * hold_base["A"] + wb * hold_base["B"] + wg * hold_base["G"]) / (wa + wb + wg)
                register(f"wABG_{wa}_{wb}_{wg}", "abg_weight", o, h, {"wa": wa, "wb": wb, "wg": wg, "members": ["A", "B", "G"]})

    # s2s ABG
    if "s2s" in P:
        for wa in range(1, 7):
            for wb in range(0, 5):
                for wg in range(0, 5):
                    if wa + wb + wg == 0:
                        continue
                    o = (wa * nested_base["As"] + wb * nested_base["Bs"] + wg * nested_base["Gs"]) / (wa + wb + wg)
                    h = (wa * hold_base["As"] + wb * hold_base["Bs"] + wg * hold_base["Gs"]) / (wa + wb + wg)
                    register(
                        f"wABGs_{wa}_{wb}_{wg}",
                        "abg_weight",
                        o,
                        h,
                        {"wa": wa, "wb": wb, "wg": wg, "members": ["As", "Bs", "Gs"]},
                    )
        # mix classic with s2s
        for wa, wb, wg in [(4, 2, 3), (3, 2, 2), (5, 2, 3), (4, 3, 2), (2, 1, 1)]:
            for mix in ["As", "Bs", "Gs", "Es", "pow_mf_s2s_gru", "eq_mf_st_s2s", "pow_mf_st_s2s_gru"]:
                for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    base_o = (wa * nested_base["A"] + wb * nested_base["B"] + wg * nested_base["G"]) / (wa + wb + wg)
                    base_h = (wa * hold_base["A"] + wb * hold_base["B"] + wg * hold_base["G"]) / (wa + wb + wg)
                    o = (base_o * (wa + wb + wg) + w * nested_base[mix]) / (wa + wb + wg + w)
                    h = (base_h * (wa + wb + wg) + w * hold_base[mix]) / (wa + wb + wg + w)
                    register(
                        f"v11style{wa}{wb}{wg}_plus_{mix}_x{w}",
                        "v11_plus",
                        o,
                        h,
                        {"wa": wa, "wb": wb, "wg": wg, "extra": mix, "w": w},
                    )

    if "ms2s" in P:
        for wa, wb, wg in [(4, 2, 3), (3, 2, 2), (5, 2, 3), (2, 1, 1), (4, 3, 2)]:
            if all(k in nested_base for k in ("Am", "Bm", "Gm")):
                o = (wa * nested_base["Am"] + wb * nested_base["Bm"] + wg * nested_base["Gm"]) / (wa + wb + wg)
                h = (wa * hold_base["Am"] + wb * hold_base["Bm"] + wg * hold_base["Gm"]) / (wa + wb + wg)
                register(
                    f"wABGm_{wa}_{wb}_{wg}",
                    "abg_weight",
                    o,
                    h,
                    {"wa": wa, "wb": wb, "wg": wg, "members": ["Am", "Bm", "Gm"]},
                )
        for mix in ["Am", "Bm", "Gm", "Em", "pow_mf_ms_gru", "eq_mf_st_ms", "pow_mf_st_ms_gru", "eq_s2s_ms",
                    "eq_mf_s2s_ms", "pow_mf_s2s_ms_gru", "eq_mf_st_s2s_ms", "Asm", "Bsm", "Gsm"]:
            if mix not in nested_base:
                continue
            for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                o = (9.0 * nested_base["v11"] + w * nested_base[mix]) / (9.0 + w)
                h = (9.0 * hold_base["v11"] + w * hold_base[mix]) / (9.0 + w)
                register(f"v11_plus_{mix}_x{w}", "v11_plus", o, h, {"extra": mix, "w": w})
        # classic ABG + ms extras (incl. v13-selected style with ms)
        for wa, wb, wg in [(4, 2, 3), (3, 2, 2), (5, 2, 3), (2, 1, 1), (4, 3, 2)]:
            for mix in ["Am", "Bm", "Gm", "eq_mf_st_ms", "pow_mf_st_ms_gru", "eq_s2s_ms",
                        "eq_mf_s2s_ms", "eq_mf_st_s2s_ms", "pow_mf_s2s_ms_gru"]:
                if mix not in nested_base:
                    continue
                for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    base_o = (wa * nested_base["A"] + wb * nested_base["B"] + wg * nested_base["G"]) / (wa + wb + wg)
                    base_h = (wa * hold_base["A"] + wb * hold_base["B"] + wg * hold_base["G"]) / (wa + wb + wg)
                    o = (base_o * (wa + wb + wg) + w * nested_base[mix]) / (wa + wb + wg + w)
                    h = (base_h * (wa + wb + wg) + w * hold_base[mix]) / (wa + wb + wg + w)
                    register(
                        f"v11style{wa}{wb}{wg}_plus_{mix}_x{w}",
                        "v11_plus",
                        o,
                        h,
                        {"wa": wa, "wb": wb, "wg": wg, "extra": mix, "w": w},
                    )
        # v13 selected + ms branch
        if "eq_mf_st_s2s" in nested_base:
            for mix in ["Am", "Bm", "Gm", "eq_mf_st_ms", "eq_s2s_ms", "eq_mf_s2s_ms", "eq_mf_st_s2s_ms"]:
                if mix not in nested_base:
                    continue
                for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    # reconstruct v13 selected nested: wa=2,wb=1,wg=1 + eq_mf_st_s2s x1
                    base_o = (2 * nested_base["A"] + nested_base["B"] + nested_base["G"]) / 4.0
                    base_h = (2 * hold_base["A"] + hold_base["B"] + hold_base["G"]) / 4.0
                    v13_o = (base_o * 4.0 + 1.0 * nested_base["eq_mf_st_s2s"]) / 5.0
                    v13_h = (base_h * 4.0 + 1.0 * hold_base["eq_mf_st_s2s"]) / 5.0
                    o = (v13_o * 5.0 + w * nested_base[mix]) / (5.0 + w)
                    h = (v13_h * 5.0 + w * hold_base[mix]) / (5.0 + w)
                    register(
                        f"v13sel_plus_{mix}_x{w}",
                        "v13_plus",
                        o,
                        h,
                        {"extra": mix, "w": w},
                    )

    if ("kd" in P) or ("kd2" in P) or ("kdv15" in P):
        kd_mixes = [k for k in nested_base if (
            k.startswith("eq_") or k.startswith("pow_")
        ) and any(x in k for x in ("kd", "kd2", "kd_c", "kd_alt", "kdv15"))]
        for mix in kd_mixes:
            for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                o = (9.0 * nested_base["v11"] + w * nested_base[mix]) / (9.0 + w)
                h = (9.0 * hold_base["v11"] + w * hold_base[mix]) / (9.0 + w)
                register(f"v11_plus_{mix}_x{w}", "v11_plus", o, h, {"extra": mix, "w": w})
            if "eq_mf_st_s2s" in nested_base:
                for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    base_o = (2 * nested_base["A"] + nested_base["B"] + nested_base["G"]) / 4.0
                    base_h = (2 * hold_base["A"] + hold_base["B"] + hold_base["G"]) / 4.0
                    v13_o = (base_o * 4.0 + 1.0 * nested_base["eq_mf_st_s2s"]) / 5.0
                    v13_h = (base_h * 4.0 + 1.0 * hold_base["eq_mf_st_s2s"]) / 5.0
                    o = (v13_o * 5.0 + w * nested_base[mix]) / (5.0 + w)
                    h = (v13_h * 5.0 + w * hold_base[mix]) / (5.0 + w)
                    register(f"v13sel_plus_{mix}_x{w}", "v13_plus", o, h, {"extra": mix, "w": w})
        for wa, wb, wg in [(4, 2, 3), (5, 2, 3), (2, 1, 1), (3, 2, 2)]:
            for mix in kd_mixes:
                for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
                    base_o = (wa * nested_base["A"] + wb * nested_base["B"] + wg * nested_base["G"]) / (wa + wb + wg)
                    base_h = (wa * hold_base["A"] + wb * hold_base["B"] + wg * hold_base["G"]) / (wa + wb + wg)
                    o = (base_o * (wa + wb + wg) + w * nested_base[mix]) / (wa + wb + wg + w)
                    h = (base_h * (wa + wb + wg) + w * hold_base[mix]) / (wa + wb + wg + w)
                    register(
                        f"v11style{wa}{wb}{wg}_plus_{mix}_x{w}",
                        "v11_plus",
                        o,
                        h,
                        {"wa": wa, "wb": wb, "wg": wg, "extra": mix, "w": w},
                    )

    if "mfw" in P:
        for mix in ["Aw", "Bw", "Gw", "A4", "B4", "pow_mfw_st_gru"]:
            if mix not in nested_base:
                continue
            for w in [0.5, 1.0, 1.5, 2.0]:
                o = (9.0 * nested_base["v11"] + w * nested_base[mix]) / (9.0 + w)
                h = (9.0 * hold_base["v11"] + w * hold_base[mix]) / (9.0 + w)
                register(f"v11_plus_{mix}_x{w}", "v11_plus", o, h, {"extra": mix, "w": w})

    # SELECT
    best_name = max(methods, key=lambda k: methods[k]["oof_acc_nested"])
    oof_best = methods[best_name]["oof_acc_nested"]
    hold_best = methods[best_name]["holdout_acc"]
    clear_win = bool(hold_best >= WIN_HOLD_ALONE or (hold_best >= WIN_HOLD_SOFT and oof_best >= WIN_OOF_SOFT))
    soft_improve = bool(hold_best >= WIN_HOLD_SOFT and oof_best >= WIN_OOF_SOFT)
    peek_best = max(methods, key=lambda k: methods[k]["holdout_acc"])
    print(f"SELECTED={best_name} nested={oof_best:.4f} hold={hold_best:.4f} clear={clear_win}", flush=True)
    print(
        f"peek hold={methods[peek_best]['holdout_acc']:.4f} {peek_best} oof={methods[peek_best]['oof_acc_nested']:.4f}",
        flush=True,
    )
    eligible = {k: v for k, v in methods.items() if v["oof_acc_nested"] >= WIN_OOF_SOFT}
    if eligible:
        bh = max(eligible, key=lambda k: eligible[k]["holdout_acc"])
        print(f"best hold OOF>={WIN_OOF_SOFT}: {bh} hold={eligible[bh]['holdout_acc']:.4f} oof={eligible[bh]['oof_acc_nested']:.4f}", flush=True)

    def resolve_probs(branch_logits):
        M_L = dict(branch_logits)
        M = {k: softmax(v) for k, v in M_L.items()}
        M_L["mf_avg"] = sum(M_L[k] for k in mf_keys) / 3.0
        M["mf_avg"] = softmax(M_L["mf_avg"])
        M_L["gru_avg"] = 0.5 * (M_L["gru"] + M_L["gru2"])
        M["gru_avg"] = softmax(M_L["gru_avg"])
        if "mfw" in M_L:
            M_L["mf4"] = (M_L["midfuse"] + M_L["midfuse2"] + M_L["midfuse3"] + M_L["mfw"]) / 4.0
            M["mf4"] = softmax(M_L["mf4"])
            M_L["mf_mfw"] = 0.5 * (M_L["mf_avg"] + M_L["mfw"])
            M["mf_mfw"] = softmax(M_L["mf_mfw"])
        return M

    def apply_named(name, probs_map):
        m = methods[name]
        fam, params = m["family"], m["params"]
        if fam in ("pow", "eq", "conf"):
            return apply_base(name, probs_map)
        if fam == "abg_weight":
            members = params["members"]
            wa, wb, wg = params["wa"], params["wb"], params["wg"]
            mixes = [apply_base(mem, probs_map) for mem in members]
            return (wa * mixes[0] + wb * mixes[1] + wg * mixes[2]) / (wa + wb + wg)
        if fam == "v11_plus":
            if "wa" in params:
                wa, wb, wg = params["wa"], params["wb"], params["wg"]
                base = (wa * apply_base("A", probs_map) + wb * apply_base("B", probs_map) + wg * apply_base("G", probs_map)) / (wa + wb + wg)
                extra = apply_base(params["extra"], probs_map)
                w = params["w"]
                return (base * (wa + wb + wg) + w * extra) / (wa + wb + wg + w)
            v = apply_base("v11", probs_map)
            e = apply_base(params["extra"], probs_map)
            w = params["w"]
            return (9.0 * v + w * e) / (9.0 + w)
        if fam == "v13_plus":
            base = (2 * apply_base("A", probs_map) + apply_base("B", probs_map) + apply_base("G", probs_map)) / 4.0
            v13 = (base * 4.0 + 1.0 * apply_base("eq_mf_st_s2s", probs_map)) / 5.0
            e = apply_base(params["extra"], probs_map)
            w = params["w"]
            return (v13 * 5.0 + w * e) / (5.0 + w)
        raise ValueError(fam)

    # Test logits
    print("building test logits...", flush=True)
    Xte, te_ids = load_skel_test_cache(V2 / "cache")
    imu_te = np.load(V2 / "cache" / "imu_test.npz")
    Xte_imu = imu_te["X"]
    te_flag = imu_te["has_imu"].astype(np.float32)

    branches_meta = {
        "midfuse": ("v2", V2 / "checkpoints_midfuse_v2b", "midfuse"),
        "midfuse2": ("v2", V2 / "checkpoints_midfuse_s123", "midfuse"),
        "midfuse3": ("v2", V2 / "checkpoints_midfuse_s7", "midfuse"),
        "stgcn": ("stgcn", V7 / "checkpoints_cv", "stgcn_fuse"),
        "gru": ("v2", V2 / "checkpoints_gru", "gru_attn"),
        "gru2": ("v2", V2 / "checkpoints_gru_s7", "gru_attn"),
        "deepconv": ("v2", V2 / "checkpoints_deepconv", "deepconv"),
        "cfuse": ("v2", V2 / "checkpoints_compact_fuse", "compact_fuse"),
    }
    if "s2s" in P:
        s2s_ckpt = ROOT / "checkpoints_stgcn2s"
        if not s2s_ckpt.exists():
            s2s_ckpt = V13 / "checkpoints_stgcn2s"
        branches_meta["s2s"] = ("stgcn_2s", s2s_ckpt, "stgcn_2s")
    if "ms2s" in P:
        ms_dir = ROOT / "checkpoints_ms2s"
        if not (ms_dir / "best_fold0.pt").exists():
            ms_dir = ROOT.parent / "v14" / "checkpoints_ms2s"
        branches_meta["ms2s"] = ("ms2s", ms_dir, "ms_stgcn_2s")
    if "mfw" in P:
        mfw_dir = ROOT / "checkpoints_mfp"
        if not (mfw_dir / "best_fold0.pt").exists():
            mfw_dir = ROOT / "checkpoints_mfw"
        branches_meta["mfw"] = ("mfw", mfw_dir, "midfuse_plus")
    if "kd" in P:
        branches_meta["kd"] = ("kd", ROOT / "checkpoints_kd", "compact")
    if "kd2" in P:
        branches_meta["kd2"] = ("kd", ROOT / "checkpoints_kd2", "compact")
    if "kd_c" in P:
        branches_meta["kd_c"] = ("kd", ROOT / "checkpoints_kd_c", "compact")
    if "kd_alt" in P:
        branches_meta["kd_alt"] = ("kd", ROOT / "checkpoints_kd_alt", "compact")
    if "kdv15" in P:
        kdv15_dir = ROOT.parent / "v15" / "checkpoints_kd"
        branches_meta["kdv15"] = ("kd", kdv15_dir, "compact")

    test_logits = {}
    for name, (kind, fold_dir, default) in branches_meta.items():
        accums = []
        for fi in range(5):
            ckpt = fold_dir / f"best_fold{fi}.pt"
            if not ckpt.exists():
                print(f"MISSING {ckpt}", flush=True)
                continue
            if kind == "ms2s" and ms2sm is None:
                ms2sm = load_mod(ROOT / "model_ms_stgcn.py", "v13_ms2s")
            m = load_ckpt(kind, ckpt, device, v2m, mfwm, ms2sm)
            accums.append(predict_logits_arr(m, Xte, Xte_imu, te_flag, device))
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not accums:
            raise FileNotFoundError(name)
        test_logits[name] = sum(accums) / float(len(accums))
        print(f"test {name} folds={len(accums)}", flush=True)

    TP = resolve_probs(test_logits)
    test_probs = apply_named(best_name, TP)
    pred = test_probs.argmax(1).astype(int)

    sub_v11 = pd.read_csv(V11 / "submission_v11.csv")
    sub = sub_v11.copy()
    sub.iloc[:, 1] = pred[: len(sub)]
    sub.to_csv(ROOT / "submission_v16_candidate.csv", index=False)
    sub.to_csv(ROOT / "submission_v16.csv", index=False)
    np.savez_compressed(ROOT / "submission_v16_probs.npz", probs=test_probs, pred=pred)

    overwrite = False
    ping = False
    if clear_win:
        sub.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submission_README.txt").write_text(
            (
                f"v16 clear-win: nested OOF {oof_best:.4f} holdout {hold_best:.4f} "
                f"(gate hold>=0.598, or hold>=0.593 & OOF>=0.585)\n"
                f"method={best_name} params={json.dumps(methods[best_name]['params'])}\n"
                f"ping_disk_saver=true.\n"
            ),
            encoding="utf-8",
        )
        overwrite = True
        ping = True
        print("OVERWROTE track submission.csv", flush=True)
    else:
        print("NO overwrite - track submission remains v15", flush=True)

    table = sorted(
        [
            {
                "name": k,
                "nested_oof": v["oof_acc_nested"],
                "holdout": v["holdout_acc"],
                "family": v["family"],
                "params": v["params"],
            }
            for k, v in methods.items()
        ],
        key=lambda r: (-r["nested_oof"], -r["holdout"]),
    )
    metrics = {
        "selected": best_name,
        "selected_nested_oof": oof_best,
        "selected_holdout": hold_best,
        "v15_nested_oof": V15_OOF, "v15_holdout": V15_HOLD,
        "v13_nested_oof": V13_OOF, "v11_nested_oof": 0.5614,
        "v13_holdout": V13_HOLD, "v11_holdout": 0.5663, "soft_improve": soft_improve,
        "clear_win": clear_win,
        "overwrite_submission": overwrite,
        "ping_disk_saver": ping,
        "new_branches": list(new_branches),
        "peek_best_hold": {
            "name": peek_best,
            "holdout": methods[peek_best]["holdout_acc"],
            "nested_oof": methods[peek_best]["oof_acc_nested"],
        },
        "n_methods": len(methods),
        "top20": table[:20],
        "elapsed_sec": time.time() - t0,
    }
    if eligible:
        metrics["best_hold_oof_eligible"] = {
            "name": bh,
            "holdout": eligible[bh]["holdout_acc"],
            "nested_oof": eligible[bh]["oof_acc_nested"],
        }
    with open(ROOT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": metrics, "all_methods": table}, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()



