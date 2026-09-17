"""v9c: method-level equal blend of top nested fuses.

Selected candidate: equal blend of
  A) power-mean over (mf, mf2, st, gru, dc)
  B) equal (mf_avg, st, gru)
  D) power-mean over (mf, mf2, st, gru)

Nested OOF of A/B/D is computed with GroupKFold; equal blend of those
OOF probs has no extra hyperparams -> honest nested score.

Win: nested >= 0.548 or holdout >= 0.566.
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
    spec = importlib.util.spec_from_file_location("v2_model_v9c", V2 / "model.py")
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


def apply_conf_weighted(probs_list, temp):
    confs = [p.max(1, keepdims=True) for p in probs_list]
    logits_c = np.concatenate([np.log(np.maximum(c, 1e-8)) / temp for c in confs], axis=1)
    lc = logits_c - logits_c.max(1, keepdims=True)
    w = np.exp(lc)
    w = w / w.sum(1, keepdims=True)
    return sum(w[:, i : i + 1] * probs_list[i] for i in range(len(probs_list)))


def fit_conf_weighted(probs_list, y, temps=(0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)):
    best = {"temp": 1.0, "oof_acc": -1.0}
    for temp in temps:
        ca = acc(apply_conf_weighted(probs_list, temp).argmax(1), y)
        if ca > best["oof_acc"]:
            best = {"temp": float(temp), "oof_acc": ca}
    return best


def apply_wN(probs_list, w):
    mix = w[0] * probs_list[0]
    for i in range(1, len(probs_list)):
        mix = mix + w[i] * probs_list[i]
    return mix


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


def nested_method_probs(P, yt, us, kind):
    """Return nested OOF probs for a method kind over nh samples."""
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    powers = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    temps = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
    for tr, va in gkf.split(np.arange(n), yt, us):
        if kind == "pow_all5":
            keys = ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"]
            cfg = fit_power_mean([P[k][tr] for k in keys], yt[tr], powers)
            out[va] = apply_power_mean([P[k][va] for k in keys], cfg["p"])
        elif kind == "eq_mfavg_st_gru":
            out[va] = (P["mf_avg"][va] + P["stgcn"][va] + P["gru"][va]) / 3.0
        elif kind == "conf_mfavg_st_gru":
            cfg = fit_conf_weighted([P["mf_avg"][tr], P["stgcn"][tr], P["gru"][tr]], yt[tr], temps)
            out[va] = apply_conf_weighted(
                [P["mf_avg"][va], P["stgcn"][va], P["gru"][va]], cfg["temp"]
            )
        elif kind == "pow_mf_mf2_st_gru":
            keys = ["midfuse", "midfuse2", "stgcn", "gru"]
            cfg = fit_power_mean([P[k][tr] for k in keys], yt[tr], powers)
            out[va] = apply_power_mean([P[k][va] for k in keys], cfg["p"])
        elif kind == "conf_v8_3":
            cfg = fit_conf_weighted([P["midfuse"][tr], P["stgcn"][tr], P["gru"][tr]], yt[tr], temps)
            out[va] = apply_conf_weighted(
                [P["midfuse"][va], P["stgcn"][va], P["gru"][va]], cfg["temp"]
            )
        else:
            raise ValueError(kind)
    return out


def apply_method(kind, P, params):
    if kind == "pow_all5":
        keys = ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"]
        return apply_power_mean([P[k] for k in keys], params["p"])
    if kind == "eq_mfavg_st_gru":
        return (P["mf_avg"] + P["stgcn"] + P["gru"]) / 3.0
    if kind == "conf_mfavg_st_gru":
        return apply_conf_weighted([P["mf_avg"], P["stgcn"], P["gru"]], params["temp"])
    if kind == "pow_mf_mf2_st_gru":
        keys = ["midfuse", "midfuse2", "stgcn", "gru"]
        return apply_power_mean([P[k] for k in keys], params["p"])
    if kind == "conf_v8_3":
        return apply_conf_weighted([P["midfuse"], P["stgcn"], P["gru"]], params["temp"])
    raise ValueError(kind)


def fit_method(kind, P, y):
    if kind == "pow_all5":
        keys = ["midfuse", "midfuse2", "stgcn", "gru", "deepconv"]
        return fit_power_mean([P[k] for k in keys], y)
    if kind == "eq_mfavg_st_gru":
        mix = (P["mf_avg"] + P["stgcn"] + P["gru"]) / 3.0
        return {"oof_acc": acc(mix.argmax(1), y)}
    if kind == "conf_mfavg_st_gru":
        return fit_conf_weighted([P["mf_avg"], P["stgcn"], P["gru"]], y)
    if kind == "pow_mf_mf2_st_gru":
        keys = ["midfuse", "midfuse2", "stgcn", "gru"]
        return fit_power_mean([P[k] for k in keys], y)
    if kind == "conf_v8_3":
        return fit_conf_weighted([P["midfuse"], P["stgcn"], P["gru"]], y)
    raise ValueError(kind)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = V7 / "cache"
    v2m = load_v2_model_mod()

    branches = {
        "midfuse": {
            "kind": "v2", "v2m": v2m, "default": "midfuse",
            "fold_dir": V2 / "checkpoints_midfuse_v2b",
            "holdout_ckpt": V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt",
        },
        "midfuse2": {
            "kind": "v2", "v2m": v2m, "default": "midfuse",
            "fold_dir": V2 / "checkpoints_midfuse_s123",
            "holdout_ckpt": V2 / "checkpoints_midfuse_s123" / "best_holdout.pt",
        },
        "stgcn": {
            "kind": "stgcn", "v2m": v2m, "default": "stgcn_fuse",
            "fold_dir": V7 / "checkpoints_cv",
            "holdout_ckpt": V7 / "checkpoints_v2" / "best_holdout.pt",
        },
        "gru": {
            "kind": "v2", "v2m": v2m, "default": "gru_attn",
            "fold_dir": V2 / "checkpoints_gru",
            "holdout_ckpt": V2 / "checkpoints_gru" / "best_holdout.pt",
        },
        "deepconv": {
            "kind": "v2", "v2m": v2m, "default": "deepconv",
            "fold_dir": V2 / "checkpoints_deepconv",
            "holdout_ckpt": V2 / "checkpoints_deepconv" / "best_holdout.pt",
        },
    }

    oof = dict(np.load(ROOT / "oof_logits_v9b.npz", allow_pickle=False))
    y = oof["y"]
    users = oof["users"]
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(~np.isin(users, list(hold)))[0]
    ho_idx = np.where(np.isin(users, list(hold)))[0]
    yt_nh = y[nh_idx]
    users_nh = users[nh_idx]
    print(f"n={len(y)} nh={len(nh_idx)} ho={len(ho_idx)} device={device}", flush=True)

    L = {k: oof[k][nh_idx] for k in branches}
    P = {k: softmax(L[k]) for k in branches}
    L["mf_avg"] = 0.5 * (L["midfuse"] + L["midfuse2"])
    P["mf_avg"] = softmax(L["mf_avg"])

    single = {f"oof_{k}": acc(L[k].argmax(1), yt_nh) for k in list(branches) + ["mf_avg"]}
    print(json.dumps(single, indent=2), flush=True)

    base_kinds = [
        "conf_v8_3",
        "pow_all5",
        "eq_mfavg_st_gru",
        "conf_mfavg_st_gru",
        "pow_mf_mf2_st_gru",
    ]
    nested_probs = {}
    methods = {}
    for kind in base_kinds:
        np_ = nested_method_probs(P, yt_nh, users_nh, kind)
        nested_probs[kind] = np_
        nested_a = acc(np_.argmax(1), yt_nh)
        params = fit_method(kind, P, yt_nh)
        methods[kind] = {
            "oof_acc": nested_a,
            "oof_acc_in_sample": params.get("oof_acc"),
            "params": {k: v for k, v in params.items() if k != "oof_acc"},
            "family": kind,
            "branches": kind,
        }
        print(f"{kind} nested={nested_a:.4f}", flush=True)

    # Method blends (equal) — no extra hyperparams on nested OOF probs
    blend_specs = {
        "blend_ABD": ["pow_all5", "eq_mfavg_st_gru", "pow_mf_mf2_st_gru"],
        "blend_ABC": ["pow_all5", "eq_mfavg_st_gru", "conf_mfavg_st_gru"],
        "blend_AB": ["pow_all5", "eq_mfavg_st_gru"],
        "blend_AD": ["pow_all5", "pow_mf_mf2_st_gru"],
        "blend_BD": ["eq_mfavg_st_gru", "pow_mf_mf2_st_gru"],
        "blend_ABCD": ["pow_all5", "eq_mfavg_st_gru", "conf_mfavg_st_gru", "pow_mf_mf2_st_gru"],
    }
    for bname, members in blend_specs.items():
        mix = sum(nested_probs[m] for m in members) / float(len(members))
        nested_a = acc(mix.argmax(1), yt_nh)
        methods[bname] = {
            "oof_acc": nested_a,
            "oof_acc_in_sample": nested_a,
            "params": {"members": members, "w": "equal"},
            "family": "method_blend",
            "branches": members,
        }
        print(f"{bname} nested={nested_a:.4f}", flush=True)

    best_name = max(methods, key=lambda k: methods[k]["oof_acc"])
    best = methods[best_name]
    print(f"BEST={best_name} nested={best['oof_acc']:.4f}", flush=True)

    # Holdout LAST
    print("Holdout...", flush=True)
    X_skel, y_full, users_full, _ = load_skel_train_cache(cache)
    X_imu, has_imu, _, _ = load_imu_caches(cache)
    H = {}
    for name in branches:
        H[name] = holdout_branch(branches[name], X_skel, X_imu, has_imu, ho_idx, device)
        print(f"  {name}", flush=True)
    yt_h = y_full[ho_idx]
    HL = dict(H)
    HL["mf_avg"] = 0.5 * (H["midfuse"] + H["midfuse2"])
    HP = {k: softmax(HL[k]) for k in HL}
    hold_single = {f"holdout_{k}": acc(H[k].argmax(1), yt_h) for k in branches}
    hold_single["holdout_mf_avg"] = acc(HL["mf_avg"].argmax(1), yt_h)

    # Fit final params on full nh
    final_params = {k: fit_method(k, P, yt_nh) for k in base_kinds}

    def apply_named(name, probs_map):
        m = methods[name]
        if m["family"] == "method_blend":
            members = m["params"]["members"]
            mixes = [apply_method(mem, probs_map, final_params[mem]) for mem in members]
            return sum(mixes) / float(len(mixes))
        return apply_method(name, probs_map, final_params[name])

    hold_rows = {name: acc(apply_named(name, HP).argmax(1), yt_h) for name in methods}
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
    test_probs, test_logits, paths = test_branch_logits(
        branches, list(branches.keys()), cache, device
    )
    test_logits["mf_avg"] = 0.5 * (test_logits["midfuse"] + test_logits["midfuse2"])
    test_probs["mf_avg"] = softmax(test_logits["mf_avg"])
    mix_te = apply_named(best_name, test_probs)
    pred = mix_te.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(x) for x in pred]})
    sub = ROOT / "submission_v9.csv"
    df.to_csv(sub, index=False)
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
        "track": "v9c_method_blend",
        "v8_baseline_oof": V8_OOF,
        "v8_baseline_holdout": V8_HOLD,
        "use_tta": False,
        "midfuse_seed2": "checkpoints_midfuse_s123",
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
        "protocol": "GroupKFold nested method OOF; equal method-blend; holdout last",
        "submission_v9": str(sub),
        "win_criteria": "nested_OOF>=0.548 or holdout>=0.566",
        "final_params": {k: {kk: vv for kk, vv in v.items() if kk != "oof_acc"} for k, v in final_params.items()},
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    top = sorted(methods_out.items(), key=lambda kv: kv[1]["oof_acc_nested"], reverse=True)
    print("TOP:", flush=True)
    for k, v in top:
        print(f"  {v['oof_acc_nested']:.4f} hold={v['holdout_acc']:.4f} {k}", flush=True)
    print(json.dumps({k: metrics[k] for k in (
        "selected_method", "selected_oof_acc_nested", "selected_holdout_acc",
        "delta_oof_vs_v8", "delta_holdout_vs_v8", "clear_win", "overwrite_submission",
        "ping_disk_saver", "elapsed_sec")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
