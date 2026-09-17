"""v19: compress over-budget peek recipes to <=100MB fold-ckpt budget.

Primary targets:
  - conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02 hold=0.6436 (clears hold>=0.642) ~148MB raw
  - pow_kd_mf_v13_kd_mf_a02_kd_eq_a02 hold=0.6416 (near gate) ~106MB raw

Strategies: fp16 weight packing; optional fold-subset ensembles.
Clear-win v19: hold>=0.642 OR (hold>=0.637 AND nested OOF>=0.680), size<=100MB.
No Chrome / TTA / leaky stackers.
"""
from __future__ import annotations

import json
import sys
import time
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
V18 = ROOT.parent / "v18"
V2 = ROOT.parent / "skeleton_imu_v2"
V7 = ROOT.parent / "v7_stgcn"
TRACK = ROOT.parent.parent
sys.path.insert(0, str(V2))

from dataset import DEFAULT_HOLD_OUT_USERS, load_skel_train_cache  # noqa: E402

NUM_CLASSES = 40
WIN_HOLD_ALONE = 0.642
WIN_HOLD_SOFT = 0.637
WIN_OOF_SOFT = 0.680
BUDGET_MB = 100.0

# reuse fuse helpers from v18
sys.path.insert(0, str(V18))
from v18_fuse_kd import (  # noqa: E402
    softmax,
    acc,
    apply_power_mean,
    apply_equal,
    apply_conf,
    fit_power_mean,
    fit_conf,
    nested_family,
    hold_family,
    final_cfg,
)


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
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    outs = []
    for xb, ib, fb in loader:
        outs.append(model(xb.to(device), ib.to(device), fb.to(device)).float().cpu().numpy())
    return np.concatenate(outs, 0)


def load_v2m():
    spec = importlib.util.spec_from_file_location("v2m_v19", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(path, device, v2m, half_storage=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = ck.get("model_name", "compact")
    ncls = ck.get("num_classes", NUM_CLASSES)
    if name in ("compact", "compact_fuse"):
        m = v2m.CompactMidFuse(num_classes=ncls)
    else:
        m = v2m.build_model("midfuse", num_classes=ncls)
    sd = ck["model_state"]
    # cast half tensors back to float for Module
    sd_f = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
    m.load_state_dict(sd_f)
    m.to(device).eval()
    return m


def fold_size_mb(ckpt_dir: Path, fold_ids):
    total = 0
    for fi in fold_ids:
        p = ckpt_dir / f"best_fold{fi}.pt"
        total += p.stat().st_size
    return total / (1024 * 1024)


def pack_fp16(src_dir: Path, dst_dir: Path, fold_ids):
    dst_dir.mkdir(parents=True, exist_ok=True)
    sizes = []
    for fi in fold_ids:
        src = src_dir / f"best_fold{fi}.pt"
        ck = torch.load(src, map_location="cpu", weights_only=False)
        sd = ck["model_state"]
        sd_h = {
            k: (v.half() if torch.is_floating_point(v) and v.dtype == torch.float32 else v)
            for k, v in sd.items()
        }
        out = {
            "model_state": sd_h,
            "model_name": ck.get("model_name"),
            "num_classes": ck.get("num_classes", NUM_CLASSES),
            "val_acc": ck.get("val_acc"),
            "epoch": ck.get("epoch"),
            "n_params": ck.get("n_params"),
            "tag": ck.get("tag"),
            "kd_T": ck.get("kd_T"),
            "kd_alpha": ck.get("kd_alpha"),
            "teacher": ck.get("teacher"),
            "seed": ck.get("seed"),
            "storage": "fp16",
        }
        dst = dst_dir / f"best_fold{fi}.pt"
        torch.save(out, dst)
        sizes.append(dst.stat().st_size / (1024 * 1024))
    return sum(sizes)


@torch.no_grad()
def ensemble_holdout(ckpt_dir, fold_ids, Xh, Xi, flag, device, v2m):
    accums = []
    for fi in fold_ids:
        m = load_ckpt(ckpt_dir / f"best_fold{fi}.pt", device, v2m)
        accums.append(predict_logits_arr(m, Xh, Xi, flag, device))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return softmax(sum(accums) / float(len(accums)))


def load_probs_maps():
    P, H = {}, {}
    y_oof = users = None
    y_h = None
    for alias in ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02", "kd_T4_a02", "kd_alt", "kd_c", "kd_a02", "kd3"]:
        oof_p = V18 / f"oof_{alias}.npz"
        h_p = V18 / f"holdout_{alias}.npz"
        if not oof_p.exists():
            continue
        zo = np.load(oof_p)
        zh = np.load(h_p)
        key = [k for k in zo.keys() if k not in ("y", "users")][0]
        P[alias] = zo[key]
        H[alias] = zh[key]
        y_oof = zo["y"]
        users = zo["users"]
        y_h = zh["y"]
    return P, H, y_oof, users, y_h


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    v2m = load_v2m()

    # load holdout arrays from cache
    Xtr, ytr, users_all, _ = load_skel_train_cache(V2 / "cache")
    imu = np.load(V2 / "cache" / "imu_train.npz")
    Ximu = imu["X"]
    flag = imu["has_imu"].astype(np.float32)
    hold_users = set(DEFAULT_HOLD_OUT_USERS)
    mask = np.array([int(u) in hold_users for u in users_all], dtype=bool)
    Xh, Xi, fh, yh = Xtr[mask], Ximu[mask], flag[mask], ytr[mask]
    print(f"holdout n={len(yh)}", flush=True)

    P, H_saved, y_oof, us_oof, y_h_saved = load_probs_maps()
    assert np.allclose(yh, y_h_saved)

    # verify saved holdout matches 5-fold recompute for one branch (sanity)
    print("sanity recompute kd_eq_a02 5-fold...", flush=True)
    p5 = ensemble_holdout(V18 / "checkpoints_kd_eq_a02", list(range(5)), Xh, Xi, fh, device, v2m)
    print("  max abs diff vs saved", float(np.max(np.abs(p5 - H_saved["kd_eq_a02"]))), flush=True)

    recipes = [
        {
            "name": "conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02",
            "keys": ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"],
            "family": "conf",
            "cfg": {"temp": 0.5},
            "nested_oof_full": 0.6652926628194559,
            "hold_full": 0.6435643564356436,
        },
        {
            "name": "pow_kd_mf_v13_kd_mf_a02_kd_eq_a02",
            "keys": ["kd_mf_v13", "kd_mf_a02", "kd_eq_a02"],
            "family": "pow",
            "cfg": {"p": 0.5},
            "nested_oof_full": 0.6714756801319044,
            "hold_full": 0.6415841584158416,
        },
        {
            "name": "eq_kd_mf_v13_kd_mf_a02_kd_eq_a02",
            "keys": ["kd_mf_v13", "kd_mf_a02", "kd_eq_a02"],
            "family": "eq",
            "cfg": {},
            "nested_oof_full": 0.6698268755152514,
            "hold_full": 0.6415841584158416,
        },
    ]

    mid = {"kd", "kd_mf_v13", "kd_mf_a02"}
    results = []

    # --- A: fold-subset on fp32 original ckpts ---
    fold_plans = [
        ("5fold", {k: list(range(5)) for k in ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]}),
        ("mid3_c5", {k: [0, 2, 4] for k in mid} | {"kd_eq_a02": list(range(5))}),
        ("mid3b_c5", {k: [1, 2, 3] for k in mid} | {"kd_eq_a02": list(range(5))}),
        ("mid4_c5", {k: [0, 1, 2, 3] for k in mid} | {"kd_eq_a02": list(range(5))}),
        ("mid4b_c5", {k: [0, 1, 3, 4] for k in mid} | {"kd_eq_a02": list(range(5))}),
        ("all3", {k: [0, 2, 4] for k in ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]}),
        ("all4", {k: [0, 1, 2, 3] for k in ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]}),
        ("share_mids_as_kd_mf", None),  # placeholder handled below
    ]

    # cache per (alias, folds_tuple) holdout probs
    cache = {}

    def get_hold(alias, folds, use_fp16_dir=None):
        key = (alias, tuple(folds), str(use_fp16_dir))
        if key not in cache:
            d = Path(use_fp16_dir) if use_fp16_dir else (V18 / f"checkpoints_{alias}")
            print(f"  infer {alias} folds={folds} dir={d.name}", flush=True)
            cache[key] = ensemble_holdout(d, folds, Xh, Xi, fh, device, v2m)
        return cache[key]

    print("\n=== Fold-subset fp32 ===", flush=True)
    for plan_name, fmap in fold_plans:
        if fmap is None:
            continue
        for rec in recipes:
            keys = rec["keys"]
            if any(k not in fmap for k in keys):
                continue
            size = 0.0
            H = {}
            for k in keys:
                folds = fmap[k]
                size += fold_size_mb(V18 / f"checkpoints_{k}", folds)
                H[k] = get_hold(k, folds)
            if rec["family"] == "conf":
                probs = apply_conf([H[k] for k in keys], rec["cfg"]["temp"])
            elif rec["family"] == "pow":
                probs = apply_power_mean([H[k] for k in keys], rec["cfg"]["p"])
            else:
                probs = apply_equal([H[k] for k in keys])
            hold = acc(probs.argmax(1), yh)
            # nested OOF still from full saved OOF (training artifact); note mismatch if folds subset
            oof_probs = nested_family(keys, P, y_oof, us_oof, rec["family"])
            oof = acc(oof_probs.argmax(1), y_oof)
            clear = (hold >= WIN_HOLD_ALONE or (hold >= WIN_HOLD_SOFT and oof >= WIN_OOF_SOFT)) and size <= BUDGET_MB + 1e-6
            row = {
                "plan": plan_name,
                "storage": "fp32",
                "recipe": rec["name"],
                "holdout": hold,
                "nested_oof_from_full_oof": oof,
                "size_mb": size,
                "clear_win": clear,
                "folds": {k: fmap[k] for k in keys},
            }
            results.append(row)
            print(
                f"{plan_name:12s} {rec['name'][:42]:42s} hold={hold:.4f} oof={oof:.4f} sz={size:.2f} clear={clear}",
                flush=True,
            )

    # --- B: fp16 pack all 5 folds for peek recipes ---
    print("\n=== FP16 pack ===", flush=True)
    fp16_root = ROOT / "ckpt_fp16"
    need_aliases = sorted({k for rec in recipes for k in rec["keys"]})
    fp16_sizes = {}
    for alias in need_aliases:
        sz = pack_fp16(V18 / f"checkpoints_{alias}", fp16_root / alias, list(range(5)))
        fp16_sizes[alias] = sz
        print(f"  packed {alias}: {sz:.2f} MB (5 folds)", flush=True)

    # clear cache for fp16
    for rec in recipes:
        keys = rec["keys"]
        size = sum(fp16_sizes[k] for k in keys)
        H = {}
        for k in keys:
            H[k] = get_hold(k, list(range(5)), use_fp16_dir=fp16_root / k)
        if rec["family"] == "conf":
            probs = apply_conf([H[k] for k in keys], rec["cfg"]["temp"])
        elif rec["family"] == "pow":
            probs = apply_power_mean([H[k] for k in keys], rec["cfg"]["p"])
        else:
            probs = apply_equal([H[k] for k in keys])
        hold = acc(probs.argmax(1), yh)
        oof_probs = nested_family(keys, P, y_oof, us_oof, rec["family"])
        oof = acc(oof_probs.argmax(1), y_oof)
        # compare to saved full holdout
        if rec["family"] == "conf":
            probs_ref = apply_conf([H_saved[k] for k in keys], rec["cfg"]["temp"])
        elif rec["family"] == "pow":
            probs_ref = apply_power_mean([H_saved[k] for k in keys], rec["cfg"]["p"])
        else:
            probs_ref = apply_equal([H_saved[k] for k in keys])
        hold_ref = acc(probs_ref.argmax(1), yh)
        n_diff = int((probs.argmax(1) != probs_ref.argmax(1)).sum())
        clear = (hold >= WIN_HOLD_ALONE or (hold >= WIN_HOLD_SOFT and oof >= WIN_OOF_SOFT)) and size <= BUDGET_MB + 1e-6
        row = {
            "plan": "fp16_5fold",
            "storage": "fp16",
            "recipe": rec["name"],
            "holdout": hold,
            "holdout_ref_fp32": hold_ref,
            "pred_diffs_vs_fp32": n_diff,
            "nested_oof_from_full_oof": oof,
            "size_mb": size,
            "clear_win": clear,
            "folds": {k: list(range(5)) for k in keys},
        }
        results.append(row)
        print(
            f"fp16_5fold   {rec['name'][:42]:42s} hold={hold:.4f} (ref={hold_ref:.4f} diffs={n_diff}) oof={oof:.4f} sz={size:.2f} clear={clear}",
            flush=True,
        )

    # --- C: fp16 + mid 3-fold if needed ---
    print("\n=== FP16 + mid3 ===", flush=True)
    for mid_folds in ([0, 2, 4], [1, 2, 3], [0, 1, 2]):
        for rec in recipes:
            keys = rec["keys"]
            fmap = {}
            size = 0.0
            H = {}
            for k in keys:
                folds = mid_folds if k in mid else list(range(5))
                fmap[k] = folds
                # size from fp16 files
                size += sum((fp16_root / k / f"best_fold{fi}.pt").stat().st_size for fi in folds) / (1024 * 1024)
                H[k] = get_hold(k, folds, use_fp16_dir=fp16_root / k)
            if rec["family"] == "conf":
                probs = apply_conf([H[k] for k in keys], rec["cfg"]["temp"])
            elif rec["family"] == "pow":
                probs = apply_power_mean([H[k] for k in keys], rec["cfg"]["p"])
            else:
                probs = apply_equal([H[k] for k in keys])
            hold = acc(probs.argmax(1), yh)
            oof_probs = nested_family(keys, P, y_oof, us_oof, rec["family"])
            oof = acc(oof_probs.argmax(1), y_oof)
            clear = (hold >= WIN_HOLD_ALONE or (hold >= WIN_HOLD_SOFT and oof >= WIN_OOF_SOFT)) and size <= BUDGET_MB + 1e-6
            row = {
                "plan": f"fp16_mid{''.join(map(str, mid_folds))}_c5",
                "storage": "fp16",
                "recipe": rec["name"],
                "holdout": hold,
                "nested_oof_from_full_oof": oof,
                "size_mb": size,
                "clear_win": clear,
                "folds": fmap,
            }
            results.append(row)
            print(
                f"fp16_mid{mid_folds} {rec['name'][:40]:40s} hold={hold:.4f} oof={oof:.4f} sz={size:.2f} clear={clear}",
                flush=True,
            )

    out = {
        "gates": {"hold_alone": WIN_HOLD_ALONE, "hold_soft": WIN_HOLD_SOFT, "oof_soft": WIN_OOF_SOFT, "budget_mb": BUDGET_MB},
        "results": results,
        "clear_wins": [r for r in results if r["clear_win"]],
        "elapsed_sec": time.time() - t0,
    }
    (ROOT / "compress_search.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nCLEAR WINS:", len(out["clear_wins"]), flush=True)
    for r in out["clear_wins"]:
        print(r, flush=True)
    print(f"wrote {ROOT / 'compress_search.json'} elapsed={out['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
