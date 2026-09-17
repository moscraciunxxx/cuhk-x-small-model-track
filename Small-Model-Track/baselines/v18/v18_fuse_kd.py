"""v18 KD-only nested fuse (memory-lean).

Clear-win: holdout >= 0.636 OR (holdout >= 0.631 AND nested OOF >= 0.670).
Selection = max nested OOF among multi-branch blends; holdout last.
No TTA / no leaky stackers.
"""
from __future__ import annotations

import json
import sys
import time
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "v7_stgcn"
V2 = ROOT.parent / "skeleton_imu_v2"
V10 = ROOT.parent / "v10"
V11 = ROOT.parent / "v11"
TRACK = ROOT.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2))  # dataset + caches

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    load_skel_train_cache,
    load_skel_test_cache,
)

NUM_CLASSES = 40
POWERS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
TEMPS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
V17_OOF, V17_HOLD = 0.6636, 0.6257
WIN_HOLD_ALONE = 0.636
WIN_HOLD_SOFT = 0.631
WIN_OOF_SOFT = 0.670

KD_KEYS = [
    "kd", "kd2", "kd_c", "kd_alt", "kd3", "kd_a02", "kd_mf_v13",
    "kd_a01", "kd_a015", "kd_a025", "kd_a02s7", "kd_a02s123",
    "kd_mf_a02", "kd_c_v15a02", "kd_eq_a02", "kd_T1_a02", "kd_T4_a02",
    "kd_rich_a02",
]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


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
        a = acc(apply_power_mean(probs_list, p).argmax(1), y)
        if best is None or a > best[0]:
            best = (a, float(p))
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
        a = acc(apply_conf(probs_list, t).argmax(1), y)
        if best is None or a > best[0]:
            best = (a, float(t))
    return best[1]


def nested_family(keys, P, yt, us, family):
    n = len(yt)
    out = np.zeros((n, NUM_CLASSES), np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), yt, us):
        plist_va = [P[k][va] for k in keys]
        if family == "eq":
            out[va] = apply_equal(plist_va)
        elif family == "pow":
            p = fit_power_mean([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_power_mean(plist_va, p)
        elif family == "conf":
            t = fit_conf([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_conf(plist_va, t)
        else:
            raise ValueError(family)
    return out


def hold_family(keys, P_nh, yt_nh, H, family):
    plist_h = [H[k] for k in keys]
    if family == "eq":
        return apply_equal(plist_h)
    if family == "pow":
        p = fit_power_mean([P_nh[k] for k in keys], yt_nh)
        return apply_power_mean(plist_h, p)
    if family == "conf":
        t = fit_conf([P_nh[k] for k in keys], yt_nh)
        return apply_conf(plist_h, t)
    raise ValueError(family)


def final_cfg(keys, P_nh, yt_nh, family):
    if family == "eq":
        return {}
    if family == "pow":
        return {"p": fit_power_mean([P_nh[k] for k in keys], yt_nh)}
    if family == "conf":
        return {"temp": fit_conf([P_nh[k] for k in keys], yt_nh)}
    raise ValueError(family)


def apply_family_cfg(keys, probs_map, family, cfg):
    plist = [probs_map[k] for k in keys]
    if family == "eq":
        return apply_equal(plist)
    if family == "pow":
        return apply_power_mean(plist, cfg["p"])
    if family == "conf":
        return apply_conf(plist, cfg["temp"])
    raise ValueError(family)


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


def load_v2_model_mod():
    spec = importlib.util.spec_from_file_location("v2_model_v18", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(ckpt_path, device, v2m):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ck.get("model_name", "compact")
    ncls = ck.get("num_classes", NUM_CLASSES)
    if name in ("compact", "compact_fuse"):
        m = v2m.CompactMidFuse(num_classes=ncls)
    else:
        m = v2m.build_model("midfuse", num_classes=ncls)
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m


def load_branch(alias, root):
    oof_path = root / f"oof_{alias}.npz"
    hold_path = root / f"holdout_{alias}.npz"
    if not oof_path.exists() or not hold_path.exists():
        return None
    od = np.load(oof_path)
    hd = np.load(hold_path)
    okey = alias if alias in od.files else next(k for k in od.files if k not in ("y", "users"))
    hkey = alias if alias in hd.files else next(k for k in hd.files if k not in ("y", "users"))
    return od[okey], hd[hkey]


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    X_skel, y, users, _ = load_skel_train_cache(V2 / "cache")
    y = np.asarray(y)
    users = np.asarray(users)
    hold_set = set(DEFAULT_HOLD_OUT_USERS)
    nh_mask = np.array([int(u) not in hold_set for u in users])
    nh_idx = np.where(nh_mask)[0]
    yt_nh, users_nh = y[nh_idx], users[nh_idx]
    # holdout labels from any holdout file
    yt_h = np.load(ROOT / "holdout_kd.npz")["y"]

    L_full, H_full = {}, {}
    for alias in KD_KEYS:
        loaded = load_branch(alias, ROOT)
        if loaded is None:
            continue
        oof_l, hold_l = loaded
        L_full[alias] = oof_l
        H_full[alias] = hold_l
        print(f"loaded {alias} oof={oof_l.shape} hold={hold_l.shape}", flush=True)

    P_nh = {k: softmax(L_full[k][nh_idx]) for k in L_full}
    H = {k: softmax(H_full[k]) for k in H_full}
    kd_all = list(L_full.keys())
    solo_oof = {k: acc(P_nh[k].argmax(1), yt_nh) for k in kd_all}
    print("solos:", sorted(((k, round(v, 4)) for k, v in solo_oof.items()), key=lambda x: -x[1]), flush=True)

    anchors = [k for k in ("kd", "kd_c", "kd_a02", "kd_alt", "kd_mf_v13", "kd3", "kd_mf_a02", "kd_T4_a02", "kd_eq_a02", "kd_a015") if k in kd_all]
    ranked = sorted(kd_all, key=lambda k: -solo_oof[k])
    pool = []
    for k in anchors + ranked:
        if k not in pool:
            pool.append(k)
        if len(pool) >= 9:
            break
    print(f"exhaustive pool ({len(pool)}): {[(k, round(solo_oof[k],4)) for k in pool]}", flush=True)

    methods = {}
    # solos (ablation / peek only)
    for k in kd_all:
        o = acc(P_nh[k].argmax(1), yt_nh)
        h = acc(H[k].argmax(1), yt_h)
        methods[f"solo_{k}"] = {
            "oof_acc_nested": o,
            "holdout_acc": h,
            "family": "eq",
            "params": {"keys": [k]},
            "keys": [k],
        }

    max_r = min(5, len(pool))
    combos = []
    for r in range(2, max_r + 1):
        combos.extend(combinations(pool, r))
    combos.append(tuple(pool))
    print(f"n_combos={len(combos)} x3 families", flush=True)

    for i, combo in enumerate(combos):
        keys = list(combo)
        tag = "_".join(keys)
        for fam in ("eq", "pow", "conf"):
            name = f"{fam}_{tag}"
            nested = nested_family(keys, P_nh, yt_nh, users_nh, fam)
            o = acc(nested.argmax(1), yt_nh)
            hold_p = hold_family(keys, P_nh, yt_nh, H, fam)
            h = acc(hold_p.argmax(1), yt_h)
            cfg = final_cfg(keys, P_nh, yt_nh, fam)
            methods[name] = {
                "oof_acc_nested": o,
                "holdout_acc": h,
                "family": fam,
                "params": {"keys": keys, **cfg},
                "keys": keys,
            }
            del nested, hold_p
        if (i + 1) % 50 == 0 or i == len(combos) - 1:
            print(f"progress {i+1}/{len(combos)}", flush=True)

    blend = {k: v for k, v in methods.items() if not k.startswith("solo_")}
    best_name = max(blend, key=lambda k: blend[k]["oof_acc_nested"])
    oof_best = methods[best_name]["oof_acc_nested"]
    hold_best = methods[best_name]["holdout_acc"]
    clear_win = bool(hold_best >= WIN_HOLD_ALONE or (hold_best >= WIN_HOLD_SOFT and oof_best >= WIN_OOF_SOFT))
    peek_best = max(methods, key=lambda k: methods[k]["holdout_acc"])
    eligible = {k: v for k, v in methods.items() if v["oof_acc_nested"] >= WIN_OOF_SOFT}
    print(f"SELECTED={best_name} nested={oof_best:.4f} hold={hold_best:.4f} clear={clear_win}", flush=True)
    print(
        f"peek hold={methods[peek_best]['holdout_acc']:.4f} {peek_best} oof={methods[peek_best]['oof_acc_nested']:.4f}",
        flush=True,
    )
    if eligible:
        bh = max(eligible, key=lambda k: eligible[k]["holdout_acc"])
        print(
            f"best hold OOF>={WIN_OOF_SOFT}: {bh} hold={eligible[bh]['holdout_acc']:.4f} oof={eligible[bh]['oof_acc_nested']:.4f}",
            flush=True,
        )

    # Test inference for selected keys only
    print("building test logits for", methods[best_name]["keys"], flush=True)
    v2m = load_v2_model_mod()
    Xte, te_ids = load_skel_test_cache(V2 / "cache")
    imu_te = np.load(V2 / "cache" / "imu_test.npz")
    Xte_imu = imu_te["X"]
    te_flag = imu_te["has_imu"].astype(np.float32)

    ckpt_dirs = {
        "kd": ROOT / "checkpoints_kd",
        "kd2": ROOT / "checkpoints_kd2",
        "kd_c": ROOT / "checkpoints_kd_c",
        "kd_alt": ROOT / "checkpoints_kd_alt",
        "kd3": ROOT / "checkpoints_kd3",
        "kd_a02": ROOT / "checkpoints_kd_a02",
        "kd_mf_v13": ROOT / "checkpoints_kd_mf_v13",
        "kd_a01": ROOT / "checkpoints_kd_a01",
        "kd_a015": ROOT / "checkpoints_kd_a015",
        "kd_a025": ROOT / "checkpoints_kd_a025",
        "kd_a02s7": ROOT / "checkpoints_kd_a02s7",
        "kd_a02s123": ROOT / "checkpoints_kd_a02s123",
        "kd_mf_a02": ROOT / "checkpoints_kd_mf_a02",
        "kd_c_v15a02": ROOT / "checkpoints_kd_c_v15a02",
        "kd_eq_a02": ROOT / "checkpoints_kd_eq_a02",
        "kd_T1_a02": ROOT / "checkpoints_kd_T1_a02",
        "kd_T4_a02": ROOT / "checkpoints_kd_T4_a02",
        "kd_rich_a02": ROOT / "checkpoints_kd_rich_a02",
    }

    test_probs_map = {}
    for name in methods[best_name]["keys"]:
        fold_dir = ckpt_dirs[name]
        accums = []
        for fi in range(5):
            ckpt = fold_dir / f"best_fold{fi}.pt"
            m = load_ckpt(ckpt, device, v2m)
            accums.append(predict_logits_arr(m, Xte, Xte_imu, te_flag, device))
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        test_probs_map[name] = softmax(sum(accums) / float(len(accums)))
        print(f"test {name} folds={len(accums)}", flush=True)

    sel = methods[best_name]
    cfg = {k: v for k, v in sel["params"].items() if k != "keys"}
    test_probs = apply_family_cfg(sel["keys"], test_probs_map, sel["family"], cfg)
    pred = test_probs.argmax(1).astype(int)

    sub_v11 = pd.read_csv(V11 / "submission_v11.csv")
    sub = sub_v11.copy()
    sub.iloc[:, 1] = pred[: len(sub)]
    sub.to_csv(ROOT / "submission_v18_candidate.csv", index=False)
    sub.to_csv(ROOT / "submission_v18.csv", index=False)
    np.savez_compressed(ROOT / "submission_v18_probs.npz", probs=test_probs, pred=pred)

    overwrite = False
    ping = False
    if clear_win:
        sub.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submission_README.txt").write_text(
            (
                f"v18 clear-win: nested OOF {oof_best:.4f} holdout {hold_best:.4f} "
                f"(gate hold>=0.636, or hold>=0.631 & OOF>=0.670)\n"
                f"method={best_name} params={json.dumps(sel['params'])}\n"
                f"ping_disk_saver=true.\n"
            ),
            encoding="utf-8",
        )
        overwrite = True
        ping = True
        print("OVERWROTE track submission.csv", flush=True)
    else:
        print("NO overwrite - track submission remains v17", flush=True)

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
        "v17_nested_oof": V17_OOF,
        "v17_holdout": V17_HOLD,
        "clear_win": clear_win,
        "overwrite_submission": overwrite,
        "ping_disk_saver": ping,
        "kd_pool": pool,
        "n_methods": len(methods),
        "peek_best_hold": {
            "name": peek_best,
            "holdout": methods[peek_best]["holdout_acc"],
            "nested_oof": methods[peek_best]["oof_acc_nested"],
        },
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
