"""v6c: cluster specialists — defer when MidFuse top1 in confuse cluster (not only top-2 pair)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
ROOT_V2 = ROOT / "baselines" / "skeleton_imu_v2"
OUT = ROOT / "baselines" / "v6"
sys.path.insert(0, str(ROOT_V2))
sys.path.insert(0, str(OUT))

from dataset import DEFAULT_HOLD_OUT_USERS, load_skel_train_cache  # noqa: E402
from refine_v6 import DualSubset, RawPairNet, predict_raw_proba, train_raw_pair  # noqa: E402
from specialists_v6 import extract_all, load_midfuse, make_loader, softmax_np, top2_margin  # noqa: E402
from final_v6 import train_raw_pair_inner  # noqa: E402

BASELINE = 0.537
CLUSTERS = [
    ("kitchen", (8, 9, 10, 11)),
    ("exercise", (29, 30, 31, 32, 34, 35)),
    ("drink_med", (6, 7, 37)),
    ("read_write", (17, 18, 21, 22)),
    ("device", (19, 24, 25, 26, 27)),
    ("clean", (12, 13, 14, 15)),
]


def fit_cluster(X_skel, X_imu, y, has_imu, users, tr_idx, classes, device, seed=0):
    classes = tuple(sorted(classes))
    model, g2l, l2g, va = train_raw_pair_inner(
        X_skel, X_imu, y, has_imu, users, tr_idx, classes, device,
        epochs=40, patience=10, seed=seed,
    )
    # train_raw_pair_inner expects pair-like but works for any classes tuple via train_raw_pair
    return model, g2l, l2g, va, classes


# Fix: train_raw_pair_inner uses pair_key logic - check it works for >2 classes
# Looking at train_raw_pair_inner - it passes classes to train_raw_pair which handles n_out=len(classes)
# But RawPairNet(n_out=len(classes)) - good.
# save uses pair_key for 2-class - fine for multi.


def apply_cluster_defer(logits, y, global_idx, X_skel, X_imu, has_imu, clusters, device,
                        margin_thr=0.3, conf_thr=0.55, require_top2_in=False):
    top1, top2, margin, probs = top2_margin(logits)
    pred = top1.copy()
    n_defer = n_flip = 0
    for i in range(len(pred)):
        t1 = int(top1[i])
        t2 = int(top2[i])
        if margin[i] >= margin_thr:
            continue
        for name, model, g2l, l2g, classes in clusters:
            cs = set(classes)
            if t1 not in cs:
                continue
            if require_top2_in and t2 not in cs:
                continue
            proba = predict_raw_proba(
                model, X_skel, X_imu, has_imu, np.array([int(global_idx[i])]), device
            )[0]
            loc = int(proba.argmax())
            if float(proba[loc]) < conf_thr:
                continue
            newp = l2g[loc]
            n_defer += 1
            if newp != pred[i]:
                n_flip += 1
            pred[i] = newp
            break
    st = {
        "acc": float((pred == y).mean()),
        "base_acc": float((top1 == y).mean()),
        "n_defer": n_defer,
        "n_flip": n_flip,
        "margin_thr": margin_thr,
        "conf_thr": conf_thr,
        "require_top2_in": require_top2_in,
    }
    return pred, st


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    cache = ROOT_V2 / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    y = np.asarray(y, np.int64)
    users = np.asarray(users, np.int64)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]

    # MidFuse OOF for selection on non-holdout
    fold_dir = ROOT_V2 / "checkpoints_midfuse_v2b"
    oof_logits = np.zeros((len(y), 40), np.float32)
    gkf = GroupKFold(n_splits=5)
    for fi, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, users)):
        mid, _ = load_midfuse(fold_dir / f"best_fold{fi}.pt", device)
        pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, va, has_imu), device)
        oof_logits[va] = pack["logits"]
        del mid
        torch.cuda.empty_cache()
    print("OOF ready", float((oof_logits.argmax(1) == y).mean()), flush=True)

    # Nested select on non-holdout with OOF logits
    y_tr, users_tr = y[tr_idx], users[tr_idx]
    all_st = []
    for fi, (a, b) in enumerate(GroupKFold(5).split(np.zeros(len(tr_idx)), y_tr, users_tr)):
        tr_f, va_f = tr_idx[a], tr_idx[b]
        logits_va, y_va = oof_logits[va_f], y[va_f]
        base = float((logits_va.argmax(1) == y_va).mean())
        clust = []
        for ci, (name, classes) in enumerate(CLUSTERS):
            model, g2l, l2g, va_acc = train_raw_pair_inner(
                X_skel, X_imu, y, has_imu, users, tr_f, classes, device,
                epochs=28, patience=7, seed=4000 + fi * 20 + ci,
            )
            if model is None:
                continue
            clust.append((name, model, g2l, l2g, tuple(sorted(classes))))
            print(f"  fold{fi} {name} n_cls={len(classes)} inner_va={va_acc:.3f}", flush=True)
        for mthr in [0.15, 0.25, 0.4, 0.6, 1.01]:
            for cthr in [0.5, 0.55, 0.65, 0.75]:
                for req2 in [False, True]:
                    _, st = apply_cluster_defer(
                        logits_va, y_va, va_f, X_skel, X_imu, has_imu, clust, device,
                        margin_thr=mthr, conf_thr=cthr, require_top2_in=req2,
                    )
                    st["policy"] = f"m={mthr}|c={cthr}|top2={req2}"
                    st["fold"] = fi
                    st["base"] = base
                    all_st.append(st)
        print(f"fold{fi} base={base:.4f}", flush=True)

    from collections import defaultdict
    by = defaultdict(list)
    for st in all_st:
        by[st["policy"]].append(st["acc"] - st["base"])
    mean_d = {k: float(np.mean(v)) for k, v in by.items()}
    best = max(mean_d, key=mean_d.get)
    print("TOP cluster policies:", flush=True)
    for k, v in sorted(mean_d.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k}: {v:+.4f}", flush=True)

    parts = dict(x.split("=") for x in best.split("|"))
    mthr, cthr, req2 = float(parts["m"]), float(parts["c"]), parts["top2"] == "True"

    # Holdout
    mid_h, _ = load_midfuse(fold_dir / "best_holdout.pt", device)
    va_pack = extract_all(mid_h, make_loader(X_skel, X_imu, y, users, va_idx, has_imu), device)
    base_h = float((va_pack["logits"].argmax(1) == va_pack["y"]).mean())
    clust_h = []
    for ci, (name, classes) in enumerate(CLUSTERS):
        model, g2l, l2g, va_acc = train_raw_pair_inner(
            X_skel, X_imu, y, has_imu, users, tr_idx, classes, device,
            epochs=40, patience=10, seed=42 + ci,
        )
        if model is None:
            continue
        clust_h.append((name, model, g2l, l2g, tuple(sorted(classes))))
        print(f"hold-train {name} inner_va={va_acc:.3f}", flush=True)

    _, st = apply_cluster_defer(
        va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, clust_h, device,
        margin_thr=mthr, conf_thr=cthr, require_top2_in=req2,
    )
    print(f"PRIMARY cluster holdout={st['acc']:.4f} base={base_h:.4f} d537={st['acc']-BASELINE:+.4f} policy={best}", flush=True)

    # sensitivity
    sens = []
    for m2 in [0.15, 0.25, 0.4, 0.6, 1.01]:
        for c2 in [0.5, 0.55, 0.65, 0.75]:
            for r2 in [False, True]:
                _, s2 = apply_cluster_defer(
                    va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, clust_h, device,
                    margin_thr=m2, conf_thr=c2, require_top2_in=r2,
                )
                s2["policy"] = f"m={m2}|c={c2}|top2={r2}"
                sens.append(s2)
    sens.sort(key=lambda d: -d["acc"])
    print("Oracle sensitivity top:", flush=True)
    for s in sens[:10]:
        print(f"  {s['policy']} acc={s['acc']:.4f} d={s['acc']-base_h:+.4f} defer={s['n_defer']}", flush=True)

    primary_acc = st["acc"] if st["acc"] >= base_h else base_h
    used = st if st["acc"] >= base_h else {"acc": base_h, "policy": "baseline", "delta_vs_0.537": base_h - BASELINE}
    if st["acc"] < base_h:
        print("Cluster policy hurts; fall back to baseline", flush=True)

    out = {
        "baseline_holdout": base_h,
        "nested_best_policy": best,
        "nested_mean_deltas_top": dict(sorted(mean_d.items(), key=lambda x: -x[1])[:15]),
        "holdout_nested_policy": st,
        "primary_acc": primary_acc,
        "delta_vs_0.537": primary_acc - BASELINE,
        "oracle_best": sens[0],
        "clear_win": bool(primary_acc >= BASELINE + 0.01),
        "ping_disk_saver": bool(primary_acc >= BASELINE + 0.01),
    }
    with open(OUT / "v6c_cluster.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ["primary_acc", "delta_vs_0.537", "clear_win", "ping_disk_saver", "nested_best_policy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
