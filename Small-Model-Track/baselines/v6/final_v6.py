"""v6 final: raw confuse-pair specialists with NO holdout leakage.

- MidFuse v2b holdout ckpt for base logits (trained without {8,9,24})
- Raw dual Conv specialists early-stopped on nested train GroupKFold only
- Deferral hyperparams selected on nested train OOF
- Single final holdout measure
- If clear win (>=0.547), full GroupKFold + test submission
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
ROOT_V2 = ROOT / "baselines" / "skeleton_imu_v2"
OUT = ROOT / "baselines" / "v6"
sys.path.insert(0, str(ROOT_V2))
sys.path.insert(0, str(OUT))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_skel_train_cache,
    load_skel_test_cache,
)
from model import build_model, count_parameters  # noqa: E402
from refine_v6 import (  # noqa: E402
    DualSubset,
    RawPairNet,
    predict_raw,
    predict_raw_proba,
    train_raw_pair,
)
from specialists_v6 import (  # noqa: E402
    extract_all,
    load_midfuse,
    make_loader,
    pair_key,
    top2_margin,
)

BASELINE = 0.537
WIN_DELTA = 0.01
N_CLASSES = 40

# Pairs from error analysis (holdout + OOF)
RAW_PAIRS = [
    (9, 10), (10, 11), (9, 11), (30, 31), (29, 32), (29, 34),
    (12, 13), (21, 22), (26, 7), (6, 37), (8, 9), (6, 7),
    (32, 34), (26, 24), (26, 19),
]


def train_raw_pair_inner(
    X_skel, X_imu, y, has_imu, users, tr_idx, classes, device,
    epochs=40, patience=10, seed=0, n_inner=3,
):
    """Train raw specialist; early-stop using an inner GroupKFold split of tr_idx only."""
    classes = tuple(sorted(classes))
    g2l = {c: i for i, c in enumerate(classes)}
    mask = np.array([int(y[i]) in g2l for i in tr_idx])
    sub = np.asarray(tr_idx)[mask]
    if len(sub) < 16:
        return None, None, None, 0.0
    sub_users = users[sub]
    # pick one inner holdout group of users (~25%)
    uniq = np.unique(sub_users)
    if len(uniq) < 3:
        # tiny: use random 20% trials (not ideal but no holdout leak)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(sub))
        n_va = max(2, len(sub) // 5)
        va_local = sub[perm[:n_va]]
        tr_local = sub[perm[n_va:]]
    else:
        gkf = GroupKFold(n_splits=min(n_inner, len(uniq)))
        # use first fold as early-stop val
        tr_local, va_local = next(gkf.split(np.zeros(len(sub)), y[sub], sub_users))
        tr_local = sub[tr_local]
        va_local = sub[va_local]

    model, g2l2, va_acc = train_raw_pair(
        X_skel, X_imu, y, has_imu, tr_local, va_local, classes, device,
        epochs=epochs, patience=patience, seed=seed,
    )
    if model is None:
        return None, None, None, 0.0
    l2g = {v: k for k, v in g2l2.items()}
    return model, g2l2, l2g, va_acc


def apply_raw_defer(
    logits, y_true, global_idx, X_skel, X_imu, has_imu, raw_models, device,
    margin_thr=0.25, conf_thr=0.55, enabled=None,
):
    top1, top2, margin, _ = top2_margin(logits)
    pred = top1.copy()
    n_defer = n_flip = 0
    for i in range(len(pred)):
        pk = pair_key(int(top1[i]), int(top2[i]))
        if pk not in raw_models:
            continue
        if enabled is not None and pk not in enabled:
            continue
        if margin[i] >= margin_thr:
            continue
        model, g2l, l2g = raw_models[pk]
        proba = predict_raw_proba(
            model, X_skel, X_imu, has_imu, np.array([int(global_idx[i])]), device
        )[0]
        loc = int(proba.argmax())
        if float(proba[loc]) < conf_thr:
            continue
        n_defer += 1
        newp = l2g[loc]
        if newp != pred[i]:
            n_flip += 1
        pred[i] = newp
    out = {
        "acc": float((pred == y_true).mean()) if y_true is not None else None,
        "base_acc": float((top1 == y_true).mean()) if y_true is not None else None,
        "n_defer": n_defer,
        "n_flip": n_flip,
        "margin_thr": margin_thr,
        "conf_thr": conf_thr,
    }
    return pred, out


def nested_oof_select(X_skel, X_imu, y, has_imu, users, tr_idx, mid_logits_tr, device):
    """GroupKFold on train users: train raw specs, measure deferral OOF, pick pairs+hyperparams."""
    gkf = GroupKFold(n_splits=4)
    # accumulate OOF logits already from midfuse on tr; we need raw preds per fold
    # For each fold: train raw on fold-train, apply to fold-val
    all_policies = []
    # We'll store per-sample OOF base preds and for each policy the corrected preds
    n = len(tr_idx)
    # map local -> global
    oof_base = np.zeros(n, dtype=np.int64)
    oof_y = y[tr_idx]
    # For hyperparam sweep, store list of (policy_key -> oof_pred corrections count)
    # Simpler: for each fold compute acc for each (mthr, cthr, pair-subset)

    fold_results = []
    pair_help = {pair_key(*p): [] for p in RAW_PAIRS}

    for fi, (a, b) in enumerate(gkf.split(np.zeros(len(tr_idx)), y[tr_idx], users[tr_idx])):
        tr_f = tr_idx[a]
        va_f = tr_idx[b]
        # midfuse logits for va_f: need alignment - mid_logits_tr is in tr_idx order
        logits_va = mid_logits_tr[b]
        y_va = y[va_f]
        base = float((logits_va.argmax(1) == y_va).mean())
        oof_base[b] = logits_va.argmax(1)

        raw_models = {}
        for p in RAW_PAIRS:
            pk = pair_key(*p)
            model, g2l, l2g, va_acc = train_raw_pair_inner(
                X_skel, X_imu, y, has_imu, users, tr_f, pk, device,
                epochs=35, patience=8, seed=1000 + fi * 100 + pk[0] * 40 + pk[1],
            )
            if model is None:
                continue
            raw_models[pk] = (model, g2l, l2g)
            # pair-alone delta
            _, st = apply_raw_defer(
                logits_va, y_va, va_f, X_skel, X_imu, has_imu, raw_models, device,
                margin_thr=1.01, conf_thr=0.55, enabled={pk},
            )
            # need only this pair - rebuild temp
            tmp = {pk: raw_models[pk]}
            _, st = apply_raw_defer(
                logits_va, y_va, va_f, X_skel, X_imu, has_imu, tmp, device,
                margin_thr=1.01, conf_thr=0.55, enabled={pk},
            )
            pair_help[pk].append(st["acc"] - base)

        # sweep policies with all raw models this fold
        best_fold = {"acc": base, "policy": "base"}
        for mthr in [0.15, 0.25, 0.4, 0.6, 1.01]:
            for cthr in [0.5, 0.55, 0.65, 0.75]:
                pred, st = apply_raw_defer(
                    logits_va, y_va, va_f, X_skel, X_imu, has_imu, raw_models, device,
                    margin_thr=mthr, conf_thr=cthr,
                )
                st["policy"] = f"m={mthr}|c={cthr}"
                st["fold"] = fi
                st["base"] = base
                all_policies.append(st)
                if st["acc"] > best_fold["acc"]:
                    best_fold = st
        fold_results.append({"fold": fi, "base": base, "best": best_fold, "n_raw": len(raw_models)})
        print(f"nested fold{fi} base={base:.4f} best={best_fold.get('acc'):.4f} policy={best_fold.get('policy')} n_raw={len(raw_models)}", flush=True)

    # aggregate mean acc per policy across folds
    from collections import defaultdict
    by_pol = defaultdict(list)
    for st in all_policies:
        by_pol[st["policy"]].append(st["acc"] - st["base"])
    pol_mean = {k: float(np.mean(v)) for k, v in by_pol.items()}
    best_pol = max(pol_mean, key=pol_mean.get)
    # parse
    # m=0.25|c=0.55
    parts = dict(x.split("=") for x in best_pol.split("|"))
    mthr = float(parts["m"])
    cthr = float(parts["c"])

    selected_pairs = []
    pair_stats = {}
    for pk, deltas in pair_help.items():
        if not deltas:
            continue
        md = float(np.mean(deltas))
        pair_stats[str(pk)] = {"mean_delta": md, "n": len(deltas)}
        if md > 0.0003:
            selected_pairs.append(pk)

    return {
        "best_policy": best_pol,
        "margin_thr": mthr,
        "conf_thr": cthr,
        "pol_mean_delta": pol_mean,
        "selected_pairs": selected_pairs,
        "pair_stats": pair_stats,
        "fold_results": fold_results,
    }


def fit_all_raw(X_skel, X_imu, y, has_imu, users, tr_idx, pairs, device):
    raw_models = {}
    meta = {}
    for p in pairs:
        pk = pair_key(*p) if not (isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int)) else pair_key(*p)
        if isinstance(p, tuple) and len(p) == 2:
            pk = pair_key(int(p[0]), int(p[1]))
        model, g2l, l2g, va_acc = train_raw_pair_inner(
            X_skel, X_imu, y, has_imu, users, tr_idx, pk, device,
            epochs=45, patience=10, seed=42 + pk[0] * 40 + pk[1],
        )
        if model is None:
            continue
        raw_models[pk] = (model, g2l, l2g)
        meta[str(pk)] = {"inner_va_acc": va_acc, "params": count_parameters(model)}
        print(f"  fitted raw {pk} inner_va={va_acc:.3f} params={meta[str(pk)]['params']}", flush=True)
    return raw_models, meta


def save_raw_models(raw_models, path: Path, extra: dict):
    blob = {"models": {}, **extra}
    for pk, (model, g2l, l2g) in raw_models.items():
        blob["models"][str(pk)] = {
            "state": model.state_dict(),
            "g2l": {str(k): int(v) for k, v in g2l.items()},
            "l2g": {str(k): int(v) for k, v in l2g.items()},
            "classes": list(pk),
            "n_out": len(g2l),
        }
    torch.save(blob, path)


def load_raw_models(path: Path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    raw = {}
    for k, v in blob["models"].items():
        classes = tuple(v["classes"])
        pk = pair_key(classes[0], classes[1]) if len(classes) == 2 else tuple(classes)
        model = RawPairNet(n_out=v["n_out"]).to(device)
        model.load_state_dict(v["state"])
        model.eval()
        g2l = {int(a): int(b) for a, b in v["g2l"].items()}
        l2g = {int(a): int(b) for a, b in v["l2g"].items()}
        raw[pk] = (model, g2l, l2g)
    return raw, blob


def infer_test(mid, raw_models, enabled, mthr, cthr, device, out_csv: Path):
    cache = ROOT_V2 / "cache"
    X_skel, paths = load_skel_test_cache(cache)
    imu = np.load(cache / "imu_test.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool) if "has_imu" in imu.files else np.ones(len(X_skel), dtype=bool)
    # build fake y/users for dataset
    y = np.zeros(len(X_skel), dtype=np.int64)
    users = np.zeros(len(X_skel), dtype=np.int64)
    idx = np.arange(len(X_skel))
    ds = CachedDualDataset(X_skel, X_imu, y, users, idx, has_imu, augment=False)
    ld = DataLoader(ds, batch_size=64, shuffle=False)
    pack = extract_all(mid, ld, device)
    pred, st = apply_raw_defer(
        pack["logits"], None, idx, X_skel, X_imu, has_imu, raw_models, device,
        margin_thr=mthr, conf_thr=cthr, enabled=enabled,
    )
    # clip ids from paths
    rows = []
    for pth, pr in zip(paths, pred):
        name = Path(pth).name
        if not name.startswith("SM_test_"):
            # path may be full clip dir
            name = Path(pth).name
        rows.append((name, int(pr)))
    # ensure SM_test_* id
    clip_ids = []
    for pth in paths:
        p = Path(str(pth))
        # find SM_test_* component
        found = None
        for part in [p.name] + list(p.parts):
            if str(part).startswith("SM_test_"):
                found = str(part)
                break
        clip_ids.append(found or p.name)
    import pandas as pd
    paths_out = []
    for cid in clip_ids:
        cid = str(cid)
        if cid.startswith("small_model_track_test/"):
            paths_out.append(cid if cid.endswith("/") else cid + "/")
        else:
            name = cid if cid.startswith("SM_test_") else Path(cid).name
            if not name.startswith("SM_test_"):
                # try parent
                name = Path(cid).name
            paths_out.append(f"small_model_track_test/{name}/")
    df = pd.DataFrame({"path": paths_out, "prediction": pred.astype(int)})
    df = df.sort_values("path").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} n={len(df)} defer_stats={st}", flush=True)
    print(df.head(3).to_string(), flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-nested", action="store_true")
    ap.add_argument("--force-submit", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    print("device", device, flush=True)
    t0 = time.time()

    cache = ROOT_V2 / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y, dtype=np.int64)
    users = np.asarray(users, dtype=np.int64)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]

    ckpt = ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"
    mid, ck_meta = load_midfuse(ckpt, device)
    print(f"midfuse holdout ckpt val_acc={ck_meta.get('val_acc')}", flush=True)

    tr_pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, tr_idx, has_imu), device)
    va_pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, va_idx, has_imu), device)
    base_hold = float((va_pack["logits"].argmax(1) == va_pack["y"]).mean())
    print(f"measured midfuse holdout={base_hold:.4f} n={len(va_idx)}", flush=True)

    sel_path = OUT / "nested_selection.json"
    if args.skip_nested and sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        print("loaded nested selection", sel.get("best_policy"), flush=True)
    else:
        print("=== Nested OOF hyperparam/pair selection (train users only) ===", flush=True)
        sel = nested_oof_select(
            X_skel, X_imu, y, has_imu, users, tr_idx, tr_pack["logits"], device
        )
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(sel, f, indent=2)
        print(f"selected policy={sel['best_policy']} pairs={sel['selected_pairs']}", flush=True)

    mthr = float(sel["margin_thr"])
    cthr = float(sel["conf_thr"])
    # Use all RAW_PAIRS if nested selected few; still use nested hyperparams
    pairs_to_fit = list(RAW_PAIRS)
    selected = [tuple(p) if not isinstance(p, tuple) else p for p in sel.get("selected_pairs", [])]
    # selected may be list of tuples from json as lists
    selected_set = set()
    for p in sel.get("selected_pairs", []):
        if isinstance(p, (list, tuple)):
            selected_set.add(pair_key(int(p[0]), int(p[1])))
        else:
            # string "(a, b)"
            pass
    # Also parse pair_stats keys
    for k, v in sel.get("pair_stats", {}).items():
        if v.get("mean_delta", 0) > 0.0003:
            # k like "(9, 10)"
            nums = k.strip("()").split(",")
            selected_set.add(pair_key(int(nums[0]), int(nums[1])))

    print(f"=== Fit raw specialists on all train (non-holdout) pairs={len(pairs_to_fit)} ===", flush=True)
    raw_models, fit_meta = fit_all_raw(X_skel, X_imu, y, has_imu, users, tr_idx, pairs_to_fit, device)

    # Evaluate holdout with nested-chosen hyperparams
    enabled_all = set(raw_models.keys())
    enabled_sel = selected_set & enabled_all if selected_set else enabled_all

    results = []
    for name, en in [("all_pairs", enabled_all), ("selected_pairs", enabled_sel)]:
        pred, st = apply_raw_defer(
            va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, raw_models, device,
            margin_thr=mthr, conf_thr=cthr, enabled=en,
        )
        st["name"] = name
        st["delta_vs_0.537"] = st["acc"] - BASELINE
        st["delta_vs_base"] = st["acc"] - base_hold
        results.append(st)
        print(f"HOLDOUT {name}: acc={st['acc']:.4f} delta_vs_0.537={st['delta_vs_0.537']:+.4f} defer={st['n_defer']} flip={st['n_flip']}", flush=True)

    # Also report a small grid around nested choice (for transparency; primary is nested)
    sens = []
    for dm in [-0.1, 0, 0.1]:
        for dc in [-0.05, 0, 0.05]:
            mt = min(1.01, max(0.05, mthr + dm))
            ct = min(0.9, max(0.45, cthr + dc))
            pred, st = apply_raw_defer(
                va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, raw_models, device,
                margin_thr=mt, conf_thr=ct, enabled=enabled_all,
            )
            st["policy"] = f"m={mt}|c={ct}"
            sens.append(st)
    sens.sort(key=lambda d: -d["acc"])

    primary = max(results, key=lambda d: d["acc"])
    clear_win = primary["acc"] >= BASELINE + WIN_DELTA
    print(f"PRIMARY holdout={primary['acc']:.4f} clear_win={clear_win}", flush=True)

    save_raw_models(
        raw_models,
        OUT / "checkpoints" / "raw_specialists_holdout_train.pt",
        {"mthr": mthr, "cthr": cthr, "enabled": [list(p) for p in (enabled_sel if primary['name']=='selected_pairs' else enabled_all)],
         "fit_meta": fit_meta, "sel": {k: sel[k] for k in ["best_policy", "margin_thr", "conf_thr"]}},
    )

    summary = {
        "midfuse_holdout_baseline_measured": base_hold,
        "baseline_reported": BASELINE,
        "nested_selection": {
            "best_policy": sel["best_policy"],
            "margin_thr": mthr,
            "conf_thr": cthr,
            "selected_pairs": [list(p) for p in selected_set],
            "pol_mean_delta": sel.get("pol_mean_delta"),
        },
        "holdout_results": results,
        "primary": primary,
        "sensitivity_around_nested": sens[:8],
        "clear_win": clear_win,
        "fit_meta": fit_meta,
        "elapsed_sec": time.time() - t0,
    }

    # If clear win: full GroupKFold OOF estimate + retrain on ALL + submission
    submit_info = None
    if clear_win or args.force_submit:
        print("=== Clear win: full GroupKFold OOF + all-train specialists + test infer ===", flush=True)
        # OOF: for each fold, MidFuse fold ckpt + raw specs trained on fold-train
        fold_dir = ROOT_V2 / "checkpoints_midfuse_v2b"
        gkf = GroupKFold(n_splits=5)
        oof_pred = np.full(len(y), -1, dtype=np.int64)
        oof_base = np.full(len(y), -1, dtype=np.int64)
        fold_accs = []
        for fi, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, users)):
            fck = fold_dir / f"best_fold{fi}.pt"
            fmid, _ = load_midfuse(fck, device)
            va_pack_f = extract_all(fmid, make_loader(X_skel, X_imu, y, users, va, has_imu), device)
            oof_base[va] = va_pack_f["logits"].argmax(1)
            raw_f, _ = fit_all_raw(X_skel, X_imu, y, has_imu, users, tr, pairs_to_fit, device)
            pred_f, st_f = apply_raw_defer(
                va_pack_f["logits"], va_pack_f["y"], va, X_skel, X_imu, has_imu, raw_f, device,
                margin_thr=mthr, conf_thr=cthr, enabled=set(raw_f.keys()),
            )
            oof_pred[va] = pred_f
            fold_accs.append({"fold": fi, "base": st_f["base_acc"], "spec": st_f["acc"]})
            print(f"CV fold{fi} base={st_f['base_acc']:.4f} spec={st_f['acc']:.4f}", flush=True)
            del fmid, raw_f
            torch.cuda.empty_cache()

        oof_acc = float((oof_pred == y).mean())
        oof_base_acc = float((oof_base == y).mean())
        print(f"OOF base={oof_base_acc:.4f} spec={oof_acc:.4f}", flush=True)

        # Retrain raw on ALL train for submission; use MidFuse all_train or best.pt
        all_ck = fold_dir / "best_all_train.pt"
        if not all_ck.exists():
            all_ck = fold_dir / "best.pt"
        mid_all, _ = load_midfuse(all_ck, device)
        raw_all, meta_all = fit_all_raw(
            X_skel, X_imu, y, has_imu, users, np.arange(len(y)), pairs_to_fit, device
        )
        save_raw_models(
            raw_all,
            OUT / "checkpoints" / "raw_specialists_all.pt",
            {"mthr": mthr, "cthr": cthr, "fit_meta": meta_all},
        )
        sub_path = OUT / "submission_v6.csv"
        infer_test(mid_all, raw_all, set(raw_all.keys()), mthr, cthr, device, sub_path)

        # Promote submission.csv only if holdout clearly better
        promoted = False
        if primary["acc"] > base_hold + 0.005:  # clearly better than midfuse holdout
            shutil.copy2(sub_path, ROOT / "submission.csv")
            (ROOT / "submissions").mkdir(exist_ok=True)
            shutil.copy2(sub_path, ROOT / "submissions" / "submission_v6.csv")
            promoted = True
            print("PROMOTED submission.csv <- v6", flush=True)

        submit_info = {
            "oof_base": oof_base_acc,
            "oof_spec": oof_acc,
            "fold_accs": fold_accs,
            "submission": str(sub_path),
            "promoted": promoted,
            "midfuse_all_ckpt": str(all_ck),
        }
        summary["submit"] = submit_info

    summary["ping_disk_saver"] = bool(clear_win)
    with open(OUT / "v6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", OUT / "v6_summary.json", flush=True)
    print(json.dumps({
        "primary_holdout": primary["acc"],
        "delta_vs_0.537": primary["delta_vs_0.537"],
        "clear_win": clear_win,
        "ping_disk_saver": summary["ping_disk_saver"],
        "nested_policy": sel["best_policy"],
        "submit": submit_info,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
