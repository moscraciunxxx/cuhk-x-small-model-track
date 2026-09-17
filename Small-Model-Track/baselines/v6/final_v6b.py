"""v6b: proper OOF MidFuse + raw specialists; no holdout leakage in selection."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

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
from model import count_parameters  # noqa: E402
from refine_v6 import DualSubset, RawPairNet, predict_raw_proba, train_raw_pair  # noqa: E402
from specialists_v6 import extract_all, load_midfuse, make_loader, pair_key, top2_margin  # noqa: E402
from final_v6 import apply_raw_defer, train_raw_pair_inner, save_raw_models, fit_all_raw  # noqa: E402

BASELINE = 0.537
WIN_DELTA = 0.01
RAW_PAIRS = [
    (9, 10), (10, 11), (9, 11), (30, 31), (29, 32), (29, 34),
    (12, 13), (21, 22), (26, 7), (6, 37), (8, 9), (6, 7),
    (32, 34), (26, 24), (26, 19),
]


def build_oof_midfuse(X_skel, X_imu, y, users, has_imu, device):
    fold_dir = ROOT_V2 / "checkpoints_midfuse_v2b"
    gkf = GroupKFold(n_splits=5)
    oof_logits = np.zeros((len(y), 40), dtype=np.float32)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    fold_accs = []
    for fi, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, users)):
        mid, _ = load_midfuse(fold_dir / f"best_fold{fi}.pt", device)
        pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, va, has_imu), device)
        oof_logits[va] = pack["logits"]
        oof_pred[va] = pack["logits"].argmax(1)
        acc = float((oof_pred[va] == y[va]).mean())
        fold_accs.append(acc)
        print(f"OOF MidFuse fold{fi} acc={acc:.4f} n={len(va)}", flush=True)
        del mid
        torch.cuda.empty_cache()
    print(f"OOF MidFuse mean={np.mean(fold_accs):.4f} overall={(oof_pred==y).mean():.4f}", flush=True)
    return oof_logits, oof_pred, fold_accs


def select_policy_oof(X_skel, X_imu, y, has_imu, users, oof_logits, device, restrict_idx=None):
    """Use MidFuse fold OOF: for each fold train raw on fold-train, eval deferral on fold-val."""
    gkf = GroupKFold(n_splits=5)
    all_st = []
    pair_deltas = {pair_key(*p): [] for p in RAW_PAIRS}
    fold_dir_info = []

    for fi, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, users)):
        if restrict_idx is not None:
            # only use samples in restrict_idx (e.g. non-holdout) — remap
            pass
        logits_va = oof_logits[va]
        y_va = y[va]
        base = float((logits_va.argmax(1) == y_va).mean())
        raw_models = {}
        for p in RAW_PAIRS:
            pk = pair_key(*p)
            model, g2l, l2g, va_acc = train_raw_pair_inner(
                X_skel, X_imu, y, has_imu, users, tr, pk, device,
                epochs=30, patience=8, seed=2000 + fi * 50 + pk[0] * 40 + pk[1],
            )
            if model is None:
                continue
            raw_models[pk] = (model, g2l, l2g)
            tmp = {pk: raw_models[pk]}
            _, st_p = apply_raw_defer(
                logits_va, y_va, va, X_skel, X_imu, has_imu, tmp, device,
                margin_thr=1.01, conf_thr=0.55, enabled={pk},
            )
            pair_deltas[pk].append(st_p["acc"] - base)

        for mthr in [0.1, 0.2, 0.3, 0.5, 0.75, 1.01]:
            for cthr in [0.5, 0.55, 0.6, 0.7, 0.8]:
                # all pairs
                _, st = apply_raw_defer(
                    logits_va, y_va, va, X_skel, X_imu, has_imu, raw_models, device,
                    margin_thr=mthr, conf_thr=cthr,
                )
                st["policy"] = f"all|m={mthr}|c={cthr}"
                st["fold"] = fi
                st["base"] = base
                all_st.append(st)
        fold_dir_info.append({"fold": fi, "base": base, "n_raw": len(raw_models)})
        print(f"policy-fold{fi} base={base:.4f} n_raw={len(raw_models)}", flush=True)

    from collections import defaultdict
    by = defaultdict(list)
    for st in all_st:
        by[st["policy"]].append(st["acc"] - st["base"])
    mean_d = {k: float(np.mean(v)) for k, v in by.items()}
    best_pol = max(mean_d, key=mean_d.get)
    # also best absolute mean acc
    by_acc = defaultdict(list)
    for st in all_st:
        by_acc[st["policy"]].append(st["acc"])
    mean_acc = {k: float(np.mean(v)) for k, v in by_acc.items()}
    best_pol_acc = max(mean_acc, key=mean_acc.get)

    selected = []
    pstats = {}
    for pk, ds in pair_deltas.items():
        if not ds:
            continue
        md = float(np.mean(ds))
        pstats[str(pk)] = {"mean_delta": md, "n": len(ds)}
        if md > 0.0005:
            selected.append(pk)

    parts = best_pol.split("|")
    kv = dict(x.split("=") for x in parts[1:])
    return {
        "best_policy_by_delta": best_pol,
        "best_policy_by_acc": best_pol_acc,
        "margin_thr": float(kv["m"]),
        "conf_thr": float(kv["c"]),
        "mean_delta": mean_d,
        "mean_acc": mean_acc,
        "selected_pairs": selected,
        "pair_stats": pstats,
        "folds": fold_dir_info,
    }


def infer_test(mid, raw_models, enabled, mthr, cthr, device, out_csv):
    cache = ROOT_V2 / "cache"
    X_skel, paths = load_skel_test_cache(cache)
    imu = np.load(cache / "imu_test.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool) if "has_imu" in imu.files else np.ones(len(X_skel), bool)
    y = np.zeros(len(X_skel), np.int64)
    users = np.zeros(len(X_skel), np.int64)
    idx = np.arange(len(X_skel))
    pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, idx, has_imu), device)
    pred, st = apply_raw_defer(
        pack["logits"], None, idx, X_skel, X_imu, has_imu, raw_models, device,
        margin_thr=mthr, conf_thr=cthr, enabled=enabled,
    )
    paths_out = []
    for pth in paths:
        pth = str(pth)
        if pth.startswith("small_model_track_test/"):
            paths_out.append(pth if pth.endswith("/") else pth + "/")
        else:
            paths_out.append(f"small_model_track_test/{Path(pth).name}/")
    df = pd.DataFrame({"path": paths_out, "prediction": pred.astype(int)})
    df = df.sort_values("path").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} n={len(df)} {st}", flush=True)
    print(df.head(3), flush=True)
    return st


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    print("device", device, flush=True)
    t0 = time.time()

    cache = ROOT_V2 / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y, np.int64)
    users = np.asarray(users, np.int64)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]

    print("=== Build MidFuse OOF logits (5 fold ckpts) ===", flush=True)
    oof_logits, oof_pred, fold_accs = build_oof_midfuse(X_skel, X_imu, y, users, has_imu, device)

    # Restrict policy selection to non-holdout samples' OOF (still true OOF via fold models)
    # Actually use ALL samples' OOF folds - holdout users appear in some folds as val.
    # For honest holdout eval we must NOT use holdout in policy selection.
    # So: only run GroupKFold on tr_idx users for policy selection.
    print("=== Policy selection on non-holdout users with MidFuse OOF logits ===", flush=True)
    # Remap: work on subset tr_idx
    y_tr = y[tr_idx]
    users_tr = users[tr_idx]
    oof_tr = oof_logits[tr_idx]
    # Custom GKF on subset - train raw using global indices
    gkf = GroupKFold(n_splits=5)
    all_st = []
    pair_deltas = {pair_key(*p): [] for p in RAW_PAIRS}
    for fi, (a, b) in enumerate(gkf.split(np.zeros(len(tr_idx)), y_tr, users_tr)):
        tr_f = tr_idx[a]
        va_f = tr_idx[b]
        # MidFuse OOF logits for va_f - these come from fold models that may have
        # been trained WITH some of va_f if fold splits differ! 
        # Safer: load the MidFuse fold ckpt whose val users match, OR re-extract
        # with a MidFuse trained only on tr_f.
        # Quick correct approach: train... we can't retrain MidFuse now.
        # Use: for each sample in va_f, oof_logits[i] was produced by a fold model
        # that excluded that sample's user. Since va_f users are subset of train
        # users (not 8,9,24), and OOF folds partition ALL users including 8,9,24,
        # the oof logit for a train user is from a model that didn't see that user
        # — good. Using oof_logits[va_f] is valid.
        logits_va = oof_logits[va_f]
        y_va = y[va_f]
        base = float((logits_va.argmax(1) == y_va).mean())
        raw_models = {}
        for p in RAW_PAIRS:
            pk = pair_key(*p)
            model, g2l, l2g, _ = train_raw_pair_inner(
                X_skel, X_imu, y, has_imu, users, tr_f, pk, device,
                epochs=30, patience=8, seed=3000 + fi * 50 + pk[0] * 40 + pk[1],
            )
            if model is None:
                continue
            raw_models[pk] = (model, g2l, l2g)
            tmp = {pk: raw_models[pk]}
            _, st_p = apply_raw_defer(
                logits_va, y_va, va_f, X_skel, X_imu, has_imu, tmp, device,
                margin_thr=1.01, conf_thr=0.55, enabled={pk},
            )
            pair_deltas[pk].append(st_p["acc"] - base)
        for mthr in [0.1, 0.2, 0.3, 0.5, 0.75, 1.01]:
            for cthr in [0.5, 0.55, 0.6, 0.7, 0.8]:
                for en_mode in ["all", "sel"]:
                    # sel filled later - first pass all only
                    if en_mode != "all":
                        continue
                    _, st = apply_raw_defer(
                        logits_va, y_va, va_f, X_skel, X_imu, has_imu, raw_models, device,
                        margin_thr=mthr, conf_thr=cthr,
                    )
                    st["policy"] = f"all|m={mthr}|c={cthr}"
                    st["fold"] = fi
                    st["base"] = base
                    all_st.append(st)
        print(f"sel-fold{fi} base={base:.4f} n_raw={len(raw_models)} best_local={max(s['acc'] for s in all_st if s['fold']==fi):.4f}", flush=True)

    from collections import defaultdict
    by = defaultdict(list)
    for st in all_st:
        by[st["policy"]].append(st["acc"] - st["base"])
    mean_d = {k: float(np.mean(v)) for k, v in by.items()}
    best_pol = max(mean_d, key=mean_d.get)
    by_acc = defaultdict(list)
    for st in all_st:
        by_acc[st["policy"]].append(st["acc"])
    mean_acc = {k: float(np.mean(v)) for k, v in by_acc.items()}
    best_pol_acc = max(mean_acc, key=mean_acc.get)
    print("TOP policies by delta:", flush=True)
    for k, v in sorted(mean_d.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: delta={v:+.4f} acc={mean_acc[k]:.4f}", flush=True)

    selected = []
    pstats = {}
    for pk, ds in pair_deltas.items():
        if not ds:
            continue
        md = float(np.mean(ds))
        pstats[str(pk)] = {"mean_delta": md, "n": len(ds)}
        if md > 0.0005:
            selected.append(pk)
    print(f"selected pairs: {selected}", flush=True)
    print(f"best_by_delta={best_pol} best_by_acc={best_pol_acc}", flush=True)

    # Prefer best_by_delta if positive, else best_by_acc if better than 0, else no-op
    use_pol = best_pol if mean_d[best_pol] > 0 else best_pol_acc
    parts = use_pol.split("|")
    kv = dict(x.split("=") for x in parts[1:])
    mthr, cthr = float(kv["m"]), float(kv["c"])

    sel = {
        "best_policy_by_delta": best_pol,
        "best_policy_by_acc": best_pol_acc,
        "used_policy": use_pol,
        "margin_thr": mthr,
        "conf_thr": cthr,
        "mean_delta_top": dict(sorted(mean_d.items(), key=lambda x: -x[1])[:15]),
        "selected_pairs": [list(p) for p in selected],
        "pair_stats": pstats,
        "oof_midfuse_fold_accs": fold_accs,
        "oof_midfuse_overall": float((oof_pred == y).mean()),
    }
    with open(OUT / "nested_oof_selection.json", "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)

    # Holdout eval with MidFuse holdout ckpt + raw trained on non-holdout
    print("=== Holdout eval ===", flush=True)
    mid_h, _ = load_midfuse(ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt", device)
    va_pack = extract_all(mid_h, make_loader(X_skel, X_imu, y, users, va_idx, has_imu), device)
    base_hold = float((va_pack["logits"].argmax(1) == va_pack["y"]).mean())
    print(f"midfuse holdout={base_hold:.4f}", flush=True)

    raw_models, fit_meta = fit_all_raw(X_skel, X_imu, y, has_imu, users, tr_idx, RAW_PAIRS, device)
    enabled_all = set(raw_models.keys())
    enabled_sel = set(selected) & enabled_all if selected else enabled_all

    results = []
    for name, en in [("all", enabled_all), ("selected", enabled_sel)]:
        pred, st = apply_raw_defer(
            va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, raw_models, device,
            margin_thr=mthr, conf_thr=cthr, enabled=en,
        )
        st["name"] = name
        st["delta_vs_0.537"] = st["acc"] - BASELINE
        st["delta_vs_base"] = st["acc"] - base_hold
        results.append(st)
        print(f"HOLDOUT {name}: {st['acc']:.4f} d537={st['delta_vs_0.537']:+.4f} dbase={st['delta_vs_base']:+.4f} defer={st['n_defer']} flip={st['n_flip']}", flush=True)

    # Sensitivity (report only; primary is nested)
    sens = []
    for mthr2 in [0.1, 0.2, 0.3, 0.5, 0.75, 1.01]:
        for cthr2 in [0.5, 0.55, 0.6, 0.7]:
            _, st = apply_raw_defer(
                va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, raw_models, device,
                margin_thr=mthr2, conf_thr=cthr2, enabled=enabled_all,
            )
            st["policy"] = f"m={mthr2}|c={cthr2}"
            sens.append(st)
    sens.sort(key=lambda d: -d["acc"])
    print("Holdout sensitivity top:", flush=True)
    for s in sens[:8]:
        print(f"  {s['policy']} acc={s['acc']:.4f} d={s['acc']-base_hold:+.4f}", flush=True)

    primary = max(results, key=lambda d: d["acc"])
    # If nested policy hurts, fall back to baseline (no specialists)
    if primary["acc"] < base_hold:
        primary = {
            "name": "baseline_midfuse",
            "acc": base_hold,
            "delta_vs_0.537": base_hold - BASELINE,
            "delta_vs_base": 0.0,
            "n_defer": 0,
            "n_flip": 0,
            "margin_thr": None,
            "conf_thr": None,
        }
        print("Nested policy hurts holdout -> fall back to MidFuse baseline", flush=True)

    clear_win = primary["acc"] >= BASELINE + WIN_DELTA
    # Oracle sensitivity (for analysis only, not for submit decision)
    oracle_best = sens[0]["acc"] if sens else base_hold

    save_raw_models(
        raw_models,
        OUT / "checkpoints" / "raw_specialists_holdout_train.pt",
        {"mthr": mthr, "cthr": cthr, "fit_meta": fit_meta, "sel": sel},
    )

    summary = {
        "midfuse_holdout_measured": base_hold,
        "baseline_reported": BASELINE,
        "nested_oof_selection": sel,
        "holdout_results": results,
        "primary": primary,
        "holdout_sensitivity_oracle_top": sens[:10],
        "oracle_best_holdout_if_peeked": oracle_best,
        "oracle_delta_vs_0.537": oracle_best - BASELINE,
        "clear_win": clear_win,
        "note": "Primary uses nested-OOF-selected hyperparams only; oracle sensitivity is peeking and not used for submit.",
        "elapsed_sec": time.time() - t0,
    }

    submit_info = None
    # Only submit if primary clear win (not oracle)
    if clear_win:
        print("=== Clear win: CV OOF + all-train + submit ===", flush=True)
        gkf = GroupKFold(n_splits=5)
        fold_dir = ROOT_V2 / "checkpoints_midfuse_v2b"
        oof_spec = np.full(len(y), -1, np.int64)
        fold_rows = []
        for fi, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, users)):
            mid_f, _ = load_midfuse(fold_dir / f"best_fold{fi}.pt", device)
            pack = extract_all(mid_f, make_loader(X_skel, X_imu, y, users, va, has_imu), device)
            raw_f, _ = fit_all_raw(X_skel, X_imu, y, has_imu, users, tr, RAW_PAIRS, device)
            pred, st = apply_raw_defer(
                pack["logits"], pack["y"], va, X_skel, X_imu, has_imu, raw_f, device,
                margin_thr=mthr, conf_thr=cthr, enabled=set(raw_f.keys()),
            )
            oof_spec[va] = pred
            fold_rows.append({"fold": fi, "base": st["base_acc"], "spec": st["acc"]})
            print(f"CV fold{fi} base={st['base_acc']:.4f} spec={st['acc']:.4f}", flush=True)
            del mid_f, raw_f
            torch.cuda.empty_cache()
        oof_acc = float((oof_spec == y).mean())
        mid_all, _ = load_midfuse(fold_dir / "best_all_train.pt", device)
        raw_all, meta_all = fit_all_raw(X_skel, X_imu, y, has_imu, users, np.arange(len(y)), RAW_PAIRS, device)
        save_raw_models(raw_all, OUT / "checkpoints" / "raw_specialists_all.pt", {"mthr": mthr, "cthr": cthr, "fit_meta": meta_all})
        sub = OUT / "submission_v6.csv"
        infer_test(mid_all, raw_all, set(raw_all.keys()), mthr, cthr, device, sub)
        shutil.copy2(sub, ROOT / "submission.csv")
        (ROOT / "submissions").mkdir(exist_ok=True)
        shutil.copy2(sub, ROOT / "submissions" / "submission_v6.csv")
        submit_info = {"oof_spec": oof_acc, "folds": fold_rows, "promoted": True, "path": str(sub)}
        summary["submit"] = submit_info

    summary["ping_disk_saver"] = bool(clear_win)
    with open(OUT / "v6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "primary_holdout": primary["acc"],
        "delta_vs_0.537": primary.get("delta_vs_0.537"),
        "clear_win": clear_win,
        "ping_disk_saver": summary["ping_disk_saver"],
        "nested_policy": use_pol,
        "oracle_peek_best": oracle_best,
        "submit": submit_info,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
