"""Validate v29b error-driven SAFE gate with honest nested + write candidate CSV (promote track only if SAFE)."""
from __future__ import annotations
import csv, json, shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import softmax_np, apply_cfg, preds_full, nested_fixed, V7_CFG

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
PT = timezone(timedelta(hours=-7))
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
NESTED_MIN = 0.75196
MAX_DIS = 15
# winning uncapped rule from probe
PMAX, AMIN = 0.40, 0.55
MAXCH = 15  # cap; with pmax=0.4 only ~3 candidates historically


def selective_swap(primary, aux, T, pmax, amin, maxch, allow_mask=None):
    pp, ap = softmax_np(primary, T), softmax_np(aux, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    pconf, aconf = pp.max(1), ap.max(1)
    mask = (pa != aa) & (pconf <= pmax) & (aconf >= amin)
    if allow_mask is not None:
        mask = mask & allow_mask
    cand = np.where(mask)[0]
    order = cand[np.lexsort((-aconf[cand], pconf[cand]))][:maxch]
    out = primary.copy()
    for i in order:
        out[i] = aux[i]
    return out, order.tolist()


def main():
    from fuse_ir_v9 import load_members
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    def base_of(m):
        return m["base"] if m.get("base") is not None else m["logits"]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)
    t24 = np.load(CK24 / "hold_logits_seed42.npy").astype(np.float32)
    mask0 = th.any(1) & mid.any(1)
    T = V7_CFG["T"]

    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)

    # Global swap (cap non-binding historically)
    ir_sw, ch = selective_swap(classic9, t24, T, PMAX, AMIN, MAXCH, allow_mask=None)
    full, _ = apply_cfg(ir_sw, th, mid, yt, mask0, V7_CFG)
    nest = nested_fixed(ir_sw, th, mid, yt, yu, mask0, V7_CFG)
    pred = preds_full(ir_sw, th, mid, mask0, V7_CFG)
    dis = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())

    # Honest nested: reselect swaps with leave-out user excluded from candidate pool
    # (for independent per-clip rule this equals applying rule on te clips only)
    nest_reselect_folds = []
    for leave in (8, 9, 24):
        te = mask0 & (yu == leave)
        tr = mask0 & (yu != leave)
        if te.sum() < 5:
            continue
        # fit: only rank/select using train clips; apply rule thresholds on te independently
        # Since rule is per-clip threshold, apply directly on te (no train needed for thresholds)
        ir_te = classic9.copy()
        _, ch_te = selective_swap(classic9, t24, T, PMAX, AMIN, MAXCH, allow_mask=te)
        for i in ch_te:
            ir_te[i] = t24[i]
        # Also: capped variant using only train to decide global budget — not needed if |cand| < maxch
        te_acc, _ = apply_cfg(ir_te, th, mid, yt, te, V7_CFG)
        # train-capped: select up to maxch from train, then for te apply all matching (independent)
        ir_te2 = classic9.copy()
        pp, ap = softmax_np(classic9, T), softmax_np(t24, T)
        pa, aa = pp.argmax(1), ap.argmax(1)
        pconf, aconf = pp.max(1), ap.max(1)
        te_cand = np.where(te & (pa != aa) & (pconf <= PMAX) & (aconf >= AMIN))[0]
        for i in te_cand:
            ir_te2[i] = t24[i]
        te_acc2, _ = apply_cfg(ir_te2, th, mid, yt, te, V7_CFG)
        v7_te, _ = apply_cfg(classic9, th, mid, yt, te, V7_CFG)
        nest_reselect_folds.append({
            "leave": int(leave), "n": int(te.sum()), "n_swap_te": int(len(te_cand)),
            "te_acc_rule": float(te_acc2), "te_acc_v7": float(v7_te),
            "delta": float(te_acc2 - v7_te),
        })
    nest_reselect = float(np.mean([f["te_acc_rule"] for f in nest_reselect_folds]))

    # Which swaps help?
    swap_details = []
    for i in ch:
        ok_v7 = int(v7_preds[i] == yt[i]) if mask0[i] else -1
        ok_sw = int(pred[i] == yt[i]) if mask0[i] else -1
        swap_details.append({
            "idx": int(i), "user": int(yu[i]), "y": int(yt[i]),
            "v7_pred": int(v7_preds[i]), "sw_pred": int(pred[i]),
            "v7_ok": ok_v7, "sw_ok": ok_sw,
            "c9_conf": float(softmax_np(classic9[i:i+1], T).max()),
            "t24_conf": float(softmax_np(t24[i:i+1], T).max()),
        })

    clears = bool(nest["mean"] >= NESTED_MIN and nest_reselect >= NESTED_MIN and dis <= MAX_DIS
                  and nest["mean"] >= float(v7_nest["mean"]) - 1e-12
                  and nest_reselect >= float(v7_nest["mean"]) - 1e-12)

    print(f"v7 full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)
    print(f"swapped n={len(ch)} full={full:.6f} nested_fixed={nest['mean']:.6f} "
          f"nest_reselect={nest_reselect:.6f} dis={dis} clears={clears}", flush=True)
    print("folds", json.dumps(nest_reselect_folds), flush=True)
    print("swaps", json.dumps(swap_details), flush=True)

    # Build TEST submission candidate
    ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
    t24_test = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
    th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_p.exists():
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_p).astype(np.float32)

    ir_test_sw, ch_test = selective_swap(ir_test, t24_test, T, PMAX, AMIN, MAXCH)
    preds_v7 = (V7_CFG["wa"] * softmax_np(ir_test, T) + V7_CFG["wb"] * softmax_np(th_test, T)
                + V7_CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
    preds = (V7_CFG["wa"] * softmax_np(ir_test_sw, T) + V7_CFG["wb"] * softmax_np(th_test, T)
             + V7_CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
    test_dis = int((preds != preds_v7).sum())
    print(f"test swaps={len(ch_test)} disagree_vs_v7_test={test_dis}", flush=True)

    cache = ROOT / "cache" / "ir_yolo_v4"
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    # paths from skeleton fallback like write_ir_v7
    skel = TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv"
    path_by_sid = {}
    with open(skel, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # path like .../sample_id or similar — write_ir_v7 uses meta sample_id match
            path_by_sid[row["path"]] = row["path"]

    # Match write_ir_v7 path construction
    import importlib.util
    spec = importlib.util.spec_from_file_location("w7", ROOT / "write_ir_v7.py")
    # Just replicate write_ir_v7 loop
    src = (ROOT / "write_ir_v7.py").read_text(encoding="utf-8")
    # extract how paths are built by running a tiny copy:
    rows_out = []
    skel_rows = list(csv.DictReader(open(skel, encoding="utf-8")))
    # write_ir_v7 maps by enumerate meta order matching skel? Read the relevant section
    print("n_meta", len(meta), "n_skel", len(skel_rows), "n_pred", len(preds), flush=True)

    # Use write_ir_v7 logic inline from file snippet
    exec_globals = {}
    # simpler: load v7 csv paths in order
    v7_rows = list(csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8")))
    assert len(v7_rows) == len(preds)
    out_csv = ROOT / "submission_ir_v29b.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "prediction"])
        for i, row in enumerate(v7_rows):
            sid = meta[i]["sample_id"] if i < len(meta) else None
            if sid in empty:
                w.writerow([row["path"], row["prediction"]])  # keep v7/fallback for empty
            else:
                w.writerow([row["path"], int(preds[i])])
    # count actual prediction disagrees vs v7 file
    file_dis = 0
    with open(out_csv, encoding="utf-8") as f:
        new_rows = list(csv.DictReader(f))
    for a, b in zip(v7_rows, new_rows):
        if a["prediction"] != b["prediction"]:
            file_dis += 1
    print(f"wrote {out_csv} file_dis_vs_v7={file_dis}", flush=True)

    promoted = False
    track_sub = TRACK / "submission.csv"
    if clears and test_dis <= MAX_DIS and file_dis <= MAX_DIS:
        shutil.copy2(out_csv, track_sub)
        promoted = True
        print("PROMOTED track submission.csv -> v29b", flush=True)
    else:
        print("NOT promoted; keep track submission as v7", flush=True)
        # ensure track still v7
        shutil.copy2(ROOT / "submission_ir_v7.csv", track_sub)

    report = {
        "tag": "ir_v29b_error_driven_validated",
        "updated_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "outcome": "SAFE_WIN" if (clears and promoted) else ("HOLD_CLEAR_NOT_PROMOTED" if clears else "MISS"),
        "keep_ir_v7": not promoted,
        "wrote_csv": True,
        "csv": str(out_csv),
        "promoted_track": promoted,
        "public_submit": None,
        "rule": {"pmax": PMAX, "amin": AMIN, "maxch": MAXCH, "T": T, "mid": "classic_aligned_mid"},
        "hold": {
            "v7_full": float(v7_full), "v7_nested": float(v7_nest["mean"]),
            "full": float(full), "nested_fixed": float(nest["mean"]),
            "nested_reselect": float(nest_reselect),
            "disagree_vs_v7": dis, "n_swapped": len(ch),
            "folds": nest_reselect_folds, "swap_details": swap_details,
        },
        "test": {"n_swapped": len(ch_test), "disagree_vs_v7_preds": test_dis, "file_dis_vs_v7": file_dis},
        "promote_gate": {"nested_fixed_min": NESTED_MIN, "disagree_vs_v7_max": MAX_DIS},
        "clears_gate": clears,
        "note": "No Kaggle submit from this script; CSV written for review.",
    }
    outp = ROOT / "metrics_ir_v29b_validated.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # also write under D:\CUHK-X explicit path
    Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\metrics_ir_v29b_validated.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["outcome", "clears_gate", "promoted_track", "hold", "test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()