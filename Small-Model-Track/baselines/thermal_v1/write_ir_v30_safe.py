"""ir_v30 SAFE: v29b IR swaps + thermal conf-gate (v6 s3141 into v2trio) + classic mid + V7_CFG.
Gate: nested_fixed > v29b local AND disagree<=15 vs v29b/v7.
Fuse/gate math lives in evaluate_v30_hold (imports nested_fixed/apply_cfg from probe_ir_v24_fuse).
CSV I/O lives in write_v30_submission. main() is the writer entry.
"""
from __future__ import annotations
import csv, json, shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import softmax_np, apply_cfg, preds_full, nested_fixed, V7_CFG
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
PT = timezone(timedelta(hours=-7))
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
CK_V6 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v6_lift"
CK_V3 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3"
DEPLOY_FP16 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "model_fp16.pt"
DEPLOY_YOLO = ROOT / "yolov8n.pt"

V29B_NESTED = 0.75405874529429
MAX_DIS = 15
# IR rule (v29b)
IR_PMAX, IR_AMIN, IR_MAXCH = 0.40, 0.55, 15
# Thermal rule (found: nested 0.756209 dis=1)
TH_TCONF, TH_PMAX, TH_AMIN, TH_MAXCH, TH_MM = 1.25, 0.25, 0.30, 20, 0.0
SAMPLE_SUB = TRACK / "Testing" / "test_file" / "sample_submission.csv"


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


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


def th_selective(th_base, th_aux, Tconf, pmax, amin, maxch, min_margin=0.0, allow_mask=None):
    pp, ap = softmax_np(th_base, Tconf), softmax_np(th_aux, Tconf)
    both = th_base.any(1) & th_aux.any(1)
    pa, aa = pp.argmax(1), ap.argmax(1)
    pconf, aconf = pp.max(1), ap.max(1)
    part = np.partition(ap, -2, axis=1)
    amargin = part[:, -1] - part[:, -2]
    mask = both & (pa != aa) & (pconf <= pmax) & (aconf >= amin) & (amargin >= min_margin)
    if allow_mask is not None:
        mask = mask & allow_mask
    cand = np.where(mask)[0]
    order = cand[np.lexsort((-amargin[cand], -aconf[cand], pconf[cand]))][:maxch]
    out = th_base.copy()
    for i in order:
        out[i] = th_aux[i]
    return out, order.tolist()


def key_align_hold(h504, th_v6, hold_idx):
    th_meta = json.loads((ROOT / "cache" / "thermal_yolo" / "train_meta.json").read_text(encoding="utf-8"))
    ir_meta = json.loads((ROOT / "cache" / "ir_yolo_v4" / "train_meta.json").read_text(encoding="utf-8"))
    tusers = np.load(ROOT / "cache" / "thermal_yolo" / "train_users.npy")
    th_hold_idx = np.where(np.isin(tusers, list(DEFAULT_HOLD_OUT_USERS)))[0]
    ir_keys = [(int(ir_meta[i]["user_id"]), int(ir_meta[i]["label"]), str(ir_meta[i].get("trial", ""))) for i in hold_idx]
    th_keys = [(int(th_meta[i]["user_id"]), int(th_meta[i]["label"]), str(th_meta[i].get("trial", ""))) for i in th_hold_idx]
    key_to_th = {k: i for i, k in enumerate(th_keys)}
    out = np.zeros_like(th_v6)
    matched = 0
    for j, k in enumerate(ir_keys):
        if k in key_to_th:
            out[j] = h504[key_to_th[k]]
            matched += 1
    return out, matched


def deploy_pack_mb():
    """Packed deploy weights this stack ships: thermal R2P1D fp16 + YOLOv8n."""
    return (DEPLOY_FP16.stat().st_size + DEPLOY_YOLO.stat().st_size) / (1024 * 1024)


def evaluate_v30_hold(th_ckpt=None, verbose=True):
    """Load hold logits, apply v30 IR+thermal rules, run shipped nested_fixed/apply_cfg.

    Returns tensors plus gate bits so callers can re-run nested_fixed/apply_cfg on the
    same swapped logits (no reimplementation).
    """
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th0 = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)
    t24 = np.load(CK24 / "hold_logits_seed42.npy").astype(np.float32)
    mask0 = th0.any(1) & mid.any(1)
    T = V7_CFG["T"]

    ckpt_path = Path(th_ckpt) if th_ckpt is not None else (CK_V6 / "pool_seed3141.pt")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = np.asarray(blob["hold_logits"], np.float32)
    th_new, matched = key_align_hold(h, th0, hold_idx)
    if verbose:
        print(f"aligned {ckpt_path.name} matched={matched} val={blob['val_acc']:.4f}", flush=True)

    ir_sw, ch_ir = selective_swap(classic9, t24, T, IR_PMAX, IR_AMIN, IR_MAXCH)
    th_sw, ch_th = th_selective(th0, th_new, TH_TCONF, TH_PMAX, TH_AMIN, TH_MAXCH, TH_MM)

    v7_full, _ = apply_cfg(classic9, th0, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th0, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th0, mid, mask0, V7_CFG)
    v29_full, _ = apply_cfg(ir_sw, th0, mid, yt, mask0, V7_CFG)
    v29_nest = nested_fixed(ir_sw, th0, mid, yt, yu, mask0, V7_CFG)
    v29_preds = preds_full(ir_sw, th0, mid, mask0, V7_CFG)

    full, _ = apply_cfg(ir_sw, th_sw, mid, yt, mask0, V7_CFG)
    nest = nested_fixed(ir_sw, th_sw, mid, yt, yu, mask0, V7_CFG)
    pred = preds_full(ir_sw, th_sw, mid, mask0, V7_CFG)
    dis_v7 = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
    dis_v29 = int(((pred >= 0) & (v29_preds >= 0) & (pred != v29_preds)).sum())

    folds = []
    for leave in (8, 9, 24):
        te = mask0 & (yu == leave)
        if te.sum() < 5:
            continue
        ir_te = classic9.copy()
        _, ch_te_ir = selective_swap(classic9, t24, T, IR_PMAX, IR_AMIN, IR_MAXCH, allow_mask=te)
        for i in ch_te_ir:
            ir_te[i] = t24[i]
        th_te = th0.copy()
        _, ch_te_th = th_selective(th0, th_new, TH_TCONF, TH_PMAX, TH_AMIN, TH_MAXCH, TH_MM, allow_mask=te)
        for i in ch_te_th:
            th_te[i] = th_new[i]
        te_acc, _ = apply_cfg(ir_te, th_te, mid, yt, te, V7_CFG)
        v29_te, _ = apply_cfg(ir_sw, th0, mid, yt, te, V7_CFG)
        ir_v29_te = classic9.copy()
        _, ch_v29 = selective_swap(classic9, t24, T, IR_PMAX, IR_AMIN, IR_MAXCH, allow_mask=te)
        for i in ch_v29:
            ir_v29_te[i] = t24[i]
        v29_te2, _ = apply_cfg(ir_v29_te, th0, mid, yt, te, V7_CFG)
        folds.append({
            "leave": int(leave), "n": int(te.sum()),
            "n_swap_ir": len(ch_te_ir), "n_swap_th": len(ch_te_th),
            "te_acc_v30": float(te_acc), "te_acc_v29b_global": float(v29_te),
            "te_acc_v29b_reselect": float(v29_te2),
            "delta_vs_v29b_reselect": float(te_acc - v29_te2),
        })
    nest_reselect = float(np.mean([f["te_acc_v30"] for f in folds]))
    v29_reselect = float(np.mean([f["te_acc_v29b_reselect"] for f in folds]))

    clears = bool(
        float(nest["mean"]) > V29B_NESTED + 1e-12
        and nest_reselect > v29_reselect + 1e-12
        and nest_reselect > V29B_NESTED + 1e-12
        and dis_v7 <= MAX_DIS
        and dis_v29 <= MAX_DIS
    )
    if verbose:
        print(f"v7 nested={v7_nest['mean']:.6f} v29b nested={v29_nest['mean']:.6f} v29_reselect={v29_reselect:.6f}", flush=True)
        print(f"v30 full={full:.6f} nested_fixed={nest['mean']:.6f} nest_reselect={nest_reselect:.6f}", flush=True)
        print(f"dis_v7={dis_v7} dis_v29={dis_v29} n_ir={len(ch_ir)} n_th={len(ch_th)} clears={clears}", flush=True)
        print("folds", json.dumps(folds), flush=True)

    return {
        "classic9": classic9, "t24": t24, "th0": th0, "th_new": th_new, "mid": mid,
        "yt": yt, "yu": yu, "mask0": mask0, "T": T,
        "ir_sw": ir_sw, "th_sw": th_sw,
        "ch_ir": ch_ir, "ch_th": ch_th,
        "v7_full": float(v7_full), "v7_nest": v7_nest, "v7_preds": v7_preds,
        "v29_full": float(v29_full), "v29_nest": v29_nest, "v29_preds": v29_preds,
        "full": float(full), "nest": nest, "pred": pred,
        "dis_v7": dis_v7, "dis_v29": dis_v29,
        "folds": folds, "nest_reselect": nest_reselect, "v29_reselect": v29_reselect,
        "clears": clears, "blob": blob, "matched": matched,
        "nested_fixed_mean": float(nest["mean"]),
    }


def write_v30_submission(hold=None, out_csv=None, test_th_path=None, verbose=True):
    """Write test CSV with the same IR+thermal swap rules. Returns write metadata."""
    if hold is None:
        hold = evaluate_v30_hold(verbose=verbose)
    T = hold["T"]
    clears = hold["clears"]
    wrote = False
    out_csv = Path(out_csv) if out_csv is not None else (ROOT / "submission_ir_v30.csv")
    test_info = None
    test_th_path = Path(test_th_path) if test_th_path is not None else (CK_V6 / "test_logits_seed3141.npy")
    if clears and test_th_path.exists():
        ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
        t24_test = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
        th_test_old = np.load(CK_V3 / "test_logits.npy").astype(np.float32)
        th_test_new = np.load(test_th_path).astype(np.float32)
        ir_test_sw, ch_te_ir = selective_swap(ir_test, t24_test, T, IR_PMAX, IR_AMIN, IR_MAXCH)
        th_test_sw, ch_te_th = th_selective(th_test_old, th_test_new, TH_TCONF, TH_PMAX, TH_AMIN, TH_MAXCH, TH_MM)

        preds_v7 = (V7_CFG["wa"] * softmax_np(ir_test, T) + V7_CFG["wb"] * softmax_np(th_test_old, T)
                    + V7_CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
        ir_v29, _ = selective_swap(ir_test, t24_test, T, IR_PMAX, IR_AMIN, IR_MAXCH)
        preds_v29 = (V7_CFG["wa"] * softmax_np(ir_v29, T) + V7_CFG["wb"] * softmax_np(th_test_old, T)
                     + V7_CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
        preds = (V7_CFG["wa"] * softmax_np(ir_test_sw, T) + V7_CFG["wb"] * softmax_np(th_test_sw, T)
                 + V7_CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
        test_dis_v7 = int((preds != preds_v7).sum())
        test_dis_v29 = int((preds != preds_v29).sum())
        test_info = {
            "n_swap_ir": len(ch_te_ir), "n_swap_th": len(ch_te_th),
            "disagree_vs_v7": test_dis_v7, "disagree_vs_v29b": test_dis_v29,
        }
        if verbose:
            print(f"test swaps ir={len(ch_te_ir)} th={len(ch_te_th)} dis_v7={test_dis_v7} dis_v29={test_dis_v29}", flush=True)

        if test_dis_v7 <= MAX_DIS and test_dis_v29 <= MAX_DIS:
            v7_rows = list(csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8")))
            meta = json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_meta.json").read_text(encoding="utf-8"))
            empty = set(json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_empty.json").read_text(encoding="utf-8")))
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "prediction"])
                for i, row in enumerate(v7_rows):
                    sid = meta[i]["sample_id"] if i < len(meta) else None
                    if sid in empty:
                        w.writerow([row["path"], row["prediction"]])
                    else:
                        w.writerow([row["path"], int(preds[i])])
            file_dis_v29 = 0
            v29_rows = list(csv.DictReader(open(ROOT / "submission_ir_v29b.csv", encoding="utf-8")))
            new_rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
            for a, b in zip(v29_rows, new_rows):
                if a["prediction"] != b["prediction"]:
                    file_dis_v29 += 1
            test_info["file_dis_vs_v29b"] = file_dis_v29
            if file_dis_v29 <= MAX_DIS and file_dis_v29 >= 1:
                shutil.copy2(out_csv, TRACK / "submission.csv")
                wrote = True
                if verbose:
                    print(f"WROTE {out_csv} file_dis_v29={file_dis_v29} PROMOTED track", flush=True)
            else:
                if verbose:
                    print(f"CSV built but file_dis_v29={file_dis_v29} not in [1,{MAX_DIS}]; not promoting", flush=True)
        else:
            if verbose:
                print("HOLD clear but test disagree too high; no CSV", flush=True)
    elif clears and not test_th_path.exists():
        if verbose:
            print(f"HOLD clears but waiting for {test_th_path}", flush=True)
    else:
        if verbose:
            print("MISS gate", flush=True)
    return {"wrote": wrote, "csv": str(out_csv) if wrote else None, "out_csv": out_csv,
            "test": test_info, "hold": hold, "clears": clears}


def main():
    hold = evaluate_v30_hold(verbose=True)
    result = write_v30_submission(hold, verbose=True)
    wrote = result["wrote"]
    clears = hold["clears"]
    nest = hold["nest"]
    nest_reselect = hold["nest_reselect"]
    dis_v7 = hold["dis_v7"]
    dis_v29 = hold["dis_v29"]
    blob = hold["blob"]
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    report = {
        "tag": "ir_v30_thermal_confgate",
        "updated_at": now,
        "best_public": {"csv": "submission_ir_v29b.csv", "public": 0.70149},
        "outcome": "SAFE_WIN" if wrote else ("HOLD_CLEAR_WAIT_TEST" if clears else "MISS"),
        "wrote_csv": wrote,
        "csv": result["csv"],
        "promoted_track": wrote,
        "rule": {
            "ir": {"pmax": IR_PMAX, "amin": IR_AMIN, "maxch": IR_MAXCH},
            "thermal": {"Tconf": TH_TCONF, "pmax": TH_PMAX, "amin": TH_AMIN, "maxch": TH_MAXCH, "min_margin": TH_MM,
                        "seed": 3141, "base": "th_v6_v2trio"},
            "mid": "classic_aligned_mid",
            "cfg": dict(V7_CFG),
        },
        "hold": {
            "v7_full": hold["v7_full"], "v7_nested": float(hold["v7_nest"]["mean"]),
            "v29b_full": hold["v29_full"], "v29b_nested": float(hold["v29_nest"]["mean"]),
            "v29b_reselect": hold["v29_reselect"],
            "full": hold["full"], "nested_fixed": float(nest["mean"]), "nested_reselect": nest_reselect,
            "disagree_vs_v7": dis_v7, "disagree_vs_v29b": dis_v29,
            "n_swap_ir": len(hold["ch_ir"]), "n_swap_th": len(hold["ch_th"]),
            "folds": hold["folds"],
            "thermal_solo_s3141": float(blob["val_acc"]),
        },
        "test": result["test"],
        "promote_gate": {"nested_min_strict_gt": V29B_NESTED, "disagree_max": MAX_DIS},
        "clears_gate": clears,
        "deploy_pack_mb": deploy_pack_mb(),
        "note": "Writer entry write_ir_v30_safe.main; kaggle submit is a separate step after local gate.",
    }
    (ROOT / "metrics_ir_v30_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v30_thermal_confgate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "HOTC_GPU_HANDOFF.md").write_text(
        f"""# CUHK-X Small Model Track - status

## BEST PUBLIC: ir_v29b @ **0.70149**
## SAFE_WIN candidate: submission_ir_v30.csv (promoted track)
- nested_fixed **{nest['mean']:.6f}** (> v29b {V29B_NESTED:.6f}) | nest_reselect {nest_reselect:.6f}
- hold dis_v7={dis_v7} dis_v29={dis_v29} | test file_dis_v29={None if result['test'] is None else result['test'].get('file_dis_vs_v29b')}
- Rule: v29b IR swaps + thermal conf-gate s3141 into v2trio (Tconf=1.25 pmax=0.25 amin=0.30 maxch=20) + classic mid + V7_CFG
- thermal s3141 solo **{blob['val_acc']:.4f}**; s7777 trained (solo weaker)
- wrote_csv={wrote} deploy_pack_mb={report['deploy_pack_mb']:.2f}

Updated {now}
""",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ["outcome", "clears_gate", "wrote_csv", "hold", "test", "deploy_pack_mb"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
