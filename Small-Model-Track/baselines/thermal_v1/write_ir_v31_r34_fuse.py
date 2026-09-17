"""ir_v31: late-fuse 4-ch R(2+1)D-34 into v29b IR+Thermal+MidFuse.

No thermal conf-gate grid. Extra MidFuse fourth-stream weight only if {8,9,24}
all non-decrease vs v29b. CSV only if nested_fixed > v29b and file_dis >= 2.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import V7_CFG, apply_cfg, nested_fixed, preds_full, softmax_np
from write_ir_v30_safe import (
    CK24, CK_V3, IR_AMIN, IR_MAXCH, IR_PMAX, MAX_DIS, ROOT, SAMPLE_SUB, TRACK,
    V29B_NESTED, evaluate_v30_hold, selective_swap,
)

PT = timezone(timedelta(hours=-7))
CK34 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v31"
V29B_CSV = ROOT / "submission_ir_v29b.csv"
MISSED_CLIP = "small_model_track_test/SM_test_0379/"
ALPHAS = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15)
# selective R34->IR (same helper as v29b T24 swap; not a thermal conf-gate)
R34_SWAPS = ((0.40, 0.55, 15), (0.35, 0.60, 10), (0.45, 0.50, 8), (0.50, 0.55, 5))
MID_WC = (None, 0.12, 0.15)  # None = keep V7_CFG


def _cfg_with_wc(wc):
    if wc is None:
        return dict(V7_CFG)
    rest = 1.0 - float(wc)
    s = V7_CFG["wa"] + V7_CFG["wb"]
    return {"wa": V7_CFG["wa"] / s * rest, "wb": V7_CFG["wb"] / s * rest, "wc": float(wc), "T": V7_CFG["T"]}


def v29b_fold_map(hold):
    return {int(f["leave"]): float(f["te_acc_v29b_reselect"]) for f in hold["folds"]}


def per_user_ok(nest, v29_folds):
    bits = {}
    all_ok = True
    for f in nest["folds"]:
        leave = int(f["leave"])
        ok = float(f["te_acc"]) + 1e-15 >= v29_folds[leave]
        bits[leave] = {"te_acc": float(f["te_acc"]), "v29b": v29_folds[leave], "ok": ok}
        all_ok = all_ok and ok
    return all_ok, bits


def greedy_safe_swap(ir, r34, th, mid, yt, yu, mask0, v29_folds, T, pmax, amin, maxch):
    """Add R34 swaps one by one only if all of {8,9,24} stay >= v29b."""
    pp, ap = softmax_np(ir, T), softmax_np(r34, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    pconf, aconf = pp.max(1), ap.max(1)
    cand = np.where((pa != aa) & (pconf <= pmax) & (aconf >= amin) & mask0)[0]
    order = cand[np.lexsort((-aconf[cand], pconf[cand]))]
    out = ir.copy()
    kept = []
    for i in order:
        if len(kept) >= maxch:
            break
        trial = out.copy()
        trial[i] = r34[i]
        nest = nested_fixed(trial, th, mid, yt, yu, mask0, V7_CFG)
        ok, _ = per_user_ok(nest, v29_folds)
        if ok and float(nest["mean"]) > V29B_NESTED + 1e-12:
            out = trial
            kept.append(int(i))
    return out, kept


def evaluate_v31_hold(verbose=True):
    hold = evaluate_v30_hold(verbose=False)
    r34_path = CK34 / "hold_logits.npy"
    if not r34_path.exists():
        raise FileNotFoundError(f"missing {r34_path}; train_r2p1d34_4ch.py first")
    r34_all_hold = np.load(r34_path).astype(np.float32)
    # hold tensors in evaluate_v30_hold are already IR-hold sized (505)
    if r34_all_hold.shape[0] != hold["ir_sw"].shape[0]:
        raise ValueError(f"r34 hold {r34_all_hold.shape} vs ir_sw {hold['ir_sw'].shape}")
    users_p = CK34 / "hold_users.npy"
    if users_p.exists():
        r34_u = np.load(users_p)
        if not np.array_equal(r34_u, hold["yu"]):
            raise ValueError("r34 hold_users order != late-fuse yu")
    v29_folds = v29b_fold_map(hold)
    v29_nest = hold["v29_nest"]
    mask0, yt, yu = hold["mask0"], hold["yt"], hold["yu"]
    ir, th, mid = hold["ir_sw"], hold["th0"], hold["mid"]
    v29_preds = hold["v29_preds"]

    trials = []
    best = None
    candidates = []
    for a in ALPHAS:
        candidates.append(("blend", a, None, ((1.0 - a) * ir + a * r34_all_hold).astype(np.float32)))
    T = V7_CFG["T"]
    for pmax, amin, maxch in R34_SWAPS:
        ir_b, ch = selective_swap(ir, r34_all_hold, T, pmax, amin, maxch)
        candidates.append(("swap", (pmax, amin, maxch, len(ch)), ch, ir_b.astype(np.float32)))
    for pmax, amin, maxch in R34_SWAPS:
        ir_b, kept = greedy_safe_swap(ir, r34_all_hold, th, mid, yt, yu, mask0, v29_folds, T, pmax, amin, maxch)
        candidates.append(("greedy", (pmax, amin, maxch, len(kept)), kept, ir_b.astype(np.float32)))

    for kind, spec, extra, ir_b in candidates:
        for wc in MID_WC:
            cfg = _cfg_with_wc(wc)
            nest = nested_fixed(ir_b, th, mid, yt, yu, mask0, cfg)
            full, _ = apply_cfg(ir_b, th, mid, yt, mask0, cfg)
            pred = preds_full(ir_b, th, mid, mask0, cfg)
            dis = int(((pred >= 0) & (v29_preds >= 0) & (pred != v29_preds)).sum())
            users_ok, bits = per_user_ok(nest, v29_folds)
            extra_mid = wc is not None
            meta = {"kind": kind, "spec": spec, "n_swap": None if extra is None else extra,
                    "cfg": cfg, "extra_mid": extra_mid,
                    "nested": float(nest["mean"]), "full": float(full), "dis": dis,
                    "users_ok": users_ok, "users": bits}
            if extra_mid and not users_ok:
                meta["skipped"] = "per_user_fail"
                trials.append(meta)
                continue
            clears = bool(
                float(nest["mean"]) > V29B_NESTED + 1e-12
                and users_ok
                and dis <= MAX_DIS
                and dis >= 1
            )
            row = dict(meta)
            row.update({"skipped": None, "clears": clears, "nest": nest, "ir_b": ir_b, "pred": pred})
            trials.append(meta)
            if best is None or (
                (clears and not best.get("clears"))
                or (clears == best.get("clears") and float(nest["mean"]) > best["nested"])
            ):
                best = row
            if verbose:
                print(
                    f"kind={kind} spec={spec} wc={cfg['wc']:.3f} extra_mid={extra_mid} "
                    f"nested={nest['mean']:.6f} dis={dis} users_ok={users_ok} clears={clears}",
                    flush=True,
                )

    if best is None:
        raise RuntimeError("no fuse trial")
    out = {
        "hold_v30": hold,
        "r34": r34_all_hold,
        "best": {k: v for k, v in best.items() if k not in ("ir_b", "pred", "nest")},
        "ir_b": best["ir_b"],
        "th": th,
        "mid": mid,
        "yt": yt,
        "yu": yu,
        "mask0": mask0,
        "pred": best["pred"],
        "nest": best["nest"],
        "v29_preds": v29_preds,
        "v29_nested": float(v29_nest["mean"]),
        "v29_folds": v29_folds,
        "trials": [t for t in trials if "ir_b" not in t or t.get("ir_b") is None],
        "clears_nested": bool(best.get("clears")),
    }
    if verbose:
        print("BEST", json.dumps(out["best"], indent=2), flush=True)
    return out


def write_v31_submission(hold=None, out_csv=None, verbose=True):
    if hold is None:
        hold = evaluate_v31_hold(verbose=verbose)
    out_csv = Path(out_csv) if out_csv is not None else (ROOT / "submission_ir_v31.csv")
    test_r34 = CK34 / "test_logits.npy"
    test_info = None
    wrote = False
    if not test_r34.exists():
        if verbose:
            print(f"missing {test_r34}", flush=True)
        return {"wrote": False, "csv": None, "out_csv": out_csv, "test": None, "hold": hold}

    T = V7_CFG["T"]
    kind = hold["best"]["kind"]
    spec = hold["best"]["spec"]
    cfg = hold["best"]["cfg"]
    ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
    t24_test = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
    th_test = np.load(CK_V3 / "test_logits.npy").astype(np.float32)
    r34_test = np.load(test_r34).astype(np.float32)
    ir_v29, _ = selective_swap(ir_test, t24_test, T, IR_PMAX, IR_AMIN, IR_MAXCH)
    if kind == "blend":
        a = float(spec)
        ir_b = ((1.0 - a) * ir_v29 + a * r34_test).astype(np.float32)
    else:
        pmax, amin, maxch, nkeep = spec
        ir_b, _ = selective_swap(ir_v29, r34_test, T, pmax, amin, int(nkeep) if kind == "greedy" else maxch)
    preds_v29 = (V7_CFG["wa"] * softmax_np(ir_v29, T) + V7_CFG["wb"] * softmax_np(th_test, T)
                 + V7_CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
    preds = (cfg["wa"] * softmax_np(ir_b, cfg["T"]) + cfg["wb"] * softmax_np(th_test, cfg["T"])
             + cfg["wc"] * softmax_np(mid_test, cfg["T"])).argmax(1)
    test_dis = int((preds != preds_v29).sum())
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
    v29_rows = list(csv.DictReader(open(V29B_CSV, encoding="utf-8")))
    new_rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    diffs = [i for i, (a_, b_) in enumerate(zip(v29_rows, new_rows)) if a_["prediction"] != b_["prediction"]]
    file_dis = len(diffs)
    only_missed = file_dis == 1 and new_rows[diffs[0]]["path"] == MISSED_CLIP
    test_info = {
        "disagree_vs_v29b": test_dis, "file_dis_vs_v29b": file_dis,
        "diff_paths": [new_rows[i]["path"] for i in diffs],
        "only_missed_0379": only_missed,
    }
    # greedy uses hold labels to skip harmful clips; that subset is not a test-time rule.
    label_free = kind in ("blend", "swap")
    submit_ok = bool(
        label_free
        and hold["clears_nested"]
        and file_dis >= 2
        and file_dis <= MAX_DIS
        and not only_missed
    )
    if submit_ok:
        shutil.copy2(out_csv, TRACK / "submission.csv")
        wrote = True
    if verbose:
        print(f"test file_dis={file_dis} only_0379={only_missed} submit_ok={submit_ok} wrote={wrote}", flush=True)
    return {"wrote": wrote, "csv": str(out_csv) if wrote else str(out_csv), "out_csv": out_csv,
            "test": test_info, "hold": hold, "submit_ok": submit_ok}


def deploy_pack_mb():
    pack = CK34 / "model_int8.pt"
    yolo = ROOT / "yolov8n.pt"
    n = 0
    if pack.exists():
        n += pack.stat().st_size
    if yolo.exists():
        n += yolo.stat().st_size
    return n / (1024 * 1024)


def main():
    hold = evaluate_v31_hold(verbose=True)
    result = write_v31_submission(hold, verbose=True)
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    report = {
        "tag": "ir_v31_r2p1d34_4ch",
        "updated_at": now,
        "best": hold["best"],
        "clears_nested": hold["clears_nested"],
        "submit_ok": result.get("submit_ok"),
        "wrote_csv": result["wrote"],
        "csv": str(result["out_csv"]),
        "test": result["test"],
        "deploy_pack_mb": deploy_pack_mb(),
        "note": "no thermal conf-gate; extra mid only if per-user non-decrease",
    }
    (ROOT / "metrics_ir_v31_status.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clears_nested", "submit_ok", "wrote_csv", "test", "deploy_pack_mb")}, indent=2, default=str), flush=True)
    return report


if __name__ == "__main__":
    main()
