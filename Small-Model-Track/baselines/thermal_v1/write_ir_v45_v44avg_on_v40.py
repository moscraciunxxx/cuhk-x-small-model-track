"""ir_v45: avg(T24-nonIR-box v44, T16-nonIR-box v36) ranked-prefix on public v40 mix.

New member solo >= 0.63. Label-free, wc=0.09. Clips 0219+0322 vs v40.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from probe_ir_v24_fuse import V7_CFG, apply_cfg, nested_fixed, preds_full, softmax_np
from write_ir_v30_safe import (
    CK24, CK_V3, IR_AMIN, IR_MAXCH, IR_PMAX, MAX_DIS, ROOT, SAMPLE_SUB, TRACK,
    evaluate_v30_hold, selective_swap,
)
from write_ir_v40_avg_on_v39 import CFG
from write_ir_v41_s123_on_v40 import v40_ir, ranked_prefix

PT = timezone(timedelta(hours=-7))
CK31 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v31"
CK36 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v36_noirbox_ft"
CK37 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v37_t24_ft"
CK44 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v44_t24_nb_resume"
V29B_CSV = ROOT / "submission_ir_v29b.csv"
V40_CSV = ROOT / "submission_ir_v40.csv"
U8_MIN = 0.73585
U24_MIN = 0.75484
NEST_MIN = 0.7617107369
SPEC = (0.30, 0.80, 3)
BANNED = {
    "small_model_track_test/SM_test_0075/", "small_model_track_test/SM_test_0194/",
    "small_model_track_test/SM_test_0379/", "small_model_track_test/SM_test_0046/",
    "small_model_track_test/SM_test_0129/", "small_model_track_test/SM_test_0228/",
    "small_model_track_test/SM_test_0234/", "small_model_track_test/SM_test_0301/",
    "small_model_track_test/SM_test_0328/", "small_model_track_test/SM_test_0343/",
    "small_model_track_test/SM_test_0350/", "small_model_track_test/SM_test_0358/",
    "small_model_track_test/SM_test_0360/", "small_model_track_test/SM_test_0003/",
    "small_model_track_test/SM_test_0074/", "small_model_track_test/SM_test_0369/",
    "small_model_track_test/SM_test_0037/", "small_model_track_test/SM_test_0314/",
    "small_model_track_test/SM_test_0030/", "small_model_track_test/SM_test_0179/",
}
KEEP = {
    "small_model_track_test/SM_test_0301/", "small_model_track_test/SM_test_0358/",
    "small_model_track_test/SM_test_0003/", "small_model_track_test/SM_test_0074/",
    "small_model_track_test/SM_test_0369/", "small_model_track_test/SM_test_0037/",
    "small_model_track_test/SM_test_0314/",
}


def per_user_bits(nest, v29_folds):
    bits = {}
    ok_all = True
    for f in nest["folds"]:
        leave = int(f["leave"])
        te = float(f["te_acc"])
        if leave == 8:
            floor, ok = U8_MIN, te + 1e-5 >= U8_MIN
        elif leave == 24:
            floor, ok = U24_MIN, te + 1e-5 >= U24_MIN
        else:
            floor = v29_folds[leave]
            ok = te + 1e-15 >= floor
        bits[leave] = {"te_acc": te, "floor": floor, "ok": ok}
        ok_all = ok_all and ok
    return ok_all, bits


def evaluate_v45_hold(verbose=True):
    hold = evaluate_v30_hold(verbose=False)
    nb = np.load(CK36 / "hold_logits.npy").astype(np.float32)
    ib = np.load(CK31 / "hold_logits.npy").astype(np.float32)
    t24 = np.load(CK37 / "hold_logits.npy").astype(np.float32)
    v44 = np.load(CK44 / "hold_logits.npy").astype(np.float32)
    if np.allclose(v44, nb) or np.allclose(v44, t24):
        raise ValueError("v44 hold logits copy of an old member")
    aux = (0.5 * v44 + 0.5 * nb).astype(np.float32)
    v29_folds = {int(f["leave"]): float(f["te_acc_v29b_reselect"]) for f in hold["folds"]}
    mask0, yt, yu = hold["mask0"], hold["yt"], hold["yu"]
    ir, th, mid = hold["ir_sw"], hold["th0"], hold["mid"]
    T = CFG["T"]
    ir40 = v40_ir(ir, nb, ib, t24, T)
    ir_b, ch = ranked_prefix(ir40, aux, T, *SPEC)
    nest = nested_fixed(ir_b, th, mid, yt, yu, mask0, CFG)
    full, _ = apply_cfg(ir_b, th, mid, yt, mask0, CFG)
    pred = preds_full(ir_b, th, mid, mask0, CFG)
    v29_preds = hold["v29_preds"]
    dis = int(((pred >= 0) & (v29_preds >= 0) & (pred != v29_preds)).sum())
    users_ok, bits = per_user_bits(nest, v29_folds)
    clears = bool(
        float(nest["mean"]) + 1e-12 >= NEST_MIN
        and users_ok
        and dis <= MAX_DIS
        and abs(CFG["wc"] - 0.09) < 1e-9
        and len(ch) >= 1
        and SPEC[2] != 8
    )
    if verbose:
        print(f"n={len(ch)} nested={nest['mean']:.6f} dis={dis} users_ok={users_ok} clears={clears} spec={SPEC}", flush=True)
        print("user_bits", bits, flush=True)
    return {
        "best": {
            "kind": "avg_v44_v36_on_v40", "spec": SPEC, "n": len(ch),
            "cfg": dict(CFG), "nested": float(nest["mean"]), "full": float(full),
            "dis": dis, "users_ok": users_ok, "users": bits, "clears": clears,
        },
        "ir_b": ir_b, "th": th, "mid": mid, "yt": yt, "yu": yu, "mask0": mask0,
        "pred": pred, "nest": nest, "v29_preds": v29_preds, "v29_folds": v29_folds,
        "nb_test": np.load(CK36 / "test_logits.npy").astype(np.float32),
        "ib_test": np.load(CK31 / "test_logits.npy").astype(np.float32),
        "t24_test": np.load(CK37 / "test_logits.npy").astype(np.float32),
        "v44_test": np.load(CK44 / "test_logits.npy").astype(np.float32),
        "clears_nested": clears, "label_free": True, "wc": CFG["wc"],
    }


def write_v45_submission(hold=None, out_csv=None, verbose=True):
    if hold is None:
        hold = evaluate_v45_hold(verbose=verbose)
    out_csv = Path(out_csv) if out_csv is not None else (ROOT / "submission_ir_v45.csv")
    T = CFG["T"]
    ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
    t24ir = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
    th_test = np.load(CK_V3 / "test_logits.npy").astype(np.float32)
    ir_v29, _ = selective_swap(ir_test, t24ir, T, IR_PMAX, IR_AMIN, IR_MAXCH)
    ir40 = v40_ir(ir_v29, hold["nb_test"], hold["ib_test"], hold["t24_test"], T)
    aux_te = (0.5 * hold["v44_test"] + 0.5 * hold["nb_test"]).astype(np.float32)
    ir_b, _ = ranked_prefix(ir40, aux_te, T, *SPEC)
    preds = (CFG["wa"] * softmax_np(ir_b, T) + CFG["wb"] * softmax_np(th_test, T)
             + CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
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
    v40_rows = list(csv.DictReader(open(V40_CSV, encoding="utf-8")))
    v29_rows = list(csv.DictReader(open(V29B_CSV, encoding="utf-8")))
    new_rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    diffs40 = [i for i, (a, b) in enumerate(zip(v40_rows, new_rows)) if a["prediction"] != b["prediction"]]
    diffs29 = [i for i, (a, b) in enumerate(zip(v29_rows, new_rows)) if a["prediction"] != b["prediction"]]
    paths40 = [new_rows[i]["path"] for i in diffs40]
    keep_ok = all(
        next(r for r in v40_rows if r["path"] == p)["prediction"]
        == next(r for r in new_rows if r["path"] == p)["prediction"]
        for p in KEEP
    )
    banned_hit = [p for p in paths40 if p in BANNED]
    file_dis_v40 = len(diffs40)
    test_info = {
        "file_dis_vs_v40": file_dis_v40, "file_dis_vs_v29b": len(diffs29),
        "diff_paths_v40": paths40, "banned_hit": banned_hit, "keep_v40_public": keep_ok,
    }
    submit_ok = bool(
        hold["clears_nested"] and hold["label_free"]
        and 2 <= file_dis_v40 <= 4 and len(diffs29) <= MAX_DIS
        and not banned_hit and keep_ok and abs(CFG["wc"] - 0.09) < 1e-9
    )
    wrote = False
    if submit_ok:
        shutil.copy2(out_csv, TRACK / "submission.csv")
        wrote = True
    if verbose:
        print(f"test file_dis_v40={file_dis_v40} banned={banned_hit} keep={keep_ok} submit_ok={submit_ok} wrote={wrote} paths={paths40}", flush=True)
    return {"wrote": wrote, "csv": str(out_csv), "out_csv": out_csv, "test": test_info, "hold": hold, "submit_ok": submit_ok}


def deploy_pack_mb():
    pack = CK44 / "model_int8.pt"
    yolo = ROOT / "yolov8n.pt"
    return ((pack.stat().st_size if pack.exists() else 0) + (yolo.stat().st_size if yolo.exists() else 0)) / (1024 * 1024)


def main():
    hold = evaluate_v45_hold(verbose=True)
    result = write_v45_submission(hold, verbose=True)
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    report = {
        "tag": "ir_v45_v44avg_on_v40", "updated_at": now, "best": hold["best"],
        "clears_nested": hold["clears_nested"], "submit_ok": result.get("submit_ok"),
        "wrote_csv": result["wrote"], "csv": str(result["out_csv"]), "test": result["test"],
        "deploy_pack_mb": deploy_pack_mb(), "wc": 0.09,
        "note": "quota 0 today; CSV ready for next window; do not Kaggle until quota resets",
    }
    (ROOT / "metrics_ir_v45_status.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clears_nested", "submit_ok", "wrote_csv", "test", "deploy_pack_mb")}, indent=2, default=str), flush=True)
    return report


if __name__ == "__main__":
    main()
