"""ir_v33: label-free ranked-prefix of T24 4ch R(2+1)D-34 (new view, not old T16 ranking).

wc=0.09. No greedy, no extra mid, no thermal conf-gate, no 0-diff submit.
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
    V29B_NESTED, evaluate_v30_hold, selective_swap,
)

PT = timezone(timedelta(hours=-7))
CK33 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v33_t24"
CK42 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v31"
CK123 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v32_s123"
V29B_CSV = ROOT / "submission_ir_v29b.csv"
MISSED_CLIP = "small_model_track_test/SM_test_0379/"
U8_MIN = 0.7295597484276729
U24_MIN = 0.7548387096774194
PREFIX_PMAX = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
PREFIX_AMIN = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
PREFIX_K = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)
BANNED_OLD_PREFIX8 = (0.45, 0.50, 8)
CFG = dict(V7_CFG)


def load_t24():
    h = CK33 / "hold_logits.npy"
    t = CK33 / "test_logits.npy"
    if not h.exists() or not t.exists():
        raise FileNotFoundError(f"T24 logits missing: {h} {t}")
    hold = np.load(h).astype(np.float32)
    test = np.load(t).astype(np.float32)
    if hold.shape[1] != 40 or test.shape != (405, 40):
        raise ValueError(f"bad T24 shapes hold={hold.shape} test={test.shape}")
    for other in (CK42 / "hold_logits.npy", CK123 / "hold_logits.npy"):
        if other.exists() and np.allclose(hold, np.load(other)):
            raise ValueError(f"T24 hold logits copy of {other}")
    return hold, test


def per_user_bits(nest, v29_folds):
    bits = {}
    ok_all = True
    for f in nest["folds"]:
        leave = int(f["leave"])
        te = float(f["te_acc"])
        if leave == 8:
            floor, ok = U8_MIN, te + 1e-15 >= U8_MIN
        elif leave == 24:
            floor, ok = U24_MIN, te + 1e-15 >= U24_MIN
        else:
            floor = v29_folds[leave]
            ok = te + 1e-15 >= floor
        bits[leave] = {"te_acc": te, "floor": floor, "ok": ok}
        ok_all = ok_all and ok
    return ok_all, bits


def ranked_prefix(primary, aux, T, pmax, amin, maxch):
    return selective_swap(primary, aux, T, pmax, amin, maxch)


def _score(ir, th, mid, yt, yu, mask0, v29_preds, v29_folds):
    nest = nested_fixed(ir, th, mid, yt, yu, mask0, CFG)
    full, _ = apply_cfg(ir, th, mid, yt, mask0, CFG)
    pred = preds_full(ir, th, mid, mask0, CFG)
    dis = int(((pred >= 0) & (v29_preds >= 0) & (pred != v29_preds)).sum())
    users_ok, bits = per_user_bits(nest, v29_folds)
    clears = bool(
        float(nest["mean"]) > V29B_NESTED + 1e-12
        and users_ok
        and dis <= MAX_DIS
        and dis >= 2
        and abs(CFG["wc"] - 0.09) < 1e-9
    )
    return nest, full, pred, dis, users_ok, bits, clears


def evaluate_v33_hold(verbose=True):
    hold = evaluate_v30_hold(verbose=False)
    r34, r34_test = load_t24()
    if r34.shape[0] != hold["ir_sw"].shape[0]:
        raise ValueError(f"r34 hold {r34.shape} vs ir {hold['ir_sw'].shape}")
    users_p = CK33 / "hold_users.npy"
    if users_p.exists() and not np.array_equal(np.load(users_p), hold["yu"]):
        raise ValueError("T24 hold_users order != late-fuse yu")
    v29_folds = {int(f["leave"]): float(f["te_acc_v29b_reselect"]) for f in hold["folds"]}
    mask0, yt, yu = hold["mask0"], hold["yt"], hold["yu"]
    ir, th, mid = hold["ir_sw"], hold["th0"], hold["mid"]
    v29_preds = hold["v29_preds"]
    T = CFG["T"]
    best = None

    def consider(kind, spec, ir_b, nch):
        nonlocal best
        nest, full, pred, dis, users_ok, bits, clears = _score(
            ir_b, th, mid, yt, yu, mask0, v29_preds, v29_folds
        )
        row = {
            "kind": kind, "spec": spec, "n": nch, "cfg": dict(CFG),
            "nested": float(nest["mean"]), "full": float(full), "dis": dis,
            "users_ok": users_ok, "users": bits, "clears": clears,
            "ir_b": ir_b, "pred": pred, "nest": nest,
        }
        if verbose:
            print(
                f"kind={kind} spec={spec} n={nch} nested={nest['mean']:.6f} dis={dis} "
                f"users_ok={users_ok} clears={clears}",
                flush=True,
            )
        if best is None or (clears and not best.get("clears")) or (
            clears == best.get("clears") and float(nest["mean"]) > best["nested"]
        ):
            best = row

    for pmax in PREFIX_PMAX:
        for amin in PREFIX_AMIN:
            for k in PREFIX_K:
                if (float(pmax), float(amin), int(k)) == BANNED_OLD_PREFIX8:
                    continue
                ir_b, ch = ranked_prefix(ir, r34, T, pmax, amin, k)
                consider("prefix", (float(pmax), float(amin), int(k)), ir_b, len(ch))

    if best is None:
        raise RuntimeError("no v33 trial")
    out = {
        "best": {k: v for k, v in best.items() if k not in ("ir_b", "pred", "nest")},
        "ir_b": best["ir_b"], "th": th, "mid": mid,
        "yt": yt, "yu": yu, "mask0": mask0, "pred": best["pred"], "nest": best["nest"],
        "v29_preds": v29_preds, "v29_folds": v29_folds, "r34_test": r34_test,
        "clears_nested": bool(best.get("clears")),
        "label_free": best["kind"] == "prefix",
        "wc": CFG["wc"],
    }
    if verbose:
        print("BEST", json.dumps(out["best"], indent=2, default=str), flush=True)
    return out


def write_v33_submission(hold=None, out_csv=None, verbose=True):
    if hold is None:
        hold = evaluate_v33_hold(verbose=verbose)
    out_csv = Path(out_csv) if out_csv is not None else (ROOT / "submission_ir_v33.csv")
    T = CFG["T"]
    kind, spec = hold["best"]["kind"], hold["best"]["spec"]
    if kind != "prefix":
        raise ValueError(f"v33 ships ranked prefix only, got {kind}")
    pmax, amin, k = spec
    ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
    t24_test = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
    th_test = np.load(CK_V3 / "test_logits.npy").astype(np.float32)
    ir_v29, _ = selective_swap(ir_test, t24_test, T, IR_PMAX, IR_AMIN, IR_MAXCH)
    ir_b, _ = ranked_prefix(ir_v29, hold["r34_test"], T, pmax, amin, k)
    preds_v29 = (CFG["wa"] * softmax_np(ir_v29, T) + CFG["wb"] * softmax_np(th_test, T)
                 + CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
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
    v29_rows = list(csv.DictReader(open(V29B_CSV, encoding="utf-8")))
    new_rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    diffs = [i for i, (a, b) in enumerate(zip(v29_rows, new_rows)) if a["prediction"] != b["prediction"]]
    file_dis = len(diffs)
    only_missed = file_dis == 1 and new_rows[diffs[0]]["path"] == MISSED_CLIP
    test_info = {
        "disagree_vs_v29b": int((preds != preds_v29).sum()),
        "file_dis_vs_v29b": file_dis,
        "diff_paths": [new_rows[i]["path"] for i in diffs],
        "only_missed_0379": only_missed,
    }
    submit_ok = bool(
        hold["clears_nested"]
        and hold["label_free"]
        and file_dis >= 2
        and file_dis <= MAX_DIS
        and not only_missed
        and abs(CFG["wc"] - 0.09) < 1e-9
    )
    wrote = False
    if submit_ok:
        shutil.copy2(out_csv, TRACK / "submission.csv")
        wrote = True
    if verbose:
        print(f"test file_dis={file_dis} only_0379={only_missed} submit_ok={submit_ok} wrote={wrote}", flush=True)
    return {"wrote": wrote, "csv": str(out_csv), "out_csv": out_csv, "test": test_info,
            "hold": hold, "submit_ok": submit_ok}


def deploy_pack_mb():
    pack = CK33 / "model_int8.pt"
    yolo = ROOT / "yolov8n.pt"
    n = (pack.stat().st_size if pack.exists() else 0) + (yolo.stat().st_size if yolo.exists() else 0)
    return n / (1024 * 1024)


def main():
    hold = evaluate_v33_hold(verbose=True)
    result = write_v33_submission(hold, verbose=True)
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    report = {
        "tag": "ir_v33_t24_prefix",
        "updated_at": now,
        "best": hold["best"],
        "clears_nested": hold["clears_nested"],
        "label_free": hold["label_free"],
        "submit_ok": result.get("submit_ok"),
        "wrote_csv": result["wrote"],
        "csv": str(result["out_csv"]),
        "test": result["test"],
        "deploy_pack_mb": deploy_pack_mb(),
        "wc": 0.09,
        "note": "T24 4ch new view; ranked prefix; no greedy; no extra mid; no 0-diff submit",
    }
    (ROOT / "metrics_ir_v33_status.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clears_nested", "label_free", "submit_ok", "wrote_csv", "test", "deploy_pack_mb")}, indent=2, default=str), flush=True)
    return report


if __name__ == "__main__":
    main()
