"""ir_v32: label-free ranked-prefix or R34∩classic9 rejector on averaged 4ch R(2+1)D-34 seeds.

Never greedy/hold-subset. MidFuse wc locked at 0.09. No thermal conf-gate.
Submit only if users 8/24/9 floors hold, nested > v29b, file_dis >= 2, not only SM_test_0379.
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
CK42 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v31"
CK123 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v32_s123"
V29B_CSV = ROOT / "submission_ir_v29b.csv"
MISSED_CLIP = "small_model_track_test/SM_test_0379/"
# Plan printed 0.72956 / 0.75484; those are 5-dp of v29b leave-8/24 te_acc.
U8_MIN = 0.7295597484276729
U24_MIN = 0.7548387096774194
# ranked prefix grid (label-free). Exclude the v31 prefix-8 spec (0.45, 0.50, 8).
PREFIX_PMAX = (0.30, 0.35, 0.40, 0.45, 0.50)
PREFIX_AMIN = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
PREFIX_K = (1, 2, 3, 4, 5, 6, 8, 10)
BANNED_PREFIX = (0.45, 0.50, 8)
REJECT_AMIN = (0.40, 0.50, 0.55, 0.60, 0.65, 0.70)
CFG = dict(V7_CFG)  # wc=0.09 locked


def average_r34():
    h42 = np.load(CK42 / "hold_logits.npy").astype(np.float32)
    t42 = np.load(CK42 / "test_logits.npy").astype(np.float32)
    h123_p, t123_p = CK123 / "hold_logits.npy", CK123 / "test_logits.npy"
    if not h123_p.exists() or not t123_p.exists():
        raise FileNotFoundError(f"seed123 logits missing: {h123_p} {t123_p}")
    h123 = np.load(h123_p).astype(np.float32)
    t123 = np.load(t123_p).astype(np.float32)
    if h42.shape != h123.shape or t42.shape != t123.shape:
        raise ValueError(f"shape mismatch hold {h42.shape}/{h123.shape} test {t42.shape}/{t123.shape}")
    if np.allclose(h42, h123):
        raise ValueError("seed123 hold logits are a copy of seed42")
    hold_avg = (0.5 * h42 + 0.5 * h123).astype(np.float32)
    test_avg = (0.5 * t42 + 0.5 * t123).astype(np.float32)
    np.save(CK123 / "hold_logits_avg.npy", hold_avg)
    np.save(CK123 / "test_logits_avg.npy", test_avg)
    return hold_avg, test_avg, h42, h123, t42, t123


def per_user_bits(nest, v29_folds):
    bits = {}
    ok_all = True
    for f in nest["folds"]:
        leave = int(f["leave"])
        te = float(f["te_acc"])
        if leave == 8:
            ok = te + 1e-15 >= U8_MIN
        elif leave == 24:
            ok = te + 1e-15 >= U24_MIN
        else:
            ok = te + 1e-15 >= v29_folds[leave]
        bits[leave] = {"te_acc": te, "floor": U8_MIN if leave == 8 else (U24_MIN if leave == 24 else v29_folds[leave]), "ok": ok}
        ok_all = ok_all and ok
    return ok_all, bits


def ranked_prefix(primary, aux, T, pmax, amin, maxch):
    """Label-free: take the first maxch of the ranked candidate list as a whole."""
    return selective_swap(primary, aux, T, pmax, amin, maxch)


def rejector_agree_vs_third(c9, r34, third, T, amin):
    """Where R34 and classic9 agree on C and third disagrees, replace third with classic9."""
    pc, pr, pt = softmax_np(c9, T), softmax_np(r34, T), softmax_np(third, T)
    cc, cr, ct = pc.argmax(1), pr.argmax(1), pt.argmax(1)
    mask = (cc == cr) & (cc != ct) & (pc.max(1) >= amin) & (pr.max(1) >= amin)
    out = third.copy()
    idx = np.where(mask)[0]
    for i in idx:
        out[i] = c9[i]
    return out, idx.tolist()


def rejector_revert_ir(ir_sw, c9, r34, T, amin):
    """Where R34 and classic9 agree and ir_sw (e.g. t24 override) disagrees, restore mean(c9,r34)."""
    pc, pr, pi = softmax_np(c9, T), softmax_np(r34, T), softmax_np(ir_sw, T)
    cc, cr, ci = pc.argmax(1), pr.argmax(1), pi.argmax(1)
    mask = (cc == cr) & (ci != cc) & (pc.max(1) >= amin) & (pr.max(1) >= amin)
    out = ir_sw.copy()
    idx = np.where(mask)[0]
    for i in idx:
        out[i] = 0.5 * (c9[i] + r34[i])
    return out, idx.tolist()


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
        and dis >= 1
        and abs(CFG["wc"] - 0.09) < 1e-9
    )
    return nest, full, pred, dis, users_ok, bits, clears


def evaluate_v32_hold(verbose=True):
    hold = evaluate_v30_hold(verbose=False)
    r34, r34_test, h42, h123, t42, t123 = average_r34()
    if r34.shape[0] != hold["ir_sw"].shape[0]:
        raise ValueError(f"r34 hold {r34.shape} vs ir {hold['ir_sw'].shape}")
    users_p = CK42 / "hold_users.npy"
    if users_p.exists() and not np.array_equal(np.load(users_p), hold["yu"]):
        raise ValueError("r34 hold_users order != late-fuse yu")
    v29_folds = {int(f["leave"]): float(f["te_acc_v29b_reselect"]) for f in hold["folds"]}
    mask0, yt, yu = hold["mask0"], hold["yt"], hold["yu"]
    ir, th, mid, c9 = hold["ir_sw"], hold["th0"], hold["mid"], hold["classic9"]
    v29_preds = hold["v29_preds"]
    T = CFG["T"]
    trials = []
    best = None

    def consider(kind, spec, ir_b, th_b, mid_b, nch):
        nonlocal best
        nest, full, pred, dis, users_ok, bits, clears = _score(ir_b, th_b, mid_b, yt, yu, mask0, v29_preds, v29_folds)
        row = {
            "kind": kind, "spec": spec, "n": nch, "cfg": dict(CFG),
            "nested": float(nest["mean"]), "full": float(full), "dis": dis,
            "users_ok": users_ok, "users": bits, "clears": clears,
            "ir_b": ir_b, "th_b": th_b, "mid_b": mid_b, "pred": pred, "nest": nest,
        }
        trials.append({k: v for k, v in row.items() if k not in ("ir_b", "th_b", "mid_b", "pred", "nest")})
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
                if (float(pmax), float(amin), int(k)) == BANNED_PREFIX:
                    continue
                ir_b, ch = ranked_prefix(ir, r34, T, pmax, amin, k)
                consider("prefix", (float(pmax), float(amin), int(k)), ir_b, th, mid, len(ch))

    for amin in REJECT_AMIN:
        th_b, ch = rejector_agree_vs_third(c9, r34, th, T, amin)
        consider("rejector_th", float(amin), ir, th_b, mid, len(ch))
        mid_b, ch = rejector_agree_vs_third(c9, r34, mid, T, amin)
        consider("rejector_mid", float(amin), ir, th, mid_b, len(ch))
        ir_b, ch = rejector_revert_ir(ir, c9, r34, T, amin)
        consider("rejector_ir", float(amin), ir_b, th, mid, len(ch))

    if best is None:
        raise RuntimeError("no v32 trial")
    out = {
        "hold_v30": hold,
        "r34": r34,
        "r34_test": r34_test,
        "h42": h42, "h123": h123,
        "best": {k: v for k, v in best.items() if k not in ("ir_b", "th_b", "mid_b", "pred", "nest")},
        "ir_b": best["ir_b"], "th_b": best["th_b"], "mid_b": best["mid_b"],
        "yt": yt, "yu": yu, "mask0": mask0, "pred": best["pred"], "nest": best["nest"],
        "v29_preds": v29_preds, "v29_folds": v29_folds,
        "trials_n": len(trials),
        "clears_nested": bool(best.get("clears")),
        "label_free": best["kind"] in ("prefix", "rejector_th", "rejector_mid", "rejector_ir"),
        "wc": CFG["wc"],
    }
    if verbose:
        print("BEST", json.dumps(out["best"], indent=2, default=str), flush=True)
    return out


def write_v32_submission(hold=None, out_csv=None, verbose=True):
    if hold is None:
        hold = evaluate_v32_hold(verbose=verbose)
    out_csv = Path(out_csv) if out_csv is not None else (ROOT / "submission_ir_v32.csv")
    T = CFG["T"]
    kind, spec = hold["best"]["kind"], hold["best"]["spec"]
    r34_test = hold["r34_test"]
    ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
    t24_test = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
    th_test = np.load(CK_V3 / "test_logits.npy").astype(np.float32)
    ir_v29, _ = selective_swap(ir_test, t24_test, T, IR_PMAX, IR_AMIN, IR_MAXCH)
    c9_test = ir_test
    ir_b, th_b, mid_b = ir_v29, th_test, mid_test
    if kind == "prefix":
        pmax, amin, k = spec
        ir_b, _ = ranked_prefix(ir_v29, r34_test, T, pmax, amin, k)
    elif kind == "rejector_th":
        th_b, _ = rejector_agree_vs_third(c9_test, r34_test, th_test, T, float(spec))
    elif kind == "rejector_mid":
        mid_b, _ = rejector_agree_vs_third(c9_test, r34_test, mid_test, T, float(spec))
    elif kind == "rejector_ir":
        ir_b, _ = rejector_revert_ir(ir_v29, c9_test, r34_test, T, float(spec))
    else:
        raise ValueError(f"non label-free kind {kind}")

    preds_v29 = (CFG["wa"] * softmax_np(ir_v29, T) + CFG["wb"] * softmax_np(th_test, T)
                 + CFG["wc"] * softmax_np(mid_test, T)).argmax(1)
    preds = (CFG["wa"] * softmax_np(ir_b, T) + CFG["wb"] * softmax_np(th_b, T)
             + CFG["wc"] * softmax_np(mid_b, T)).argmax(1)
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
        and kind != "greedy"
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
    pack = CK123 / "model_int8.pt"
    if not pack.exists():
        pack = CK42 / "model_int8.pt"
    yolo = ROOT / "yolov8n.pt"
    n = (pack.stat().st_size if pack.exists() else 0) + (yolo.stat().st_size if yolo.exists() else 0)
    return n / (1024 * 1024)


def main():
    hold = evaluate_v32_hold(verbose=True)
    result = write_v32_submission(hold, verbose=True)
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    report = {
        "tag": "ir_v32_prefix_rejector",
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
        "note": "no greedy; no prefix-8 (0.45,0.50,8); no extra mid; no thermal conf-gate",
    }
    (ROOT / "metrics_ir_v32_status.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("clears_nested", "label_free", "submit_ok", "wrote_csv", "test", "deploy_pack_mb")}, indent=2, default=str), flush=True)
    return report


if __name__ == "__main__":
    main()
