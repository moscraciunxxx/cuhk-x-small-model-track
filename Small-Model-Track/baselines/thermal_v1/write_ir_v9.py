"""Rewrite submission_ir_v9.csv from best nested-honest clear-win candidate."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, write_sub, V7, CLEAR

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")


def main():
    cache = ROOT / "cache" / "ir_yolo_v4"
    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    members, yt, yu = load_members()
    th = np.load(old / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")
    from dataset import DEFAULT_HOLD_OUT_USERS
    tu = np.load(cache / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid_h = mid[hold_idx]
    mask = th.any(1) & mid_h.any(1)
    stack = np.stack([m["logits"] for m in members], 0)
    w_acc = np.array([max(m["acc"], 1e-3) for m in members], float)
    w_acc /= w_acc.sum()
    members9 = [m for m in members if m["tag"] != "pool_seed55"]

    # Candidates from metrics (clear full >= 0.755)
    # Prefer highest nested among clear full wins
    top5 = np.mean(stack[:5], 0)
    all_sel = np.mean(stack, 0)
    all_accw = np.tensordot(w_acc, stack, axes=(0, 0))

    cands = [
        {
            "name": "perT_top5_sel",
            "mode": "perT",
            "ens": "top5_sel",
            "hold_logits": top5,
            "cfg": {"wa": 0.375, "wb": 0.25, "wc": 0.375, "Ta": 0.75, "Tb": 0.75, "Tc": 2.0},
            "member_slice": members[:5],
        },
        {
            "name": "sameT_all_sel",
            "mode": "sameT",
            "ens": "all_sel",
            "hold_logits": all_sel,
            "cfg": None,  # fill from metrics
            "member_slice": members,
        },
        {
            "name": "sameT_all_acc_w_sel",
            "mode": "sameT",
            "ens": "all_acc_w_sel",
            "hold_logits": all_accw,
            "cfg": None,
            "member_slice": members,
        },
        {
            "name": "geom_top5_sel",
            "mode": "geom",
            "ens": "top5_sel",
            "hold_logits": top5,
            "cfg": {"wa": 0.575, "wb": 0.275, "wc": 0.15, "T": 0.5},
            "member_slice": members[:5],
        },
        {
            "name": "sameT_top5_sel",
            "mode": "sameT",
            "ens": "top5_sel",
            "hold_logits": top5,
            "cfg": None,
            "member_slice": members[:5],
        },
    ]
    rep = json.loads((ROOT / "metrics_ir_v9.json").read_text(encoding="utf-8"))
    by_ens = {r["ens"]: r for r in rep["results"]}
    for c in cands:
        r = by_ens[c["ens"]]
        if c["cfg"] is None:
            if c["mode"] == "sameT":
                c["cfg"] = {k: r["triple"][k] for k in ("wa", "wb", "wc", "T")}
            elif c["mode"] == "geom":
                c["cfg"] = {k: r["geom"][k] for k in ("wa", "wb", "wc", "T")}
            elif c["mode"] == "perT":
                c["cfg"] = {k: r["perT"][k] for k in ("wa", "wb", "wc", "Ta", "Tb", "Tc")}

    def score_hold(c):
        el, cfg, mode = c["hold_logits"], c["cfg"], c["mode"]
        if mode == "sameT":
            T = cfg["T"]
            pred = (
                cfg["wa"] * softmax_np(el[mask], T)
                + cfg["wb"] * softmax_np(th[mask], T)
                + cfg["wc"] * softmax_np(mid_h[mask], T)
            ).argmax(1)
            full = float((pred == yt[mask]).mean())
            nest = nested_fixed(el, th, mid_h, yt, yu, mask, cfg)["mean"]
        elif mode == "geom":
            T = cfg["T"]
            eps = 1e-8
            folds = []
            # full
            la = np.log(np.clip(softmax_np(el[mask], T), eps, 1))
            lb = np.log(np.clip(softmax_np(th[mask], T), eps, 1))
            lc = np.log(np.clip(softmax_np(mid_h[mask], T), eps, 1))
            full = float(((cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1) == yt[mask]).mean())
            for leave in (8, 9, 24):
                te = mask & (yu == leave)
                la = np.log(np.clip(softmax_np(el[te], T), eps, 1))
                lb = np.log(np.clip(softmax_np(th[te], T), eps, 1))
                lc = np.log(np.clip(softmax_np(mid_h[te], T), eps, 1))
                pred = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
                folds.append(float((pred == yt[te]).mean()))
            nest = float(np.mean(folds))
        else:  # perT
            pred = (
                cfg["wa"] * softmax_np(el[mask], cfg["Ta"])
                + cfg["wb"] * softmax_np(th[mask], cfg["Tb"])
                + cfg["wc"] * softmax_np(mid_h[mask], cfg["Tc"])
            ).argmax(1)
            full = float((pred == yt[mask]).mean())
            folds = []
            for leave in (8, 9, 24):
                te = mask & (yu == leave)
                p = (
                    cfg["wa"] * softmax_np(el[te], cfg["Ta"])
                    + cfg["wb"] * softmax_np(th[te], cfg["Tb"])
                    + cfg["wc"] * softmax_np(mid_h[te], cfg["Tc"])
                ).argmax(1)
                folds.append(float((p == yt[te]).mean()))
            nest = float(np.mean(folds))
        c["full"] = full
        c["nested"] = nest
        return c

    for c in cands:
        score_hold(c)
        print(f"{c['name']}: full={c['full']:.6f} nested={c['nested']:.6f} cfg={c['cfg']}", flush=True)

    # choose: clear full (>=V7+CLEAR) and max nested; tie-break simpler sameT
    clear = [c for c in cands if c["full"] >= V7 + CLEAR - 1e-9]
    if not clear:
        clear = [c for c in cands if c["full"] > V7 + 1e-9]
    clear.sort(key=lambda c: (-c["nested"], 0 if c["mode"] == "sameT" else 1, -c["full"]))
    best = clear[0]
    print(f"\nCHOSEN {best['name']} full={best['full']:.6f} nested={best['nested']:.6f}", flush=True)

    # build test IR from member_slice selective test logits
    ir_test = np.mean([m["test_logits"] for m in best["member_slice"]], 0).astype(np.float32)
    if best["ens"] == "all_acc_w_sel":
        ir_test = np.tensordot(w_acc, np.stack([m["test_logits"] for m in members], 0), axes=(0, 0)).astype(np.float32)
    tags = [m["tag"] for m in best["member_slice"]]
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")
    cfg = best["cfg"]
    mode = best["mode"]
    if mode == "sameT":
        T = cfg["T"]
        preds = (
            cfg["wa"] * softmax_np(ir_test, T)
            + cfg["wb"] * softmax_np(th_test, T)
            + cfg["wc"] * softmax_np(mid_test, T)
        ).argmax(1)
    elif mode == "geom":
        T = cfg["T"]
        eps = 1e-8
        la = np.log(np.clip(softmax_np(ir_test, T), eps, 1))
        lb = np.log(np.clip(softmax_np(th_test, T), eps, 1))
        lc = np.log(np.clip(softmax_np(mid_test, T), eps, 1))
        preds = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        preds = (
            cfg["wa"] * softmax_np(ir_test, cfg["Ta"])
            + cfg["wb"] * softmax_np(th_test, cfg["Tb"])
            + cfg["wc"] * softmax_np(mid_test, cfg["Tc"])
        ).argmax(1)

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    out = ROOT / "submission_ir_v9.csv"
    nfb = write_sub(out, meta, preds, empty, fb)
    v7p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8"))]
    disagree = int(sum(int(a) != int(b) for a, b in zip(preds, v7p)))

    # also write safer sameT all_sel as alt if different
    alt = [c for c in cands if c["name"] == "sameT_all_sel"][0]
    if alt["name"] != best["name"]:
        ir_alt = np.mean([m["test_logits"] for m in members], 0).astype(np.float32)
        Ta = alt["cfg"]["T"]
        preds_alt = (
            alt["cfg"]["wa"] * softmax_np(ir_alt, Ta)
            + alt["cfg"]["wb"] * softmax_np(th_test, Ta)
            + alt["cfg"]["wc"] * softmax_np(mid_test, Ta)
        ).argmax(1)
        out_alt = ROOT / "submission_ir_v9_sameT.csv"
        write_sub(out_alt, meta, preds_alt, empty, fb)
        disagree_alt = int(sum(int(a) != int(b) for a, b in zip(preds_alt, v7p)))
    else:
        out_alt, disagree_alt = None, None

    fp16 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "model_fp16.pt"
    yolo = ROOT / "yolov8n.pt"
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = yolo.stat().st_size / (1024 * 1024)

    report = {
        "tag": "ir_v9",
        "primary": "submission_ir_v9.csv",
        "method": f"selective-TTA IR ({best['ens']}) + Thermal + Mid; {mode} fuse",
        "holdout_acc": best["full"],
        "nested_fixed_cfg": best["nested"],
        "nested_v7": 0.7519623092355898,
        "cfg": cfg,
        "mode": mode,
        "ens": best["ens"],
        "ir_tags": tags,
        "ir_tta_flags": {m["tag"]: m["use_tta"] for m in best["member_slice"]},
        "delta_vs_v7": float(best["full"] - V7),
        "clear_win": bool(best["full"] >= V7 + CLEAR - 1e-9),
        "disagree_vs_v7": disagree,
        "empty_fallback": nfb,
        "alt_sameT": {
            "csv": str(out_alt) if out_alt else None,
            "hold": alt["full"],
            "nested": alt["nested"],
            "cfg": alt["cfg"],
            "disagree_vs_v7": disagree_alt,
        },
        "candidates": [
            {"name": c["name"], "full": c["full"], "nested": c["nested"], "cfg": c["cfg"], "mode": c["mode"]}
            for c in sorted(cands, key=lambda x: (-x["nested"], -x["full"]))
        ],
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) <= 100,
        "promoted_track": None,
        "notes": [
            f"PRIMARY hold {best['full']:.4f} nested {best['nested']:.4f} (+{best['full']-V7:.4f} vs v7 full; nested +{best['nested']-0.751962:.4f})",
            f"mode={mode} ens={best['ens']} tags={tags}",
            f"selective TTA: {[m['tag'] for m in best['member_slice'] if m['use_tta']]}",
            "Depth 4-way / conf-gate did not help (wd->0)",
            "CSV written; parent submits if desired. Track submission.csv NOT auto-promoted.",
            "Do not Kaggle-submit from this script",
        ],
    }
    (ROOT / "metrics_ir_v9.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("holdout_acc", "nested_fixed_cfg", "delta_vs_v7", "disagree_vs_v7", "cfg", "mode", "ens", "clear_win", "alt_sameT")}, indent=2), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
