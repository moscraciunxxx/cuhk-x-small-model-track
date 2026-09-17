"""Fuse YOLO11n IR R2+1D members (+ optional classic v7 IR) with Thermal+Mid.
Write CSV only if hold AND nested_fixed >= V7+0.01 and disagree>=20.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import (
    load_members, softmax_np, nested_fixed, write_sub, fuse3_sameT, fuse3_geom,
)
from dataset import DEFAULT_HOLD_OUT_USERS

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7_HOLD = 0.7530364372469636
GATE = V7_HOLD + 0.01
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}


def apply_cfg(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
         + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean())


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=21, leave_users=(8, 9, 24)):
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": acc,
                      "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def load_y11_members(yt_ref, users_ref):
    d = ROOT / "checkpoints" / "ir_yolo11n_r2p1d18_v1"
    z = np.load(d / "hold_logits_strong.npz", allow_pickle=True)
    assert np.array_equal(z["y"], yt_ref)
    assert np.array_equal(z["users"], users_ref)
    seeds = [int(s) for s in z["seeds"]]
    members = []
    for i, seed in enumerate(seeds):
        base = z["logits"][i].astype(np.float32)
        ab = float((base.argmax(1) == yt_ref).mean())
        tta_p = d / f"test_logits_seed{seed}_tta.npy"
        test_base = np.load(d / f"test_logits_seed{seed}.npy")
        test = np.load(tta_p) if tta_p.exists() else test_base
        members.append({
            "tag": f"y11_seed{seed}",
            "logits": base,
            "acc": ab,
            "test_logits": test.astype(np.float32),
            "source": "yolo11n",
        })
    return members


def build_variants(members):
    stack = np.stack([m["logits"] for m in members], 0)
    w = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64)
    w /= w.sum()
    variants = {"all_mean": np.mean(stack, 0),
                "all_acc_w": np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)}
    for k in range(3, len(members) + 1):
        variants[f"top{k}"] = np.mean(stack[:k], 0)
    sm = np.stack([softmax_np(m["logits"], 1.0) for m in members], 0)
    variants["sm_mean"] = np.log(np.mean(sm, 0) + 1e-8).astype(np.float32)
    return variants, stack, w


def main():
    t0 = time.time()
    members_v7, yt, yu = load_members()
    classic9 = sorted([m for m in members_v7 if m["tag"] != "pool_seed55"], key=lambda d: -d["acc"])
    y11 = sorted(load_y11_members(yt, yu), key=lambda d: -d["acc"])
    print("y11:", [(m["tag"], round(m["acc"], 4)) for m in y11], flush=True)
    print("classic9 top:", [(m["tag"], round(m["acc"], 4)) for m in classic9[:4]], flush=True)

    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy")
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx] if len(mid_full) == len(tu) else mid_full
    mask = th.any(1) & mid.any(1)
    print(f"mask n={int(mask.sum())}/{len(yt)}", flush=True)

    pools = {
        "y11_only": y11,
        "classic9": classic9,
        "classic9_plus_y11": sorted(classic9 + y11, key=lambda d: -d["acc"]),
        "classic_top6_plus_y11": sorted(classic9[:6] + y11, key=lambda d: -d["acc"]),
        "classic_top4_plus_y11": sorted(classic9[:4] + y11, key=lambda d: -d["acc"]),
        "y11_plus_classic_top3": sorted(y11 + classic9[:3], key=lambda d: -d["acc"]),
    }

    Ts_fine = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
    Ts_med = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    results = []
    for pool_name, mems in pools.items():
        variants, _, w = build_variants(mems)
        for ens_name, elogs in variants.items():
            ens_acc = float((elogs.argmax(1) == yt).mean())
            if ens_name.startswith("sm_"):
                ens_acc = float((np.exp(elogs).argmax(1) == yt).mean())
            b_acc, bcfg = fuse3_sameT(elogs, th, mid, yt, mask, Ts_fine, ngrid=41)
            nest_fixed = nested_fixed(elogs, th, mid, yt, yu, mask, bcfg)
            nest_ret = nested_retune(elogs, th, mid, yt, yu, mask, Ts_med, ngrid=21)
            row = {
                "pool": pool_name, "ens": ens_name, "ens_acc": ens_acc,
                "n_members": len(mems), "triple_acc": b_acc, "triple_cfg": bcfg,
                "nested_fixed_mean": nest_fixed["mean"], "nested_fixed_folds": nest_fixed["folds"],
                "nested_retune_mean": nest_ret["mean"],
                "gate_hold": b_acc, "gate_nested": nest_fixed["mean"],
                "clears_metric_gate": bool(b_acc >= GATE and nest_fixed["mean"] >= GATE),
            }
            results.append(row)
            print(f"{pool_name}/{ens_name}: ens={ens_acc:.4f} trip={b_acc:.4f} "
                  f"nestF={nest_fixed['mean']:.4f} nestR={nest_ret['mean']:.4f} "
                  f"gate={row['clears_metric_gate']}", flush=True)

    # refine nested-scored on promising pools
    refine = []
    for pool_name in ["y11_only", "classic9_plus_y11", "classic_top6_plus_y11", "y11_plus_classic_top3", "classic9"]:
        mems = pools[pool_name]
        variants, _, _ = build_variants(mems)
        for ens_name in ["all_mean", "all_acc_w"] + [k for k in variants if k.startswith("top")]:
            elogs = variants[ens_name]
            best_n = (-1.0, None)
            for T in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]:
                for wa in np.linspace(0.40, 0.75, 15):
                    for wb in np.linspace(0.15, 0.50, 15):
                        wc = 1.0 - wa - wb
                        if wc < 0.02 or wc > 0.35:
                            continue
                        cfg = {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T), "mode": "sameT"}
                        full = apply_cfg(elogs, th, mid, yt, mask, cfg)
                        nest = nested_fixed(elogs, th, mid, yt, yu, mask, cfg)["mean"]
                        if nest > best_n[0]:
                            best_n = (nest, {"cfg": {**cfg, "acc": full, "n": int(mask.sum())},
                                             "full": full, "nested": nest})
            bn = best_n[1]
            clears = bool(bn["full"] >= GATE and bn["nested"] >= GATE)
            refine.append({"pool": pool_name, "ens": ens_name, "best_nested": bn, "clears": clears})
            print(f"refine {pool_name}/{ens_name}: full={bn['full']:.4f} nest={bn['nested']:.4f} clears={clears}", flush=True)

    candidates = []
    for r in results:
        candidates.append({"source": "grid", "pool": r["pool"], "ens": r["ens"],
                           "full": r["gate_hold"], "nested": r["gate_nested"],
                           "cfg": r["triple_cfg"], "members": r["n_members"]})
    for r in refine:
        bn = r["best_nested"]
        candidates.append({"source": "refine_nested", "pool": r["pool"], "ens": r["ens"],
                           "full": bn["full"], "nested": bn["nested"],
                           "cfg": bn["cfg"], "members": len(pools[r["pool"]])})
    candidates = sorted(candidates, key=lambda c: (min(c["full"], c["nested"]), c["full"], c["nested"]), reverse=True)
    top = candidates[0]
    print("TOP:", top, flush=True)

    mems = pools[top["pool"]]
    variants, _, w = build_variants(mems)
    test_stack = np.stack([m["test_logits"] for m in mems], 0)
    if top["ens"].startswith("top"):
        k = int(top["ens"].replace("top", ""))
        ir_test = np.mean(test_stack[:k], 0)
    elif top["ens"] == "all_acc_w":
        ir_test = np.tensordot(w, test_stack, axes=(0, 0)).astype(np.float32)
    elif top["ens"] == "sm_mean":
        sm = np.stack([softmax_np(t, 1.0) for t in test_stack], 0)
        ir_test = np.log(np.mean(sm, 0) + 1e-8).astype(np.float32)
    else:
        ir_test = np.mean(test_stack, 0)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_p.exists():
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_p)
    cfg = top["cfg"]
    T = cfg["T"]
    preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T)
             + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)

    v7_preds = []
    with open(ROOT / "submission_ir_v7.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v7_preds.append(int(row["prediction"]))
    disagree = int(sum(int(a) != int(b) for a, b in zip(preds, v7_preds)))
    print(f"disagree vs ir_v7: {disagree}", flush=True)

    clears = bool(top["full"] >= GATE and top["nested"] >= GATE and disagree >= MIN_DISAGREE)
    wrote_csv = False
    out_csv = None
    if clears:
        cache = ROOT / "cache" / "ir_yolo_11n_v1"
        # meta should match test order from same discover; fall back to v4 meta if needed
        meta_p = cache / "test_meta.json"
        if not meta_p.exists():
            meta_p = ROOT / "cache" / "ir_yolo_v4" / "test_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        empty_p = cache / "test_empty.json"
        if not empty_p.exists():
            empty_p = ROOT / "cache" / "ir_yolo_v4" / "test_empty.json"
        empty = set(json.loads(empty_p.read_text(encoding="utf-8")))
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
        out_csv = ROOT / "submission_ir_yolo11n_v1.csv"
        nfb = write_sub(out_csv, meta, preds, empty, fb)
        wrote_csv = True
        print(f"WROTE {out_csv} nfb={nfb}", flush=True)
    else:
        print("GATE NOT CLEARED — keep ir_v7", flush=True)

    # detect quality note
    det_train = json.loads((ROOT / "cache" / "ir_yolo_11n_v1" / "detect_stats_train.json").read_text())
    det_test = json.loads((ROOT / "cache" / "ir_yolo_11n_v1" / "detect_stats_test.json").read_text())
    det_v4_tr = json.loads((ROOT / "cache" / "ir_yolo_v4" / "detect_stats_train.json").read_text())
    det_v4_te = json.loads((ROOT / "cache" / "ir_yolo_v4" / "detect_stats_test.json").read_text())

    report = {
        "tag": "ir_yolo11n_v1",
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "detect_yolo11n": {"train": det_train, "test": det_test},
        "detect_yolov8n_v4": {"train": det_v4_tr, "test": det_v4_te},
        "y11_members": {m["tag"]: m["acc"] for m in y11},
        "y11_ens_hold": float(np.mean([m["logits"] for m in y11], 0).argmax(1).mean() if False else
                              (np.mean(np.stack([m["logits"] for m in y11], 0), 0).argmax(1) == yt).mean()),
        "top_candidate": top,
        "disagree_vs_v7": disagree,
        "clears_gate": clears,
        "wrote_csv": wrote_csv,
        "csv": str(out_csv) if out_csv else None,
        "keep_ir_v7": not clears,
        "grid_top5": sorted(results, key=lambda r: (min(r["gate_hold"], r["gate_nested"]), r["gate_hold"]), reverse=True)[:5],
        "refine_top5": sorted(refine, key=lambda r: (min(r["best_nested"]["full"], r["best_nested"]["nested"]), r["best_nested"]["full"]), reverse=True)[:5],
        "elapsed_s": time.time() - t0,
        "pack_note": "yolo11n.pt ~5.4MB + r2p1d18 fp16 ~60MB <=100MB OK",
        "notes": [
            "YOLO11n IR crop cache ir_yolo_11n_v1; detect ~98.8% vs v4 ~99.0% (proceed)",
            "Do NOT auto-submit; parent decides",
            "GPU freed after train",
        ],
    }
    out_json = ROOT / "metrics_ir_yolo11n_v1_status.json"
    out_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    print(json.dumps({"clears_gate": clears, "top": top, "disagree": disagree,
                      "y11_ens": report["y11_ens_hold"]}, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
