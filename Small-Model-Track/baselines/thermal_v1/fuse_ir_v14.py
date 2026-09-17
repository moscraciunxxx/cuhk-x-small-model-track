"""IR v14 CPU fuse retune: blend ir_v7 IR members + v13 strong seeds (+ Thermal + MidFuse).
Write submission_ir_v14.csv ONLY if holdout AND nested_fixed both >= V7+0.01 (~0.763)
AND disagree vs ir_v7 preds >= 20. Else metrics-only status.
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
V7_NESTED = 0.7519623092355898
GATE = V7_HOLD + 0.01  # ~0.763
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}


def apply_cfg(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
         + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean())


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=26, leave_users=(8, 9, 24)):
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


def load_v13_members(yt_ref, users_ref):
    d = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v13"
    z = np.load(d / "hold_logits_strong.npz", allow_pickle=True)
    assert np.array_equal(z["y"], yt_ref), "v13 y mismatch"
    assert np.array_equal(z["users"], users_ref), "v13 users mismatch"
    seeds = [int(s) for s in z["seeds"]]
    members = []
    for i, seed in enumerate(seeds):
        base = z["logits"][i].astype(np.float32)
        ab = float((base.argmax(1) == yt_ref).mean())
        tta_p = d / f"test_logits_seed{seed}_tta.npy"
        test_base = np.load(d / f"test_logits_seed{seed}.npy")
        test = np.load(tta_p) if tta_p.exists() else test_base
        members.append({
            "tag": f"v13_seed{seed}",
            "logits": base,
            "base": base,
            "tta": None,
            "acc": ab,
            "acc_base": ab,
            "acc_tta": None,
            "use_tta": False,
            "test_logits": test.astype(np.float32),
            "test_base": test_base.astype(np.float32),
            "source": "v13",
        })
    return members


def build_variants(members):
    stack = np.stack([m["logits"] for m in members], 0)
    w = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64)
    w /= w.sum()
    variants = {}
    for k in range(3, len(members) + 1):
        variants[f"top{k}"] = np.mean(stack[:k], 0)
    variants["all_mean"] = np.mean(stack, 0)
    variants["all_acc_w"] = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)
    sm = np.stack([softmax_np(m["logits"], 1.0) for m in members], 0)
    variants["sm_mean"] = np.log(np.mean(sm, 0) + 1e-8).astype(np.float32)
    variants["sm_acc_w"] = np.log(np.tensordot(w, sm, axes=(0, 0)) + 1e-8).astype(np.float32)
    return variants, stack, w


def main():
    t0 = time.time()
    members_v7, yt, yu = load_members()
    classic9 = [m for m in members_v7 if m["tag"] != "pool_seed55"]
    seed55 = [m for m in members_v7 if m["tag"] == "pool_seed55"]
    v13 = load_v13_members(yt, yu)
    print("classic9:", [(m["tag"], round(m["acc"], 4)) for m in sorted(classic9, key=lambda d: -d["acc"])], flush=True)
    print("v13:", [(m["tag"], round(m["acc"], 4)) for m in v13], flush=True)
    if seed55:
        print("seed55:", round(seed55[0]["acc"], 4), flush=True)

    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx] if len(mid_full) == len(tu) else mid_full
    assert len(yt) == len(th) == len(mid), (len(yt), len(th), len(mid))
    mask = th.any(1) & mid.any(1)
    print(f"mask n={int(mask.sum())}/{len(yt)}", flush=True)

    c9_sorted = sorted(classic9, key=lambda d: -d["acc"])
    ir_v7_ens = np.mean([m["logits"] for m in c9_sorted], 0)
    v7_full = apply_cfg(ir_v7_ens, th, mid, yt, mask, V7_CFG)
    v7_nest = nested_fixed(ir_v7_ens, th, mid, yt, yu, mask, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    pools = {
        "v7_9": c9_sorted,
        "v7_9_plus_v13": sorted(c9_sorted + v13, key=lambda d: -d["acc"]),
        "v7_10_plus_v13": sorted(c9_sorted + seed55 + v13, key=lambda d: -d["acc"]),
        "v13_only": sorted(v13, key=lambda d: -d["acc"]),
        "classic_top6_plus_v13": sorted(c9_sorted[:6] + v13, key=lambda d: -d["acc"]),
        "classic_top4_plus_v13": sorted(c9_sorted[:4] + v13, key=lambda d: -d["acc"]),
    }

    Ts_fine = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
    Ts_med = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    results = []

    for pool_name, mems in pools.items():
        variants, stack, w = build_variants(mems)
        for ens_name, elogs in variants.items():
            ens_acc = float((elogs.argmax(1) == yt).mean())
            if ens_name.startswith("sm_"):
                ens_acc = float((np.exp(elogs).argmax(1) == yt).mean())
            b_acc, bcfg = fuse3_sameT(elogs, th, mid, yt, mask, Ts_fine, ngrid=41)
            g_acc, gcfg = fuse3_geom(elogs, th, mid, yt, mask, Ts_med, ngrid=31)
            nest_fixed = nested_fixed(elogs, th, mid, yt, yu, mask, bcfg)
            nest_ret = nested_retune(elogs, th, mid, yt, yu, mask, Ts_med, ngrid=21)
            v7cfg_full = apply_cfg(elogs, th, mid, yt, mask, V7_CFG)
            v7cfg_nest = nested_fixed(elogs, th, mid, yt, yu, mask, V7_CFG)
            row = {
                "pool": pool_name,
                "ens": ens_name,
                "ens_acc": ens_acc,
                "n_members": len(mems),
                "triple_acc": b_acc,
                "triple_cfg": bcfg,
                "geom_acc": g_acc,
                "geom_cfg": gcfg,
                "nested_fixed_mean": nest_fixed["mean"],
                "nested_fixed_folds": nest_fixed["folds"],
                "nested_retune_mean": nest_ret["mean"],
                "nested_retune_folds": nest_ret["folds"],
                "v7cfg_full": v7cfg_full,
                "v7cfg_nested": v7cfg_nest["mean"],
                "gate_hold": b_acc,
                "gate_nested": nest_fixed["mean"],
                "clears_metric_gate": bool(b_acc >= GATE and nest_fixed["mean"] >= GATE),
            }
            results.append(row)
            print(
                f"{pool_name}/{ens_name}: ens={ens_acc:.4f} trip={b_acc:.4f} "
                f"nestF={nest_fixed['mean']:.4f} nestR={nest_ret['mean']:.4f} "
                f"gate={row['clears_metric_gate']} cfg={bcfg}",
                flush=True,
            )

    def score(r):
        return (min(r["gate_hold"], r["gate_nested"]), r["gate_hold"], r["gate_nested"])

    results_sorted = sorted(results, key=score, reverse=True)
    best = results_sorted[0]
    print("\nBEST by min(hold,nestF):", best["pool"], best["ens"],
          f"hold={best['gate_hold']:.6f} nest={best['gate_nested']:.6f}", flush=True)

    refine_candidates = []
    for pool_name in ["v7_9_plus_v13", "v7_10_plus_v13", "classic_top6_plus_v13", "v7_9"]:
        mems = pools[pool_name]
        variants, _, _ = build_variants(mems)
        for ens_name in ["all_mean", "all_acc_w", f"top{min(9, len(mems))}", f"top{min(6, len(mems))}"]:
            if ens_name not in variants:
                continue
            elogs = variants[ens_name]
            best_n = (-1.0, None)
            best_f = (-1.0, None)
            for T in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]:
                for wa in np.linspace(0.40, 0.75, 15):
                    for wb in np.linspace(0.15, 0.50, 15):
                        wc = 1.0 - wa - wb
                        if wc < 0.02 or wc > 0.35:
                            continue
                        cfg = {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                               "mode": "sameT"}
                        full = apply_cfg(elogs, th, mid, yt, mask, cfg)
                        nest = nested_fixed(elogs, th, mid, yt, yu, mask, cfg)["mean"]
                        if nest > best_n[0]:
                            best_n = (nest, {"cfg": {**cfg, "acc": full, "n": int(mask.sum())},
                                             "full": full, "nested": nest})
                        if full > best_f[0]:
                            best_f = (full, {"cfg": {**cfg, "acc": full, "n": int(mask.sum())},
                                             "full": full, "nested": nest})
            bn = best_n[1]
            clears_r = bool(bn and bn["full"] >= GATE and bn["nested"] >= GATE)
            refine_candidates.append({
                "pool": pool_name, "ens": ens_name,
                "best_nested": bn, "best_full": best_f[1],
                "clears": clears_r,
            })
            print(f"refine {pool_name}/{ens_name}: bestNest full={bn['full']:.4f} nest={bn['nested']:.4f} "
                  f"cfg={bn['cfg']} clears={clears_r}", flush=True)

    candidates = []
    for r in results:
        candidates.append({
            "source": "grid", "pool": r["pool"], "ens": r["ens"],
            "full": r["gate_hold"], "nested": r["gate_nested"],
            "cfg": r["triple_cfg"], "members": r["n_members"],
        })
    for r in refine_candidates:
        bn = r["best_nested"]
        if bn:
            candidates.append({
                "source": "refine_nested", "pool": r["pool"], "ens": r["ens"],
                "full": bn["full"], "nested": bn["nested"],
                "cfg": bn["cfg"], "members": len(pools[r["pool"]]),
            })
        bf = r["best_full"]
        if bf:
            candidates.append({
                "source": "refine_full", "pool": r["pool"], "ens": r["ens"],
                "full": bf["full"], "nested": bf["nested"],
                "cfg": bf["cfg"], "members": len(pools[r["pool"]]),
            })

    candidates = sorted(candidates, key=lambda c: (min(c["full"], c["nested"]), c["full"], c["nested"]), reverse=True)
    top = candidates[0]
    print(f"\nTOP candidate: {top}", flush=True)

    mems = pools[top["pool"]]
    variants, _, w = build_variants(mems)
    test_stack = np.stack([m["test_logits"] for m in mems], 0)
    if top["ens"].startswith("top"):
        k = int(top["ens"].replace("top", ""))
        ir_test = np.mean(test_stack[:k], 0)
    elif top["ens"] == "all_mean":
        ir_test = np.mean(test_stack, 0)
    elif top["ens"] == "all_acc_w":
        ir_test = np.tensordot(w, test_stack, axes=(0, 0)).astype(np.float32)
    elif top["ens"] == "sm_mean":
        sm = np.stack([softmax_np(t, 1.0) for t in test_stack], 0)
        ir_test = np.log(np.mean(sm, 0) + 1e-8).astype(np.float32)
    elif top["ens"] == "sm_acc_w":
        sm = np.stack([softmax_np(t, 1.0) for t in test_stack], 0)
        ir_test = np.log(np.tensordot(w, sm, axes=(0, 0)) + 1e-8).astype(np.float32)
    else:
        ir_test = np.mean(test_stack, 0)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_p.exists():
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_p)
    cfg = top["cfg"]
    T = cfg["T"]
    probs = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T)
             + cfg["wc"] * softmax_np(mid_test, T))
    preds = probs.argmax(1)

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
        cache = ROOT / "cache" / "ir_yolo_v4"
        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
        out_csv = ROOT / "submission_ir_v14.csv"
        nfb = write_sub(out_csv, meta, preds, empty, fb)
        wrote_csv = True
        print(f"WROTE {out_csv} nfb={nfb} (gate cleared)", flush=True)
    else:
        print("GATE NOT CLEARED — metrics only; keep ir_v7", flush=True)

    report = {
        "tag": "ir_v14_fuse_retune",
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": v7_full, "nested": v7_nest},
        "v13_members": {m["tag"]: m["acc"] for m in v13},
        "classic9": {m["tag"]: m["acc"] for m in c9_sorted},
        "top_candidate": top,
        "disagree_vs_v7": disagree,
        "clears_gate": clears,
        "wrote_csv": wrote_csv,
        "csv": str(out_csv) if out_csv else None,
        "keep_ir_v7": not clears,
        "grid_top5": results_sorted[:5],
        "refine": refine_candidates,
        "elapsed_s": time.time() - t0,
        "notes": [
            "CPU-only retune blending v7 IR members + v13 strong seeds on cache/ir_yolo_v4",
            "earlyfuse/sharedstem Depth DEAD-END — not used",
            "Do NOT auto-submit; parent decides",
        ],
    }
    out_json = ROOT / "metrics_ir_v14_status.json"
    out_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    print(json.dumps({"clears_gate": clears, "top": top, "disagree": disagree}, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
