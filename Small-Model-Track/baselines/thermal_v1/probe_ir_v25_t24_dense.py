"""Denser nest search around phaseA T24 ens. CPU. No CSV unless clears."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune,
    GATE, MIN_DISAGREE, V7_CFG, V7_HOLD,
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))
CK = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"

def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]

def conf_gate_ir(a, b, T=1.5, thr=0.0):
    pa, pb = softmax_np(a, T), softmax_np(b, T)
    ca, cb = pa.max(1), pb.max(1)
    out = a.copy()
    use_b = (pb.argmax(1) != pa.argmax(1)) & (cb > ca + thr)
    out[use_b] = b[use_b]
    return out

def main():
    t0 = time.time()
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)
    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    s42 = np.load(CK / "hold_logits_seed42.npy").astype(np.float32)
    s888 = np.load(CK / "hold_logits_seed888.npy").astype(np.float32)
    s2024 = np.load(CK / "hold_logits_seed2024.npy").astype(np.float32)
    s2025 = np.load(CK / "hold_logits_seed2025.npy").astype(np.float32)
    phaseA = np.mean([s42, s888, s2024], 0).astype(np.float32)
    top2 = np.mean([s42, s888], 0).astype(np.float32)
    top3w = np.load(CK / "hold_logits_top3w.npy").astype(np.float32) if (CK / "hold_logits_top3w.npy").exists() else np.mean([s42,s888,s2025],0).astype(np.float32)

    pools = {}
    for tag, t24 in [("phaseA", phaseA), ("top2", top2), ("top3w", top3w), ("s42", s42)]:
        pools[tag] = t24
        for wa in np.linspace(0.30, 0.55, 11):
            for wb in np.linspace(0.15, 0.45, 11):
                wc = 1.0 - wa - wb
                if wc < 0.12 or wc > 0.45:
                    continue
                pools[f"{tag}_c9{wa:.2f}_ff{wb:.2f}_t{wc:.2f}"] = (wa*classic9 + wb*ff + wc*t24).astype(np.float32)
        for T in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
            for thr in (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15):
                pools[f"g_c9_{tag}_T{T}_m{thr}"] = conf_gate_ir(classic9, t24, T=T, thr=thr)
                blend = (0.5*classic9 + 0.5*ff).astype(np.float32)
                pools[f"g_c9ff_{tag}_T{T}_m{thr}"] = conf_gate_ir(blend, t24, T=T, thr=thr)
                blend2 = (0.4*classic9 + 0.3*ff + 0.3*t24).astype(np.float32)
                pools[f"g_blend_{tag}_T{T}_m{thr}"] = conf_gate_ir(blend2, t24, T=T, thr=thr)  # noop-ish
                # gate classic9 vs blend
                pools[f"g_c9_vs_blend_{tag}_T{T}_m{thr}"] = conf_gate_ir(classic9, blend2, T=T, thr=thr)

    mask0 = th.any(1) & mid.any(1)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    Ts_fine = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,4.0,4.5]
    Ts_med = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5]
    results = []
    for ir_name, ir in pools.items():
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask0, Ts_fine, ngrid=41)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask0, Ts_med, ngrid=25)
        pred = preds_full(ir, th, mid, mask0, bcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = min(float(nest_fixed["mean"]), float(nest_rt["mean"]))
        row = {
            "ir": ir_name, "full": b_acc, "cfg": bcfg,
            "nested_fixed": float(nest_fixed["mean"]), "nested_retune": float(nest_rt["mean"]),
            "honest_nested": honest, "disagree_vs_v7": disagree,
            "ir_solo": float((ir.argmax(1)==yt).mean()),
            "clears": bool(b_acc >= GATE and honest >= GATE and disagree >= MIN_DISAGREE),
        }
        results.append(row)

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best = ranked[0]
    clears = [r for r in ranked if r["clears"]]
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    for r in ranked[:12]:
        print(f"{r['ir']} full={r['full']:.4f} honest={r['honest_nested']:.4f} nf={r['nested_fixed']:.4f} nr={r['nested_retune']:.4f} dis={r['disagree_vs_v7']} clear={r['clears']}", flush=True)

    status = {
        "tag": "ir_v25_t24_dense_nest",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not bool(clears),
        "wrote_csv": False,
        "csv": None,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "progress": {
            "best_honest": best["honest_nested"],
            "best_full": best["full"],
            "best_ir": best["ir"],
            "best_disagree": best["disagree_vs_v7"],
            "n_clear": len(clears),
            "delta_vs_gate": best["honest_nested"] - GATE,
            "n_pools": len(pools),
            "phaseA_ens_solo": float((phaseA.argmax(1)==yt).mean()),
        },
        "best": best,
        "top15": ranked[:15],
        "clears": clears[:5],
        "next_roi": (
            ["PROMOTE"] if clears else [
                "MISS after dense nest; keep ir_v7",
                "Train stronger T24 seed (patience12, lr 7e-5) e.g. 314",
                "Gap to gate still ~0.002",
            ]
        ),
        "elapsed_sec": round(time.time()-t0,1),
        "updated_at": now,
    }
    (ROOT / "metrics_ir_v25_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v25_t24_dense.json").write_text(
        json.dumps({"best": best, "top30": ranked[:30], "n_pools": len(pools), "clears": clears}, indent=2), encoding="utf-8")
    print("BEST", best["ir"], "honest", best["honest_nested"], "full", best["full"], "clears", len(clears))
    return 0 if clears else 1

if __name__ == "__main__":
    raise SystemExit(main())
