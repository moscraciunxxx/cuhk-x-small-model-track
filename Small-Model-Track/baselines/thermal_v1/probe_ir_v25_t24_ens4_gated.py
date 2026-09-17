"""CPU-only: T24 s42 + T16 focal_ft ens4 gated/finegrid nested-honest. No CSV unless gate clears. No GPU."""
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
    if len(c9m) < 9:
        c9m = allc[:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)

    ft = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    t24 = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)

    print("c9", float((classic9.argmax(1)==yt).mean()),
          "ff_ens4", float((ff.argmax(1)==yt).mean()),
          "ft", float((ft.argmax(1)==yt).mean()),
          "t24_s42", float((t24.argmax(1)==yt).mean()), flush=True)

    pools = {
        "classic9_base": classic9,
        "t24_s42": t24,
        "ff_ens4": ff,
        "classic9_plus_t24": np.mean([classic9, t24], 0).astype(np.float32),
        "classic9_plus_ff_t24": np.mean([classic9, ff, t24], 0).astype(np.float32),
        "c9_0.5_t24_0.5": (0.5*classic9 + 0.5*t24).astype(np.float32),
        "c9_0.6_t24_0.4": (0.6*classic9 + 0.4*t24).astype(np.float32),
        "c9_0.4_ff_0.4_t24_0.2": (0.4*classic9 + 0.4*ff + 0.2*t24).astype(np.float32),
        "c9_0.4_ff_0.3_t24_0.3": (0.4*classic9 + 0.3*ff + 0.3*t24).astype(np.float32),
        "c9_0.35_ff_0.35_t24_0.3": (0.35*classic9 + 0.35*ff + 0.3*t24).astype(np.float32),
        "c9_0.3_ff_0.4_t24_0.3": (0.3*classic9 + 0.4*ff + 0.3*t24).astype(np.float32),
        "c9_0.45_ff_0.25_t24_0.3": (0.45*classic9 + 0.25*ff + 0.3*t24).astype(np.float32),
        "c9_0.5_ff_0.25_t24_0.25": (0.5*classic9 + 0.25*ff + 0.25*t24).astype(np.float32),
        "ff_0.5_t24_0.5": (0.5*ff + 0.5*t24).astype(np.float32),
        "c9_0.4_ff_0.4_ft_0.2": (0.4*classic9 + 0.4*ff + 0.2*ft).astype(np.float32),  # prior
        "c9_0.35_ff_0.3_t24_0.2_ft_0.15": (0.35*classic9 + 0.3*ff + 0.2*t24 + 0.15*ft).astype(np.float32),
        "c9_0.3_ff_0.3_t24_0.25_ft_0.15": (0.3*classic9 + 0.3*ff + 0.25*t24 + 0.15*ft).astype(np.float32),
    }
    # finegrid c9/ff/t24
    for wa in np.linspace(0.25, 0.55, 7):
        for wb in np.linspace(0.20, 0.50, 7):
            wc = 1.0 - wa - wb
            if wc < 0.10 or wc > 0.45:
                continue
            pools[f"fg_c9_{wa:.2f}_ff_{wb:.2f}_t24_{wc:.2f}"] = (wa*classic9 + wb*ff + wc*t24).astype(np.float32)

    # swaps
    kept1 = [base_of(m) for m in c9m[:-1]]
    kept2 = [base_of(m) for m in c9m[:-2]]
    pools["swap1_t24"] = np.mean(kept1 + [t24], 0).astype(np.float32)
    pools["swap2_t24_ff"] = np.mean(kept2 + [t24, ff], 0).astype(np.float32)

    # conf gates involving t24
    for T in (1.0, 1.5, 2.0, 2.5):
        for thr in (0.0, 0.05, 0.1, 0.15):
            pools[f"gate_c9_t24_T{T}_m{thr}"] = conf_gate_ir(classic9, t24, T=T, thr=thr)
            pools[f"gate_ff_t24_T{T}_m{thr}"] = conf_gate_ir(ff, t24, T=T, thr=thr)
            blend = (0.5*classic9 + 0.5*ff).astype(np.float32)
            pools[f"gate_c9ff_t24_T{T}_m{thr}"] = conf_gate_ir(blend, t24, T=T, thr=thr)
            blend2 = (0.4*classic9 + 0.4*ff + 0.2*t24).astype(np.float32)
            pools[f"gate_blend_ft_T{T}_m{thr}"] = conf_gate_ir(blend2, ft, T=T, thr=thr)

    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    Ts_fine = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0]
    Ts_med = [1.0,1.5,2.0,2.5,3.0,3.5]

    results = []
    for ir_name, ir in pools.items():
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask0, Ts_fine, ngrid=31)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask0, Ts_med, ngrid=17)
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
        print(f"{ir_name} full={b_acc:.4f} honest={honest:.4f} dis={disagree} solo={row['ir_solo']:.4f} clear={row['clears']}", flush=True)

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best = ranked[0]
    clears = [r for r in ranked if r["clears"]]
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    # preserve prior t24 best from status if higher
    prior_t24_honest = 0.7500822794797232
    best_honest = max(best["honest_nested"], prior_t24_honest)
    status = {
        "tag": "ir_v25_t24_ens4_gated_cpu",
        "outcome": "WIN" if best["clears"] else "MISS_WAITING_HOTC",
        "keep_ir_v7": not best["clears"],
        "wrote_csv": False,
        "csv": None,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce_full": v7_full,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "progress": {
            "focal_ft_ens4": float((ff.argmax(1)==yt).mean()),
            "ft_ens": float((ft.argmax(1)==yt).mean()),
            "focal_ft_t24_s42": float((t24.argmax(1)==yt).mean()),
            "best_honest": best["honest_nested"],
            "best_full": best["full"],
            "best_ir": best["ir"],
            "best_disagree": best["disagree_vs_v7"],
            "n_clear": len(clears),
            "t24_best_honest_prior": prior_t24_honest,
            "gpu": "CPU_ONLY_WAITING_HOTC",
        },
        "best": best,
        "top15": ranked[:15],
        "angles": {
            "11_focal_ft_more_seeds": "DONE ens4=0.695",
            "12_ens4_probe": "DONE best_honest~0.744 miss",
            "13_x3d_ft": "DONE_MISS solo0.560",
            "14_ft_t24_cache": "DONE yolo2930",
            "15_focal_ft_t24": "DONE s42 solo0.697; WAIT seeds 2024/888 for GPU",
            "16_t24_probe": "DONE prior honest0.750 MISS",
            "17_t24_ens4_gated_cpu": "DONE this run",
        },
        "next_roi": (
            ["PROMOTE CSV"] if best["clears"] else
            [
                "WAIT GPU free (HOTC HELIOS/ViPT) — do not train until nvidia-smi 0 and parent resume",
                "Then multi-seed train_ir_focal_ft_t24_v24.py seeds 2024 888 on cache/ir_yolo_ft_v24_t24",
                "Blend T24 ens + T16 focal_ft ens4 nested-honest; promote only if honest>=0.758 & dis>=20",
                "hold ir_v7 — no weak CSV",
            ]
        ),
        "elapsed_sec": round(time.time()-t0,1),
        "updated_at": now,
        "handoff": {"for": "HOTC", "gpu": "YIELDING", "note": "Small-Model CPU-only gated probe; no GPU train"},
    }
    out = ROOT / "metrics_ir_v25_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    detail = ROOT / "metrics_ir_v25_t24_ens4_gated.json"
    detail.write_text(json.dumps({"best": best, "top30": ranked[:30], "n_pools": len(pools), "clears": clears}, indent=2), encoding="utf-8")
    print("BEST", best["ir"], "honest", best["honest_nested"], "full", best["full"],
          "dis", best["disagree_vs_v7"], "clears", best["clears"], "n_clear", len(clears), "->", out)

if __name__ == "__main__":
    main()
