"""ir_v25: focal_ft ens4 + FT + classic9 multi-blend / swap / class-conf gate. Nested-honest gate >=0.758, disagree>=20."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune, GATE, MIN_DISAGREE, V7_CFG, V7_HOLD
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))

def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]

def conf_gate_ir(a, b, T=1.5, thr=0.0):
    """Prefer higher-confidence source when disagree; optional thr margin."""
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
    fo = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)

    seed_ff = {}
    for p in sorted((ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24").glob("hold_logits_seed*.npy")):
        seed = p.stem.replace("hold_logits_seed", "")
        seed_ff[seed] = np.load(p).astype(np.float32)

    # top3 focal_ft seeds by solo
    ranked_seeds = sorted(seed_ff.items(), key=lambda kv: -(kv[1].argmax(1)==yt).mean())
    ff_top3 = np.mean([a for _, a in ranked_seeds[:3]], 0).astype(np.float32)
    ff_best2 = np.mean([a for _, a in ranked_seeds[:2]], 0).astype(np.float32)

    print("classic9 solo", float((classic9.argmax(1)==yt).mean()), "tags", [m["tag"] for m in c9m])
    print("ft", float((ft.argmax(1)==yt).mean()), "fo", float((fo.argmax(1)==yt).mean()),
          "ff_ens4", float((ff.argmax(1)==yt).mean()), "ff_top3", float((ff_top3.argmax(1)==yt).mean()))
    for s, a in ranked_seeds:
        print(f"  ff_seed{s}", float((a.argmax(1)==yt).mean()))

    pools = {
        "classic9_base": classic9,
        "ff_ens4": ff,
        "ff_top3": ff_top3,
        "ff_best2": ff_best2,
        "ft_ens": ft,
        "fo_ens": fo,
        "c9_plus_ff": np.mean([classic9, ff], 0).astype(np.float32),
        "c9_plus_ft_ff": np.mean([classic9, ft, ff], 0).astype(np.float32),
        "c9_0.5_ff_0.5": (0.5*classic9 + 0.5*ff).astype(np.float32),
        "c9_0.6_ff_0.4": (0.6*classic9 + 0.4*ff).astype(np.float32),
        "c9_0.7_ff_0.3": (0.7*classic9 + 0.3*ff).astype(np.float32),
        "c9_0.4_ff_0.4_ft_0.2": (0.4*classic9 + 0.4*ff + 0.2*ft).astype(np.float32),
        "c9_0.45_ff_0.35_ft_0.2": (0.45*classic9 + 0.35*ff + 0.2*ft).astype(np.float32),
        "c9_0.5_ff_0.3_ft_0.2": (0.5*classic9 + 0.3*ff + 0.2*ft).astype(np.float32),
        "c9_0.55_ff_0.3_ft_0.15": (0.55*classic9 + 0.3*ff + 0.15*ft).astype(np.float32),
        "c9_0.4_ff_0.35_ft_0.25": (0.4*classic9 + 0.35*ff + 0.25*ft).astype(np.float32),
        "c9_0.35_ff_0.4_ft_0.25": (0.35*classic9 + 0.4*ff + 0.25*ft).astype(np.float32),
        "c9_0.3_ff_0.4_ft_0.3": (0.3*classic9 + 0.4*ff + 0.3*ft).astype(np.float32),
        "ff_0.5_ft_0.5": (0.5*ff + 0.5*ft).astype(np.float32),
        "c9_0.5_fftop3_0.3_ft_0.2": (0.5*classic9 + 0.3*ff_top3 + 0.2*ft).astype(np.float32),
        "c9_0.4_fftop3_0.4_ft_0.2": (0.4*classic9 + 0.4*ff_top3 + 0.2*ft).astype(np.float32),
    }
    # finegrid around prior best
    for wa in np.linspace(0.25, 0.55, 7):
        for wb in np.linspace(0.25, 0.55, 7):
            wc = 1.0 - wa - wb
            if wc < 0.05 or wc > 0.45:
                continue
            pools[f"fg_c9_{wa:.2f}_ff_{wb:.2f}_ft_{wc:.2f}"] = (wa*classic9 + wb*ff + wc*ft).astype(np.float32)

    # swap weakest classic members with strongest ff seeds
    kept1 = [base_of(m) for m in c9m[:-1]]
    kept2 = [base_of(m) for m in c9m[:-2]]
    kept3 = [base_of(m) for m in c9m[:-3]]
    pools["swap1_ff"] = np.mean(kept1 + [ff], 0).astype(np.float32)
    pools["swap2_ff_ft"] = np.mean(kept2 + [ff, ft], 0).astype(np.float32)
    pools["swap2_ff_s888"] = np.mean(kept2 + [ff, seed_ff["888"]], 0).astype(np.float32)
    pools["swap3_ff_ft_s888"] = np.mean(kept3 + [ff, ft, seed_ff["888"]], 0).astype(np.float32)
    # add best seeds into pool mean
    pools["c9_plus_s888_s2024"] = np.mean([classic9, seed_ff["888"], seed_ff["2024"]], 0).astype(np.float32)

    # class-conf gated blends (IR-level)
    for T in (1.0, 1.5, 2.0):
        for thr in (0.0, 0.05, 0.1):
            pools[f"gate_c9_ff_T{T}_m{thr}"] = conf_gate_ir(classic9, ff, T=T, thr=thr)
            pools[f"gate_ff_ft_T{T}_m{thr}"] = conf_gate_ir(ff, ft, T=T, thr=thr)
            blend = (0.5*classic9 + 0.5*ff).astype(np.float32)
            pools[f"gate_blend_ft_T{T}_m{thr}"] = conf_gate_ir(blend, ft, T=T, thr=thr)

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
    status = {
        "tag": "ir_v25_focal_ft_ens4",
        "outcome": "WIN" if best["clears"] else "MISS_IN_PROGRESS",
        "keep_ir_v7": not best["clears"],
        "wrote_csv": False,
        "csv": None,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce_full": v7_full,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "progress": {
            "focal_ft_ens4": float((ff.argmax(1)==yt).mean()),
            "focal_ft_seeds": {f"s{s}": float((a.argmax(1)==yt).mean()) for s,a in seed_ff.items()},
            "ft_ens": float((ft.argmax(1)==yt).mean()),
            "best_honest": best["honest_nested"],
            "best_full": best["full"],
            "best_ir": best["ir"],
            "best_disagree": best["disagree_vs_v7"],
            "n_clear": len(clears),
        },
        "best": best,
        "top15": ranked[:15],
        "angles": {
            "11_focal_ft_more_seeds": "DONE ens4=0.695",
            "12_ens4_probe": "DONE",
            "13_x3d_ft": "NEXT" if not best["clears"] else "SKIP",
        },
        "next_roi": (
            ["PROMOTE CSV"] if best["clears"] else
            ["Train X3D-M on cache/ir_yolo_ft_v24", "Optional R2P1D-18 T=16 FT already covered by focal_ft", "hold ir_v7 — no weak CSV"]
        ),
        "elapsed_sec": round(time.time()-t0,1),
        "updated_at": now,
    }
    out = ROOT / "metrics_ir_v25_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("BEST", best["ir"], "honest", best["honest_nested"], "full", best["full"],
          "dis", best["disagree_vs_v7"], "clears", best["clears"], "n_clear", len(clears), "->", out)

if __name__ == "__main__":
    main()
