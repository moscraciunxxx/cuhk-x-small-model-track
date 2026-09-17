"""Multi-IR pool fuse: classic members + FT + focal swaps. Gate honest nested>=0.758, disagree>=20."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune, GATE, MIN_DISAGREE, V7_CFG, V7_HOLD
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent

def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]

def main():
    t0 = time.time()
    members, yt, yu = load_members()
    # all classic sorted by base acc desc
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9 = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    if len(c9) < 9:
        c9 = allc[:9]
    classic9 = np.mean([base_of(m) for m in c9], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)

    ft = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    fo = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    # per-seed FT/focal if present
    extras = [("ft_ens", ft), ("focal_ens", fo)]
    for p in sorted((ROOT / "checkpoints" / "ir_yolo_r2p1d18_ft_v24").glob("hold_logits_seed*.npy")):
        extras.append((p.stem.replace("hold_logits_", "ft_"), np.load(p).astype(np.float32)))
    for p in sorted((ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_v24").glob("hold_logits_seed*.npy")):
        extras.append((p.stem.replace("hold_logits_", "fo_"), np.load(p).astype(np.float32)))

    print("classic9 tags", [m["tag"] for m in c9], "solo", float((classic9.argmax(1)==yt).mean()))
    for name, arr in extras:
        print(name, float((arr.argmax(1)==yt).mean()))

    pools = {"classic9_base": classic9}
    # replace bottom k with FT+focal ranked by solo
    ranked_new = sorted(extras, key=lambda kv: -(kv[1].argmax(1)==yt).mean())
    for k in (1, 2, 3):
        kept = [base_of(m) for m in c9[:-k]]
        add = [arr for _, arr in ranked_new[:k]]
        pools[f"swap_bot{k}_topnew"] = np.mean(kept + add, 0).astype(np.float32)
    # replace specific weakest with both FT and focal (drop 2)
    kept = [base_of(m) for m in c9[:-2]]
    pools["swap2_ft_focal"] = np.mean(kept + [ft, fo], 0).astype(np.float32)
    kept1 = [base_of(m) for m in c9[:-1]]
    pools["swap1_ft"] = np.mean(kept1 + [ft], 0).astype(np.float32)
    pools["swap1_focal"] = np.mean(kept1 + [fo], 0).astype(np.float32)
    # enlarge pool
    pools["classic9_plus_ft"] = np.mean([classic9, ft], 0).astype(np.float32)
    pools["classic9_plus_ft_fo"] = np.mean([classic9, ft, fo], 0).astype(np.float32)
    pools["c9_0.6_ft_0.25_fo_0.15"] = (0.6*classic9 + 0.25*ft + 0.15*fo).astype(np.float32)
    pools["c9_0.5_ft_0.3_fo_0.2"] = (0.5*classic9 + 0.3*ft + 0.2*fo).astype(np.float32)
    pools["c9_0.7_ft_0.3"] = (0.7*classic9 + 0.3*ft).astype(np.float32)
    pools["ft_only"] = ft
    pools["ft_focal_mean"] = np.mean([ft, fo], 0).astype(np.float32)

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
    status = {
        "tag": "ir_v25_multi_swap",
        "outcome": "WIN" if best["clears"] else "MISS",
        "keep_ir_v7": not best["clears"],
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce_full": v7_full,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "best": best,
        "top10": ranked[:10],
        "wrote_csv": False,
        "csv": None,
        "elapsed_sec": round(time.time()-t0,1),
        "ft_solo": float((ft.argmax(1)==yt).mean()),
        "focal_solo": float((fo.argmax(1)==yt).mean()),
    }
    out = ROOT / "metrics_ir_v25_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("BEST", best["ir"], "honest", best["honest_nested"], "clears", best["clears"], "->", out)

if __name__ == "__main__":
    main()
