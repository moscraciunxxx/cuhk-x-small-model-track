"""Selective T24 ens re-probe after 6-seed dilution. CPU only. CSV only if gate clears."""
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
PRIOR = 0.7555837334595704

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
    ft = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)

    seeds = {}
    for p in sorted(CK.glob("hold_logits_seed*.npy")):
        sid = int(p.stem.replace("hold_logits_seed", ""))
        seeds[sid] = np.load(p).astype(np.float32)
    # val from report
    rep = json.loads((CK / "train_report.json").read_text(encoding="utf-8-sig"))
    vacc = {int(k[1:]): float(v) for k, v in rep["members"].items()}
    print("seed solos", {s: float((a.argmax(1)==yt).mean()) for s,a in seeds.items()}, "vacc", vacc, flush=True)

    def ens(ids, weighted=False):
        arrs = [seeds[i] for i in ids]
        if not weighted:
            return np.mean(arrs, 0).astype(np.float32)
        w = np.array([max(vacc.get(i, 0.5), 1e-3) for i in ids], dtype=np.float64)
        w /= w.sum()
        return np.tensordot(w, np.stack(arrs, 0), axes=(0, 0)).astype(np.float32)

    ranked_ids = sorted(seeds.keys(), key=lambda s: -vacc.get(s, 0))
    pools = {
        "classic9": classic9,
        "ff": ff,
        "t24_all6": ens(ranked_ids),
        "t24_all6_w": ens(ranked_ids, True),
        "t24_top1": ens(ranked_ids[:1]),
        "t24_top2": ens(ranked_ids[:2]),
        "t24_top3": ens(ranked_ids[:3]),
        "t24_top3_w": ens(ranked_ids[:3], True),
        "t24_top4": ens(ranked_ids[:4]),
        "t24_top4_w": ens(ranked_ids[:4], True),
        "t24_42_888": ens([42, 888]),
        "t24_42_888_2025": ens([42, 888, 2025]),
        "t24_42_888_2025_w": ens([42, 888, 2025], True),
        "t24_42_888_w": ens([42, 888], True),
        "t24_phaseA": ens([42, 888, 2024]),  # prior best phase
    }
    # blends with c9/ff
    for name, t24 in list(pools.items()):
        if not name.startswith("t24_"):
            continue
        pools[f"c9_0.4_ff_0.3_{name}_0.3"] = (0.4*classic9 + 0.3*ff + 0.3*t24).astype(np.float32)
        pools[f"c9_0.35_ff_0.35_{name}_0.3"] = (0.35*classic9 + 0.35*ff + 0.3*t24).astype(np.float32)
        pools[f"c9_0.45_ff_0.25_{name}_0.3"] = (0.45*classic9 + 0.25*ff + 0.3*t24).astype(np.float32)
        pools[f"c9_0.5_{name}_0.5"] = (0.5*classic9 + 0.5*t24).astype(np.float32)
        pools[f"c9_0.4_ff_0.4_{name}_0.2"] = (0.4*classic9 + 0.4*ff + 0.2*t24).astype(np.float32)
        # finegrid around best prior
        for wa, wb, wc in [(0.40,0.30,0.30),(0.38,0.32,0.30),(0.42,0.28,0.30),(0.40,0.28,0.32),(0.36,0.32,0.32),(0.40,0.25,0.35)]:
            pools[f"fg_{name}_{wa:.2f}_{wb:.2f}_{wc:.2f}"] = (wa*classic9 + wb*ff + wc*t24).astype(np.float32)
        for T in (1.0, 1.5, 2.0):
            for thr in (0.0, 0.05, 0.1):
                pools[f"gate_c9_{name}_T{T}_m{thr}"] = conf_gate_ir(classic9, t24, T=T, thr=thr)

    # also keep prior best-named mix with top3
    t24 = ens(ranked_ids[:3], True)
    pools["prior_style_top3w"] = (0.4*classic9 + 0.3*ff + 0.3*t24).astype(np.float32)

    mask0 = th.any(1) & mid.any(1)
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
        if row["honest_nested"] >= 0.752 or row["clears"]:
            print(f"{ir_name} full={b_acc:.4f} honest={honest:.4f} dis={disagree} solo={row['ir_solo']:.4f} clear={row['clears']}", flush=True)

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best = ranked[0]
    clears = [r for r in ranked if r["clears"]]
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    # restore best hold_logits_strong to top3 weighted for future
    top3 = ens(ranked_ids[:3], True)
    np.savez_compressed(CK / "hold_logits_strong.npz", ens=top3, y=yt, users=yu)
    np.save(CK / "hold_logits_top3w.npy", top3)
    (CK / "selective_ens_report.json").write_text(json.dumps({
        "ranked_ids": ranked_ids, "top3": ranked_ids[:3], "top3_solo": float((top3.argmax(1)==yt).mean()),
        "all6_solo": float((ens(ranked_ids).argmax(1)==yt).mean()),
        "vacc": vacc,
    }, indent=2), encoding="utf-8")

    status = {
        "tag": "ir_v25_t24_selective_ens_probe",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not bool(clears),
        "wrote_csv": False,
        "csv": None,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "t24_train": {"members": rep["members"], "ens_all6": rep["ens"], "selective_top3": ranked_ids[:3],
                      "top3w_solo": float((top3.argmax(1)==yt).mean())},
        "progress": {
            "best_honest": best["honest_nested"],
            "best_full": best["full"],
            "best_ir": best["ir"],
            "best_disagree": best["disagree_vs_v7"],
            "n_clear": len(clears),
            "phaseA_honest": PRIOR,
            "delta_vs_gate": best["honest_nested"] - GATE,
            "improved_vs_phaseA": best["honest_nested"] - PRIOR,
            "focal_ft_t24_top3w": float((top3.argmax(1)==yt).mean()),
        },
        "best": best,
        "top15": ranked[:15],
        "clears": clears[:5],
        "next_roi": (
            ["PROMOTE CSV"] if clears else [
                f"MISS gate; best honest {best['honest_nested']:.4f} (phaseA was {PRIOR:.4f})",
                "Keep ir_v7; no weak CSV",
                "Next: try LOUO-aware seed pick / mid weight densify / exclude weak seeds from IR pool",
                "Or train stronger T24 recipe (longer patience / lower lr) on seed42 init",
            ]
        ),
        "elapsed_sec": round(time.time()-t0,1),
        "updated_at": now,
    }
    (ROOT / "metrics_ir_v25_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v25_t24_selective.json").write_text(
        json.dumps({"best": best, "top30": ranked[:30], "n_pools": len(pools), "clears": clears}, indent=2), encoding="utf-8")
    print("BEST", best["ir"], "honest", best["honest_nested"], "full", best["full"],
          "dis", best["disagree_vs_v7"], "clears", len(clears), "top3", ranked_ids[:3])
    return 0 if clears else 1

if __name__ == "__main__":
    raise SystemExit(main())
