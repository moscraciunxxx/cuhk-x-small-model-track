"""ir_v27 SAFE near-v7 fuse (CPU). Lesson from v26 public 0.67164 < v7 0.69154 with 53 disagrees.

Prefer:
- small disagree (<=15, hard fail >20) with nested >= v7 (~0.753) and slight lift, OR
- much stronger IR solo that can replace weakly without wholesale swap.

NO CSV unless clears. Keep submission_ir_v7 as best public.
"""
from __future__ import annotations
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune, V7_CFG, V7_HOLD,
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
CK_SCRATCH = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_strong_scratch"

NESTED_MIN = 0.7530
MAX_DISAGREE = 15
SOFT_MAX_DISAGREE = 20


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def conf_gate_low_primary(primary, aux, T=2.5, thr=0.55, max_changes=20):
    """Only swap primary->aux on lowest-conf primary clips where aux disagrees, capped."""
    pp = softmax_np(primary, T)
    ap = softmax_np(aux, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    conf = pp.max(1)
    disagree = pa != aa
    cand = np.where(disagree & (conf <= thr))[0]
    order = cand[np.argsort(conf[cand])]
    out = primary.copy()
    n = 0
    for i in order:
        if n >= max_changes:
            break
        out[i] = aux[i]
        n += 1
    return out, int(n)


def fuse3_sameT_near(a, b, c, y, mask, center, Ts, ngrid=11, radius=0.12):
    """sameT search restricted near center weights (v7-like)."""
    best = (-1.0, None)
    yt = y[mask]
    pa0, pb0, pc0 = a[mask], b[mask], c[mask]
    wa0, wb0, wc0 = center["wa"], center["wb"], center["wc"]
    was = np.unique(np.clip(np.linspace(wa0 - radius, wa0 + radius, ngrid), 0.0, 1.0))
    for T in Ts:
        pa, pb, pc = softmax_np(pa0, T), softmax_np(pb0, T), softmax_np(pc0, T)
        for wa in was:
            for wb in np.unique(np.clip(np.linspace(wb0 - radius, wb0 + radius, ngrid), 0.0, 1.0 - wa)):
                wc = 1.0 - wa - wb
                if abs(wc - wc0) > radius + 1e-9 or wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "sameT_near"})
    return best


def main():
    t0 = time.time()
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)

    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)

    seeds = {}
    for p in sorted(CK24.glob("hold_logits_seed*.npy")):
        sid = int(p.stem.replace("hold_logits_seed", ""))
        seeds[sid] = np.load(p).astype(np.float32)
    phaseA = np.mean([seeds[42], seeds[888], seeds[2024]], 0).astype(np.float32)
    t24_42 = seeds[42]
    t24_top2 = np.mean([seeds[42], seeds[888]], 0).astype(np.float32)

    scratch = None
    sp = CK_SCRATCH / "hold_logits_seed42.npy"
    if sp.exists():
        scratch = np.load(sp).astype(np.float32)

    print("c9", float((classic9.argmax(1) == yt).mean()),
          "ff", float((ff.argmax(1) == yt).mean()),
          "phaseA", float((phaseA.argmax(1) == yt).mean()),
          "t24_42", float((t24_42.argmax(1) == yt).mean()),
          "scratch", None if scratch is None else float((scratch.argmax(1) == yt).mean()),
          flush=True)

    pools = {
        "classic9": classic9,
        "c9_0.90_pa_0.10": (0.90 * classic9 + 0.10 * phaseA).astype(np.float32),
        "c9_0.85_pa_0.15": (0.85 * classic9 + 0.15 * phaseA).astype(np.float32),
        "c9_0.80_pa_0.20": (0.80 * classic9 + 0.20 * phaseA).astype(np.float32),
        "c9_0.75_pa_0.25": (0.75 * classic9 + 0.25 * phaseA).astype(np.float32),
        "c9_0.90_t42_0.10": (0.90 * classic9 + 0.10 * t24_42).astype(np.float32),
        "c9_0.85_t42_0.15": (0.85 * classic9 + 0.15 * t24_42).astype(np.float32),
        "c9_0.80_t42_0.20": (0.80 * classic9 + 0.20 * t24_42).astype(np.float32),
        "c9_0.85_top2_0.15": (0.85 * classic9 + 0.15 * t24_top2).astype(np.float32),
        "c9_0.80_top2_0.20": (0.80 * classic9 + 0.20 * t24_top2).astype(np.float32),
        "c9_0.85_ff_0.15": (0.85 * classic9 + 0.15 * ff).astype(np.float32),
        "c9_0.80_ff_0.20": (0.80 * classic9 + 0.20 * ff).astype(np.float32),
        "c9_0.70_ff_0.15_pa_0.15": (0.70 * classic9 + 0.15 * ff + 0.15 * phaseA).astype(np.float32),
        "c9_0.75_ff_0.125_pa_0.125": (0.75 * classic9 + 0.125 * ff + 0.125 * phaseA).astype(np.float32),
        "c9_0.80_ff_0.10_t42_0.10": (0.80 * classic9 + 0.10 * ff + 0.10 * t24_42).astype(np.float32),
        "c9_0.70_pa_0.15_t42_0.15": (0.70 * classic9 + 0.15 * phaseA + 0.15 * t24_42).astype(np.float32),
    }
    if scratch is not None:
        for a in (0.05, 0.10, 0.15, 0.20, 0.25):
            pools[f"c9_{1-a:.2f}_scratch_{a:.2f}"] = ((1 - a) * classic9 + a * scratch).astype(np.float32)

    for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        for aux_name, aux in [("pa", phaseA), ("t42", t24_42), ("top2", t24_top2), ("ff", ff)]:
            gated, nch = conf_gate_low_primary(classic9, aux, T=2.5, thr=thr, max_changes=20)
            pools[f"cgate_c9_{aux_name}_thr{thr:.2f}_n{nch}"] = gated

    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    Ts_near = [2.0, 2.25, 2.5, 2.75, 3.0]
    Ts_med = [1.5, 2.0, 2.5, 3.0]
    results = []

    for ir_name, ir in pools.items():
        full_fixed, _ = apply_cfg(ir, th, mid, yt, mask0, V7_CFG)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask0, V7_CFG)
        pred_fixed = preds_full(ir, th, mid, mask0, V7_CFG)
        dis_fixed = int(((pred_fixed >= 0) & (v7_preds >= 0) & (pred_fixed != v7_preds)).sum())
        honest_fixed = float(nest_fixed["mean"])
        nest_rt_fixedcfg = nested_retune(ir, th, mid, yt, yu, mask0, Ts_med, ngrid=13)
        honest_f = min(honest_fixed, float(nest_rt_fixedcfg["mean"]))
        row_f = {
            "ir": ir_name, "path": "fixed_v7cfg", "full": float(full_fixed), "cfg": dict(V7_CFG),
            "nested_fixed": honest_fixed, "nested_retune": float(nest_rt_fixedcfg["mean"]),
            "honest_nested": honest_f, "disagree_vs_v7": dis_fixed,
            "ir_solo": float((ir.argmax(1) == yt).mean()),
            "delta_nested": honest_f - float(v7_nest["mean"]),
            "clears": bool(honest_f >= NESTED_MIN and dis_fixed <= MAX_DISAGREE and honest_f >= float(v7_nest["mean"]) - 1e-9),
            "safe_tradeoff": bool(honest_f >= NESTED_MIN and dis_fixed <= SOFT_MAX_DISAGREE),
        }
        results.append(row_f)

        b_acc, bcfg = fuse3_sameT_near(ir, th, mid, yt, mask0, V7_CFG, Ts_near, ngrid=9, radius=0.10)
        if bcfg is None:
            continue
        nest_f = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask0, Ts_med, ngrid=13)
        pred = preds_full(ir, th, mid, mask0, bcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = min(float(nest_f["mean"]), float(nest_rt["mean"]))
        row = {
            "ir": ir_name, "path": "near_v7_sameT", "full": float(b_acc), "cfg": bcfg,
            "nested_fixed": float(nest_f["mean"]), "nested_retune": float(nest_rt["mean"]),
            "honest_nested": honest, "disagree_vs_v7": disagree,
            "ir_solo": float((ir.argmax(1) == yt).mean()),
            "delta_nested": honest - float(v7_nest["mean"]),
            "clears": bool(honest >= NESTED_MIN and disagree <= MAX_DISAGREE and honest >= float(v7_nest["mean"]) - 1e-9),
            "safe_tradeoff": bool(honest >= NESTED_MIN and disagree <= SOFT_MAX_DISAGREE),
        }
        results.append(row)
        if row_f["clears"] or row["clears"] or row_f["disagree_vs_v7"] <= 20 or row["disagree_vs_v7"] <= 20:
            print(f"{ir_name} fixed h={row_f['honest_nested']:.4f} d={row_f['disagree_vs_v7']} | "
                  f"near h={row['honest_nested']:.4f} d={row['disagree_vs_v7']} solo={row['ir_solo']:.4f}", flush=True)

    for aux_ir_name in ["c9_0.85_pa_0.15", "c9_0.80_pa_0.20", "c9_0.85_t42_0.15", "c9_0.80_ff_0.20",
                        "c9_0.75_ff_0.125_pa_0.125", "phaseA_raw"]:
        aux_ir = phaseA if aux_ir_name == "phaseA_raw" else pools[aux_ir_name]
        T = V7_CFG["T"]
        p_v7 = (V7_CFG["wa"] * softmax_np(classic9, T) + V7_CFG["wb"] * softmax_np(th, T) + V7_CFG["wc"] * softmax_np(mid, T))
        p_alt = (V7_CFG["wa"] * softmax_np(aux_ir, T) + V7_CFG["wb"] * softmax_np(th, T) + V7_CFG["wc"] * softmax_np(mid, T))
        for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            for maxch in (5, 10, 15, 20):
                conf = p_v7.max(1)
                pv, pa = p_v7.argmax(1), p_alt.argmax(1)
                cand = np.where(mask0 & (pv != pa) & (conf <= thr))[0]
                order = cand[np.argsort(conf[cand])][:maxch]
                pred = pv.copy()
                pred[order] = pa[order]
                yt_m = yt[mask0]
                pred_m = pred[mask0]
                full = float((pred_m == yt_m).mean())
                folds = []
                for leave in (8, 9, 24):
                    te = mask0 & (yu == leave)
                    if te.sum() < 5:
                        continue
                    folds.append(float((pred[te] == yt[te]).mean()))
                nested = float(np.mean(folds)) if folds else 0.0
                dis2 = int(((pred[mask0] != v7_preds[mask0]) & (v7_preds[mask0] >= 0)).sum())
                results.append({
                    "ir": f"fuse_cgate_{aux_ir_name}_thr{thr:.2f}_max{maxch}",
                    "path": "fuse_conf_gate",
                    "full": full,
                    "cfg": dict(V7_CFG),
                    "nested_fixed": nested,
                    "nested_retune": nested,
                    "honest_nested": nested,
                    "disagree_vs_v7": dis2,
                    "n_swaps": int(len(order)),
                    "ir_solo": float((aux_ir.argmax(1) == yt).mean()),
                    "delta_nested": nested - float(v7_nest["mean"]),
                    "clears": bool(nested >= NESTED_MIN and dis2 <= MAX_DISAGREE and nested >= float(v7_nest["mean"]) - 1e-9),
                    "safe_tradeoff": bool(nested >= NESTED_MIN and dis2 <= SOFT_MAX_DISAGREE),
                })

    ranked = sorted(results, key=lambda r: (r["clears"], r["honest_nested"], -r["disagree_vs_v7"], r["full"]), reverse=True)
    clears = [r for r in ranked if r["clears"]]
    safe = [r for r in ranked if r.get("safe_tradeoff")]
    best = clears[0] if clears else None
    if best is None:
        cand = [r for r in ranked if r["disagree_vs_v7"] <= MAX_DISAGREE and r["honest_nested"] >= float(v7_nest["mean"]) - 0.002]
        best = cand[0] if cand else ranked[0]

    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    status = {
        "tag": "ir_v27_safe_near_v7",
        "outcome": "WIN" if clears else "KEEP_V7",
        "keep_ir_v7": len(clears) == 0,
        "wrote_csv": False,
        "csv": None,
        "lesson": "v26 public 0.67164 worse than v7 0.69154 despite nested 0.75988 / 53 disagrees",
        "gate": {"nested_min": NESTED_MIN, "max_disagree": MAX_DISAGREE, "soft_max_disagree": SOFT_MAX_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "best": best,
        "n_clear": len(clears),
        "n_safe_tradeoff": len(safe),
        "top20": ranked[:20],
        "top_clears": clears[:10],
        "top_small_dis": sorted(
            [r for r in ranked if r["disagree_vs_v7"] <= SOFT_MAX_DISAGREE],
            key=lambda r: (r["honest_nested"], r["full"]), reverse=True
        )[:15],
        "scratch_available": scratch is not None,
        "strong_t24_scratch": "TRAINING",
        "elapsed_sec": round(time.time() - t0, 1),
        "updated_at": now,
        "next_roi": [
            "Finish from-scratch strong T24 s42 (aim solo>=0.71); re-run this probe with scratch logits",
            "Only submit if disagree_vs_v7<=15 and nested>=0.753 with lift",
            "Optional: seed888 strong only if s42 solo>=0.71",
            "Keep submission_ir_v7.csv as best public",
        ],
    }
    out = ROOT / "metrics_ir_v27_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v27_safe_near_v7.json").write_text(
        json.dumps({"best": best, "clears": clears[:20], "top40": ranked[:40], "n_pools_rows": len(results)}, indent=2),
        encoding="utf-8",
    )
    print("BEST", best.get("path"), best.get("ir"), "honest", best.get("honest_nested"),
          "dis", best.get("disagree_vs_v7"), "clears", best.get("clears"), "n_clear", len(clears), "->", out)


if __name__ == "__main__":
    main()
