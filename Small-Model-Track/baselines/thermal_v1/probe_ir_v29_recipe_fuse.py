"""Quick v29 recipe fuse probe vs SAFE v29b / v7 gate. Classic mid only."""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import softmax_np, apply_cfg, preds_full, nested_fixed, V7_CFG
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
CK29 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v29_recipe"
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
NESTED_MIN = 0.75196
MAX_DIS = 15
PT = timezone(timedelta(hours=-7))


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def selective_swap(primary, aux, T, pmax, amin, maxch):
    pp, ap = softmax_np(primary, T), softmax_np(aux, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    pconf, aconf = pp.max(1), ap.max(1)
    cand = np.where((pa != aa) & (pconf <= pmax) & (aconf >= amin))[0]
    order = cand[np.lexsort((-aconf[cand], pconf[cand]))][:maxch]
    out = primary.copy()
    for i in order:
        out[i] = aux[i]
    return out, len(order)


def main():
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)
    t24 = np.load(CK24 / "hold_logits_seed42.npy").astype(np.float32)
    v29_4096 = np.load(CK29 / "hold_logits_seed4096.npy").astype(np.float32)
    v29_ens = np.load(CK29 / "hold_logits_v29.npz")["ens"].astype(np.float32)
    mask0 = th.any(1) & mid.any(1)
    T = V7_CFG["T"]

    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)

    pools = {
        "classic9": classic9,
        "v29_s4096": v29_4096,
        "v29_ens": v29_ens,
        "c9_0.85_v29_0.15": (0.85 * classic9 + 0.15 * v29_4096).astype(np.float32),
        "c9_0.80_v29_0.20": (0.80 * classic9 + 0.20 * v29_4096).astype(np.float32),
        "c9_0.90_v29_0.10": (0.90 * classic9 + 0.10 * v29_4096).astype(np.float32),
        "c9_0.85_t24_0.15": (0.85 * classic9 + 0.15 * t24).astype(np.float32),
    }
    ir_sw, n = selective_swap(classic9, t24, T, 0.40, 0.55, 15)
    pools["v29b_errdrive"] = ir_sw
    ir_sw2, n2 = selective_swap(classic9, v29_4096, T, 0.40, 0.55, 15)
    pools[f"cgate_c9_v29_n{n2}"] = ir_sw2

    results = []
    for name, ir in pools.items():
        full, _ = apply_cfg(ir, th, mid, yt, mask0, V7_CFG)
        nest = nested_fixed(ir, th, mid, yt, yu, mask0, V7_CFG)
        pred = preds_full(ir, th, mid, mask0, V7_CFG)
        dis = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = float(nest["mean"])
        row = {
            "ir": name, "full": float(full), "nested_fixed": honest,
            "disagree_vs_v7": dis, "ir_solo": float((ir.argmax(1) == yt).mean()),
            "delta_nested": honest - float(v7_nest["mean"]),
            "clears": bool(honest >= NESTED_MIN and dis <= MAX_DIS and honest >= float(v7_nest["mean"]) - 1e-12),
        }
        results.append(row)
        print(f"{name}: solo={row['ir_solo']:.4f} full={row['full']:.4f} nest={row['nested_fixed']:.4f} dis={dis} clears={row['clears']}", flush=True)

    ranked = sorted(results, key=lambda r: (r["clears"], r["nested_fixed"], -r["disagree_vs_v7"]), reverse=True)
    clears = [r for r in ranked if r["clears"]]
    out = {
        "tag": "ir_v29_recipe_fuse_probe",
        "updated_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "recipe_members": {"s2024": 0.6554, "s4096": 0.6851, "ens": 0.6832},
        "recipe_verdict": "MISS_solo_below_0.72",
        "v7_reproduce": {"full": float(v7_full), "nested_fixed": float(v7_nest["mean"])},
        "n_clear": len(clears), "best": ranked[0], "clears": clears[:5], "all": ranked,
        "keep_prior_safe": "submission_ir_v29b.csv",
    }
    p = Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\metrics_ir_v29_recipe_fuse.json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("BEST", ranked[0], "n_clear", len(clears), "->", p, flush=True)


if __name__ == "__main__":
    main()