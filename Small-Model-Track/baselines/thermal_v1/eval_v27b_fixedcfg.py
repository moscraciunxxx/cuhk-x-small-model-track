"""Quick eval near-v7 with classic mid + nested_fixed."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import softmax_np, apply_cfg, preds_full, nested_fixed, V7_CFG
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"

def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]

def near_cfg(ir, th, mid, yt, mask0):
    best = (-1.0, None)
    yt_m = yt[mask0]
    for T in (2.25, 2.5, 2.75, 3.0):
        pa = softmax_np(ir[mask0], T)
        pb = softmax_np(th[mask0], T)
        pc = softmax_np(mid[mask0], T)
        for wa in np.linspace(0.50, 0.62, 7):
            for wb in np.linspace(0.28, 0.40, 7):
                wc = 1 - wa - wb
                if not (0.05 <= wc <= 0.15):
                    continue
                if abs(wc - 0.09) > 0.06:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt_m).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T), "acc": acc, "mode": "near"})
    return best

def main():
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")[hold_idx].astype(np.float32)
    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    seeds = {int(p.stem.replace("hold_logits_seed", "")): np.load(p).astype(np.float32) for p in CK24.glob("hold_logits_seed*.npy")}
    phaseA = np.mean([seeds[42], seeds[888], seeds[2024]], 0).astype(np.float32)
    t24_42 = seeds[42]
    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)["mean"]
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    print(f"v7 full={v7_full:.6f} nested_fixed={v7_nest:.6f}", flush=True)

    pools = {
        "classic9": classic9,
        "c9_0.95_pa_0.05": (0.95 * classic9 + 0.05 * phaseA).astype(np.float32),
        "c9_0.90_pa_0.10": (0.90 * classic9 + 0.10 * phaseA).astype(np.float32),
        "c9_0.85_pa_0.15": (0.85 * classic9 + 0.15 * phaseA).astype(np.float32),
        "c9_0.95_t42_0.05": (0.95 * classic9 + 0.05 * t24_42).astype(np.float32),
        "c9_0.90_t42_0.10": (0.90 * classic9 + 0.10 * t24_42).astype(np.float32),
        "c9_0.85_t42_0.15": (0.85 * classic9 + 0.15 * t24_42).astype(np.float32),
        "c9_0.90_ff_0.10": (0.90 * classic9 + 0.10 * ff).astype(np.float32),
        "c9_0.85_ff_0.15": (0.85 * classic9 + 0.15 * ff).astype(np.float32),
        "c9_0.92_pa_0.04_t42_0.04": (0.92 * classic9 + 0.04 * phaseA + 0.04 * t24_42).astype(np.float32),
    }
    for thr in (0.45, 0.55, 0.65):
        for name, aux in (("pa", phaseA), ("t42", t24_42), ("ff", ff)):
            pp, ap = softmax_np(classic9, 2.5), softmax_np(aux, 2.5)
            conf = pp.max(1)
            pa, aa = pp.argmax(1), ap.argmax(1)
            cand = np.where((pa != aa) & (conf <= thr))[0]
            order = cand[np.argsort(conf[cand])][:15]
            out = classic9.copy()
            out[order] = aux[order]
            pools[f"cgate15_{name}_thr{thr}"] = out

    rows = []
    for ir_name, ir in pools.items():
        full_f, _ = apply_cfg(ir, th, mid, yt, mask0, V7_CFG)
        nest_f = nested_fixed(ir, th, mid, yt, yu, mask0, V7_CFG)["mean"]
        pred_f = preds_full(ir, th, mid, mask0, V7_CFG)
        dis_f = int(((pred_f >= 0) & (v7_preds >= 0) & (pred_f != v7_preds)).sum())
        rows.append({"ir": ir_name, "path": "fixed", "full": float(full_f), "nested": float(nest_f), "dis": dis_f,
                     "solo": float((ir.argmax(1) == yt).mean()), "delta": float(nest_f - v7_nest), "cfg": dict(V7_CFG)})
        bacc, bcfg = near_cfg(ir, th, mid, yt, mask0)
        if bcfg:
            nest_n = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)["mean"]
            pred_n = preds_full(ir, th, mid, mask0, bcfg)
            dis_n = int(((pred_n >= 0) & (v7_preds >= 0) & (pred_n != v7_preds)).sum())
            rows.append({"ir": ir_name, "path": "near", "full": float(bacc), "nested": float(nest_n), "dis": dis_n,
                         "solo": float((ir.argmax(1) == yt).mean()), "delta": float(nest_n - v7_nest), "cfg": bcfg})
        print(f"{ir_name} fixed nest={nest_f:.4f} full={full_f:.4f} d={dis_f}", flush=True)

    T = V7_CFG["T"]
    p_v7 = (V7_CFG["wa"] * softmax_np(classic9, T) + V7_CFG["wb"] * softmax_np(th, T) + V7_CFG["wc"] * softmax_np(mid, T))
    for aux_name, aux_ir in (("pa", phaseA), ("t42", t24_42), ("ff", ff),
                             ("c9pa", (0.85 * classic9 + 0.15 * phaseA).astype(np.float32)),
                             ("c9t42", (0.85 * classic9 + 0.15 * t24_42).astype(np.float32))):
        aux_ir = np.asarray(aux_ir, np.float32)
        p_alt = (V7_CFG["wa"] * softmax_np(aux_ir, T) + V7_CFG["wb"] * softmax_np(th, T) + V7_CFG["wc"] * softmax_np(mid, T))
        for thr in (0.40, 0.50, 0.60):
            for maxch in (5, 10, 15):
                conf = p_v7.max(1)
                pv, pa = p_v7.argmax(1), p_alt.argmax(1)
                cand = np.where(mask0 & (pv != pa) & (conf <= thr))[0]
                order = cand[np.argsort(conf[cand])][:maxch]
                pred = pv.copy(); pred[order] = pa[order]
                full = float((pred[mask0] == yt[mask0]).mean())
                folds = [float((pred[mask0 & (yu == leave)] == yt[mask0 & (yu == leave)]).mean())
                         for leave in (8, 9, 24) if (mask0 & (yu == leave)).sum() >= 5]
                nested = float(np.mean(folds)) if folds else 0.0
                dis = int(((pred[mask0] != v7_preds[mask0]) & (v7_preds[mask0] >= 0)).sum())
                rows.append({"ir": f"fcgate_{aux_name}_t{thr}_m{maxch}", "path": "fcgate", "full": full, "nested": nested,
                             "dis": dis, "solo": float((aux_ir.argmax(1) == yt).mean()), "delta": nested - v7_nest,
                             "cfg": dict(V7_CFG), "nswaps": len(order)})

    clears = sorted([r for r in rows if r["nested"] >= v7_nest - 1e-12 and r["dis"] <= 15 and r["nested"] >= 0.75196],
                    key=lambda r: (r["nested"], -r["dis"], r["full"]), reverse=True)
    small = sorted([r for r in rows if r["dis"] <= 15], key=lambda r: (r["nested"], r["full"]), reverse=True)
    print("n_rows", len(rows), "n_clear", len(clears), flush=True)
    for r in clears[:12]:
        print(f"CLEAR {r['path']:6s} {r['ir'][:48]:48s} nest={r['nested']:.5f} full={r['full']:.5f} d={r['dis']:2d} delta={r['delta']:+.5f}", flush=True)
    for r in small[:12]:
        print(f"SMALL {r['path']:6s} {r['ir'][:48]:48s} nest={r['nested']:.5f} full={r['full']:.5f} d={r['dis']:2d} delta={r['delta']:+.5f}", flush=True)
    out = {"tag": "ir_v27b_fixedcfg_classic_mid", "v7_reproduce": {"full": float(v7_full), "nested_fixed": float(v7_nest)},
           "n_clear": len(clears), "clears": clears[:20], "top_small_dis": small[:20], "keep_ir_v7": len(clears) == 0, "wrote_csv": False}
    (ROOT / "metrics_ir_v27b_fixedcfg.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote metrics_ir_v27b_fixedcfg.json", flush=True)

if __name__ == "__main__":
    main()
