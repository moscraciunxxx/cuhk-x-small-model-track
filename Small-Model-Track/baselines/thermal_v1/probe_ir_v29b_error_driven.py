"""ir_v29b error-driven: swap only when T24 s42 high-conf and v7/c9 low-conf; cap <=15; classic mid; nested gate."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, apply_cfg, preds_full, nested_fixed, V7_CFG, V7_HOLD,
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
NESTED_MIN = 0.75196
MAX_DIS = 15


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def selective_swap(primary_logits, aux_logits, T=2.5, primary_conf_max=0.55,
                   aux_conf_min=0.70, max_changes=15):
    """Swap primary->aux only where primary low-conf, aux high-conf, and they disagree."""
    pp = softmax_np(primary_logits, T)
    ap = softmax_np(aux_logits, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    pconf, aconf = pp.max(1), ap.max(1)
    cand = np.where((pa != aa) & (pconf <= primary_conf_max) & (aconf >= aux_conf_min))[0]
    # prioritize: lowest primary conf, then highest aux conf
    order = cand[np.lexsort((-aconf[cand], pconf[cand]))]
    out = primary_logits.copy()
    changed = []
    for i in order:
        if len(changed) >= max_changes:
            break
        out[i] = aux_logits[i]
        changed.append(int(i))
    return out, changed


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
    t24_42 = np.load(CK24 / "hold_logits_seed42.npy").astype(np.float32)

    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)
    print(f"c9 solo={(classic9.argmax(1)==yt).mean():.4f} t24s42={(t24_42.argmax(1)==yt).mean():.4f}", flush=True)

    # Analyze error overlap on hold
    T = V7_CFG["T"]
    p_v7 = (V7_CFG["wa"] * softmax_np(classic9, T) + V7_CFG["wb"] * softmax_np(th, T)
            + V7_CFG["wc"] * softmax_np(mid, T))
    c9p, t24p = classic9.argmax(1), t24_42.argmax(1)
    v7_wrong = (v7_preds != yt) & (v7_preds >= 0) & mask0
    t24_right = t24p == yt
    both_disagree = c9p != t24p
    print(f"v7_wrong={int(v7_wrong.sum())} of which t24_right={int((v7_wrong & t24_right).sum())} "
          f"c9_vs_t24_disagree={int(both_disagree.sum())}", flush=True)

    results = []
    grids = []
    for pmax in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for amin in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            for maxch in [5, 8, 10, 12, 15]:
                grids.append((pmax, amin, maxch))

    # Also pure IR-logit swap then v7 cfg fuse
    for pmax, amin, maxch in grids:
        ir_sw, ch = selective_swap(classic9, t24_42, T=T, primary_conf_max=pmax,
                                   aux_conf_min=amin, max_changes=maxch)
        if not ch:
            continue
        full, _ = apply_cfg(ir_sw, th, mid, yt, mask0, V7_CFG)
        nest = nested_fixed(ir_sw, th, mid, yt, yu, mask0, V7_CFG)
        pred = preds_full(ir_sw, th, mid, mask0, V7_CFG)
        dis = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = float(nest["mean"])
        row = {
            "path": "ir_logit_swap_then_v7cfg",
            "pmax": pmax, "amin": amin, "maxch": maxch, "n_swapped": len(ch),
            "full": float(full), "nested_fixed": honest,
            "disagree_vs_v7": dis,
            "delta_nested": honest - float(v7_nest["mean"]),
            "ir_solo": float((ir_sw.argmax(1) == yt).mean()),
            "clears": bool(honest >= NESTED_MIN and dis <= MAX_DIS and honest >= float(v7_nest["mean"]) - 1e-12),
        }
        results.append(row)

    # Fuse-space selective: swap fused softmax predictions (more direct on final)
    for pmax, amin, maxch in grids:
        pp = p_v7
        ap = (V7_CFG["wa"] * softmax_np(t24_42, T) + V7_CFG["wb"] * softmax_np(th, T)
              + V7_CFG["wc"] * softmax_np(mid, T))
        pa, aa = pp.argmax(1), ap.argmax(1)
        pconf, aconf = pp.max(1), ap.max(1)
        cand = np.where(mask0 & (pa != aa) & (pconf <= pmax) & (aconf >= amin))[0]
        order = cand[np.lexsort((-aconf[cand], pconf[cand]))][:maxch]
        if len(order) == 0:
            continue
        pred = v7_preds.copy()
        for i in order:
            pred[i] = aa[i]
        # nested LOUO by user
        users = yu
        folds = []
        for u in np.unique(users[mask0]):
            m = mask0 & (users != u)
            # rebuild with same swaps but only evaluate on fold - use pred already global
            folds.append(float((pred[m] == yt[m]).mean()) if m.sum() else 0.0)
        # Actually LOUO should exclude swaps? Honest: apply swap rule fitted on all is optimistic.
        # Better nested: for each left-out user, re-select swaps on train users only, eval on left-out.
        nest_folds = []
        for u in np.unique(users):
            tr = mask0 & (users != u)
            te = mask0 & (users == u)
            if te.sum() == 0:
                continue
            cand_u = np.where(tr & (pa != aa) & (pconf <= pmax) & (aconf >= amin))[0]
            order_u = cand_u[np.lexsort((-aconf[cand_u], pconf[cand_u]))][:maxch]
            pred_u = v7_preds.copy()
            for i in order_u:
                pred_u[i] = aa[i]
            nest_folds.append(float((pred_u[te] == yt[te]).mean()))
        honest = float(np.mean(nest_folds)) if nest_folds else 0.0
        dis = int(((pred[mask0] != v7_preds[mask0]) & (v7_preds[mask0] >= 0)).sum())
        full = float((pred[mask0] == yt[mask0]).mean())
        row = {
            "path": "fuse_pred_swap_nested_reselect",
            "pmax": pmax, "amin": amin, "maxch": maxch, "n_swapped": int(len(order)),
            "full": full, "nested_fixed": honest,
            "disagree_vs_v7": dis,
            "delta_nested": honest - float(v7_nest["mean"]),
            "ir_solo": None,
            "clears": bool(honest >= NESTED_MIN and dis <= MAX_DIS and honest >= float(v7_nest["mean"]) - 1e-12),
        }
        results.append(row)

    ranked = sorted(results, key=lambda r: (r["clears"], r["nested_fixed"], -r["disagree_vs_v7"], r["full"]), reverse=True)
    clears = [r for r in ranked if r["clears"]]
    near = [r for r in ranked if r["disagree_vs_v7"] <= MAX_DIS][:15]
    best = clears[0] if clears else (near[0] if near else ranked[0])
    out = {
        "tag": "ir_v29b_error_driven",
        "updated_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "outcome": "WIN" if clears else "MISS_KEEP_V7",
        "keep_ir_v7": len(clears) == 0,
        "wrote_csv": False,
        "public_submit": None,
        "promote_gate": {"nested_fixed_min": NESTED_MIN, "disagree_vs_v7_max": MAX_DIS, "mid": "classic_aligned_mid"},
        "v7_reproduce": {"full": float(v7_full), "nested_fixed": float(v7_nest["mean"])},
        "error_stats": {
            "v7_wrong": int(v7_wrong.sum()),
            "v7_wrong_t24_right": int((v7_wrong & t24_right).sum()),
            "c9_t24_disagree": int(both_disagree.sum()),
            "t24_s42_solo": float((t24_42.argmax(1) == yt).mean()),
            "c9_solo": float((classic9.argmax(1) == yt).mean()),
        },
        "n_clear": len(clears),
        "best": best,
        "top_near": near[:10],
        "clears": clears[:5],
        "elapsed_s": time.time() - t0,
        "next_roi": [
            "Keep ir_v7 unless clears",
            "Await ir_v29 recipe train (classic cache mixup0.1 freeze@10)",
            "No kaggle submit without SAFE win",
        ],
    }
    path = ROOT / "metrics_ir_v29b_error_driven.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("BEST", best.get("path"), "nested", best.get("nested_fixed"), "dis", best.get("disagree_vs_v7"),
          "clears", best.get("clears"), "n_clear", len(clears), "->", path, flush=True)
    if clears:
        print("CLEAR CANDIDATES", json.dumps(clears[:3], indent=2), flush=True)


if __name__ == "__main__":
    main()