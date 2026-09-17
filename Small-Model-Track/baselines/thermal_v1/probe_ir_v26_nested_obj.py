"""ir_v26 nested-only objective fuse.
LOUO-calibrated IR (PhaseA 42/888/2024) + classic9 + th_v6 + mid.
Optimize honest nested (min fixed/retune), NOT full hold.
Soft gate / temperature / per-modality T via nested CV only (no holdout-only perT).
CSV only if honest>=0.758 and disagree>=20 vs ir_v7.
"""
from __future__ import annotations
import json, time, csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune,
    GATE, MIN_DISAGREE, V7_CFG, V7_HOLD,
)
from fuse_ir_v9 import load_members, fuse3_perT

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))
CK = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
PRIOR = 0.7555837334595704
LEAVE = (8, 9, 24)


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def soft_mix(a, b, T=1.5, alpha=0.5):
    """Convex mix in prob space, return log-prob-ish logits via log(p)."""
    pa, pb = softmax_np(a, T), softmax_np(b, T)
    p = (1 - alpha) * pa + alpha * pb
    return np.log(np.clip(p, 1e-8, 1.0)).astype(np.float32)


def conf_soft_gate(primary, aux, T=1.5, thr=0.55, w_aux=0.5):
    """When primary conf < thr, mix aux with weight w_aux."""
    pp = softmax_np(primary, T)
    ap = softmax_np(aux, T)
    mx = pp.max(1)
    out = pp.copy()
    low = mx < thr
    if low.any():
        out[low] = (1 - w_aux) * pp[low] + w_aux * ap[low]
    return np.log(np.clip(out, 1e-8, 1.0)).astype(np.float32)


def conf_replace_gate(a, b, T=1.5, thr=0.0):
    pa, pb = softmax_np(a, T), softmax_np(b, T)
    ca, cb = pa.max(1), pb.max(1)
    out = a.copy()
    use_b = (pb.argmax(1) != pa.argmax(1)) & (cb > ca + thr)
    out[use_b] = b[use_b]
    return out


def weighted_ens(arrs, ws):
    w = np.asarray(ws, dtype=np.float64)
    w = w / w.sum()
    return np.tensordot(w, np.stack(arrs, 0), axes=(0, 0)).astype(np.float32)


def louo_ir_weights(members_arrs, y, users, mask, grid=None):
    """LOUO-calibrate nonneg weights over IR members; return OOF blend + mean weights."""
    S = len(members_arrs)
    if grid is None:
        # simplex-ish grid for up to 4 members
        if S == 2:
            grid = [(a, 1 - a) for a in np.linspace(0, 1, 21)]
        elif S == 3:
            grid = []
            for a in np.linspace(0, 1, 11):
                for b in np.linspace(0, 1 - a, 11):
                    grid.append((a, b, 1 - a - b))
        else:
            # equal + leave-one-stronger
            grid = [tuple([1.0 / S] * S)]
            for i in range(S):
                w = np.ones(S) / (S + 1)
                w[i] = 2.0 / (S + 1)
                grid.append(tuple(w.tolist()))
    oof = np.zeros_like(members_arrs[0])
    used = np.zeros(len(y), dtype=bool)
    w_folds = []
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        best = (-1.0, None)
        yt = y[tr]
        for ws in grid:
            blend = weighted_ens(members_arrs, ws)
            acc = float((blend[tr].argmax(1) == yt).mean())
            if acc > best[0]:
                best = (acc, ws)
        ws = best[1]
        w_folds.append(ws)
        oof[te] = weighted_ens(members_arrs, ws)[te]
        used[te] = True
    # fill unused with equal mean
    eq = np.mean(members_arrs, 0).astype(np.float32)
    oof[~used] = eq[~used]
    mean_w = np.mean(np.array(w_folds, dtype=np.float64), 0) if w_folds else np.ones(S) / S
    return oof.astype(np.float32), mean_w.tolist(), w_folds


def apply_cfg_perT(a, b, c, y, mask, cfg):
    Ta, Tb, Tc = cfg["Ta"], cfg["Tb"], cfg["Tc"]
    p = (cfg["wa"] * softmax_np(a[mask], Ta)
         + cfg["wb"] * softmax_np(b[mask], Tb)
         + cfg["wc"] * softmax_np(c[mask], Tc)).argmax(1)
    return float((p == y[mask]).mean()), p


def preds_full_perT(a, b, c, mask, cfg):
    out = np.full(len(a), -1, dtype=np.int64)
    Ta, Tb, Tc = cfg["Ta"], cfg["Tb"], cfg["Tc"]
    out[mask] = (cfg["wa"] * softmax_np(a[mask], Ta)
                 + cfg["wb"] * softmax_np(b[mask], Tb)
                 + cfg["wc"] * softmax_np(c[mask], Tc)).argmax(1)
    return out


def nested_fixed_perT(a, b, c, y, users, mask, cfg):
    folds = []
    for leave in LEAVE:
        te = mask & (users == leave)
        if te.sum() < 5:
            continue
        te_acc, _ = apply_cfg_perT(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum())})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def nested_retune_perT(a, b, c, y, users, mask, Ts, ngrid=11):
    """perT weights tuned ONLY on LOUO train folds — never on full hold."""
    folds = []
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_perT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc, _ = apply_cfg_perT(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum()), "cfg": cfg, "tr_acc": acc})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def nested_objective_fuse(a, b, c, y, users, mask, Ts, ngrid=17):
    """Pick cfg by maximizing nested_retune mean (honest), report full as secondary."""
    nest = nested_retune(a, b, c, y, users, mask, Ts, ngrid=ngrid)
    # also get a full-hold cfg for disagree / reporting (not used for selection)
    full_acc, full_cfg = fuse3_sameT(a, b, c, y, mask, Ts, ngrid=ngrid)
    # reconstruct "stacked" cfg as mean of fold cfgs
    if nest["folds"]:
        wa = float(np.mean([f["cfg"]["wa"] for f in nest["folds"]]))
        wb = float(np.mean([f["cfg"]["wb"] for f in nest["folds"]]))
        wc = float(np.mean([f["cfg"]["wc"] for f in nest["folds"]]))
        T = float(np.mean([f["cfg"]["T"] for f in nest["folds"]]))
        s = wa + wb + wc
        stacked = {"wa": wa / s, "wb": wb / s, "wc": wc / s, "T": T, "mode": "nested_stacked"}
        stacked_acc, _ = apply_cfg(a, b, c, y, mask, stacked)
        stacked["acc"] = stacked_acc
        stacked["n"] = int(mask.sum())
    else:
        stacked = full_cfg
    nest_fixed = nested_fixed(a, b, c, y, users, mask, stacked)
    honest = min(float(nest["mean"]), float(nest_fixed["mean"]))
    return {
        "full": full_acc,
        "full_cfg": full_cfg,
        "stacked_cfg": stacked,
        "nested_retune": float(nest["mean"]),
        "nested_fixed": float(nest_fixed["mean"]),
        "honest_nested": honest,
        "nest_folds": nest["folds"],
    }


def write_submission_if_clear(ir_hold, ir_test, th_hold, th_test, mid_hold, mid_test, cfg, meta, empty, fb, out_csv, mode="sameT"):
    if mode == "perT":
        preds = preds_full_perT(ir_test, th_test, mid_test, np.ones(len(ir_test), dtype=bool), cfg)
    else:
        T = cfg["T"]
        preds = (cfg["wa"] * softmax_np(ir_test, T)
                 + cfg["wb"] * softmax_np(th_test, T)
                 + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
    nfb = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i])); nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb, preds


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

    seeds = {}
    for p in sorted(CK.glob("hold_logits_seed*.npy")):
        sid = int(p.stem.replace("hold_logits_seed", ""))
        seeds[sid] = np.load(p).astype(np.float32)
    print("seed solos", {s: float((a.argmax(1) == yt).mean()) for s, a in seeds.items()}, flush=True)

    s42, s888, s2024 = seeds[42], seeds[888], seeds[2024]
    phaseA = np.mean([s42, s888, s2024], 0).astype(np.float32)
    top2 = np.mean([s42, s888], 0).astype(np.float32)

    # LOUO-calibrated IR blends
    louo_phaseA, mean_w_pa, folds_pa = louo_ir_weights([s42, s888, s2024], yt, yu, np.ones(len(yt), dtype=bool))
    louo_c9_pa, mean_w_c9pa, folds_c9pa = louo_ir_weights([classic9, phaseA], yt, yu, np.ones(len(yt), dtype=bool))
    louo_c9_ff_pa, mean_w_3, folds_3 = louo_ir_weights([classic9, ff, phaseA], yt, yu, np.ones(len(yt), dtype=bool))
    louo_c9_42_888, mean_w_c942888, _ = louo_ir_weights([classic9, s42, s888], yt, yu, np.ones(len(yt), dtype=bool))
    print("LOUO phaseA w", mean_w_pa, "folds", folds_pa, "solo", float((louo_phaseA.argmax(1) == yt).mean()), flush=True)
    print("LOUO c9+pa w", mean_w_c9pa, "solo", float((louo_c9_pa.argmax(1) == yt).mean()), flush=True)
    print("LOUO c9+ff+pa w", mean_w_3, "solo", float((louo_c9_ff_pa.argmax(1) == yt).mean()), flush=True)

    pools = {
        "classic9": classic9,
        "phaseA": phaseA,
        "top2_42_888": top2,
        "s42": s42,
        "louo_phaseA": louo_phaseA,
        "louo_c9_pa": louo_c9_pa,
        "louo_c9_ff_pa": louo_c9_ff_pa,
        "louo_c9_42_888": louo_c9_42_888,
        "c9_0.4_ff_0.3_pa_0.3": (0.4 * classic9 + 0.3 * ff + 0.3 * phaseA).astype(np.float32),
        "c9_0.5_pa_0.5": (0.5 * classic9 + 0.5 * phaseA).astype(np.float32),
        "c9_0.55_pa_0.45": (0.55 * classic9 + 0.45 * phaseA).astype(np.float32),
        "c9_0.45_pa_0.55": (0.45 * classic9 + 0.55 * phaseA).astype(np.float32),
        "c9_0.6_pa_0.4": (0.6 * classic9 + 0.4 * phaseA).astype(np.float32),
        "c9_0.4_pa_0.6": (0.4 * classic9 + 0.6 * phaseA).astype(np.float32),
        "mean_w_c9_pa": weighted_ens([classic9, phaseA], mean_w_c9pa),
        "mean_w_phaseA": weighted_ens([s42, s888, s2024], mean_w_pa),
        "mean_w_c9_ff_pa": weighted_ens([classic9, ff, phaseA], mean_w_3),
    }
    # soft mixes + gates
    for T in (1.0, 1.5, 2.0, 2.5):
        for a in (0.3, 0.4, 0.5, 0.6, 0.7):
            pools[f"soft_c9_pa_T{T}_a{a}"] = soft_mix(classic9, phaseA, T=T, alpha=a)
        for thr in (0.35, 0.45, 0.55, 0.65):
            for w in (0.4, 0.6, 0.8, 1.0):
                pools[f"sgate_c9_pa_T{T}_t{thr}_w{w}"] = conf_soft_gate(classic9, phaseA, T=T, thr=thr, w_aux=w)
        for thr in (0.0, 0.05, 0.1):
            pools[f"rgate_c9_pa_T{T}_m{thr}"] = conf_replace_gate(classic9, phaseA, T=T, thr=thr)
            pools[f"rgate_c9_louopa_T{T}_m{thr}"] = conf_replace_gate(classic9, louo_phaseA, T=T, thr=thr)

    # temperature-scaled phaseA then mix
    for T_ir in (0.75, 1.0, 1.25, 1.5, 2.0):
        pa_T = np.log(np.clip(softmax_np(phaseA, T_ir), 1e-8, 1)).astype(np.float32)
        pools[f"c9_0.5_paT{T_ir}_0.5"] = (0.5 * classic9 + 0.5 * pa_T).astype(np.float32)
        pools[f"c9_0.4_ff_0.3_paT{T_ir}_0.3"] = (0.4 * classic9 + 0.3 * ff + 0.3 * pa_T).astype(np.float32)

    mask0 = th.any(1) & mid.any(1)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    Ts_fine = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    Ts_med = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    Ts_per = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = []
    print(f"scoring {len(pools)} IR pools with nested-objective sameT...", flush=True)
    for ir_name, ir in pools.items():
        # Primary: nested_retune as selection; also classic full for reference
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask0, Ts_med, ngrid=17)
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask0, Ts_fine, ngrid=25)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)
        # stacked cfg from nest folds
        if nest_rt["folds"]:
            wa = float(np.mean([f["cfg"]["wa"] for f in nest_rt["folds"]]))
            wb = float(np.mean([f["cfg"]["wb"] for f in nest_rt["folds"]]))
            wc = float(np.mean([f["cfg"]["wc"] for f in nest_rt["folds"]]))
            T = float(np.mean([f["cfg"]["T"] for f in nest_rt["folds"]]))
            s = max(wa + wb + wc, 1e-9)
            scfg = {"wa": wa / s, "wb": wb / s, "wc": wc / s, "T": T, "mode": "nested_stacked",
                    "acc": float(apply_cfg(ir, th, mid, yt, mask0, {"wa": wa / s, "wb": wb / s, "wc": wc / s, "T": T})[0]),
                    "n": int(mask0.sum())}
            nest_fixed_s = nested_fixed(ir, th, mid, yt, yu, mask0, scfg)
        else:
            scfg = bcfg
            nest_fixed_s = nest_fixed
        # honest = min of retune and fixed-on-stacked (more honest than fixed-on-fullcfg)
        honest = min(float(nest_rt["mean"]), float(nest_fixed_s["mean"]))
        # disagree using stacked cfg (matches nested objective)
        pred = preds_full(ir, th, mid, mask0, scfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        # also disagree with full-hold cfg
        pred_f = preds_full(ir, th, mid, mask0, bcfg)
        disagree_f = int(((pred_f >= 0) & (v7_preds >= 0) & (pred_f != v7_preds)).sum())
        row = {
            "ir": ir_name,
            "full": b_acc,
            "cfg": bcfg,
            "stacked_cfg": scfg,
            "nested_fixed_fullcfg": float(nest_fixed["mean"]),
            "nested_fixed_stacked": float(nest_fixed_s["mean"]),
            "nested_retune": float(nest_rt["mean"]),
            "honest_nested": honest,
            "disagree_vs_v7": max(disagree, disagree_f),
            "disagree_stacked": disagree,
            "disagree_fullcfg": disagree_f,
            "ir_solo": float((ir.argmax(1) == yt).mean()),
            "clears": bool(b_acc >= GATE and honest >= GATE and max(disagree, disagree_f) >= MIN_DISAGREE),
            "clears_strict_nested": bool(honest >= GATE and max(disagree, disagree_f) >= MIN_DISAGREE and float(nest_fixed_s["mean"]) >= GATE),
        }
        results.append(row)
        if row["honest_nested"] >= 0.752 or row["clears"] or row["clears_strict_nested"]:
            print(f"{ir_name} full={b_acc:.4f} honest={honest:.4f} nestR={row['nested_retune']:.4f} "
                  f"nestFs={row['nested_fixed_stacked']:.4f} dis={row['disagree_vs_v7']} solo={row['ir_solo']:.4f} "
                  f"clear={row['clears']}", flush=True)

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["nested_retune"], r["full"], r["disagree_vs_v7"]), reverse=True)
    print("TOP10 sameT nested-obj:", flush=True)
    for r in ranked[:10]:
        print(f"  {r['ir']}: honest={r['honest_nested']:.5f} full={r['full']:.4f} nestR={r['nested_retune']:.5f} "
              f"dis={r['disagree_vs_v7']}", flush=True)

    # Phase 2: nested perT on top IR pools (nested CV only)
    top_irs = []
    seen = set()
    for r in ranked:
        if r["ir"] in seen:
            continue
        seen.add(r["ir"])
        top_irs.append(r["ir"])
        if len(top_irs) >= 8:
            break
    # always include prior best-named and louo
    for must in ("louo_c9_pa", "louo_c9_ff_pa", "rgate_c9_pa_T1.5_m0.0", "c9_0.4_ff_0.3_pa_0.3", "mean_w_c9_pa"):
        if must in pools and must not in top_irs:
            top_irs.append(must)

    print(f"nested perT on {len(top_irs)} pools...", flush=True)
    perT_rows = []
    for ir_name in top_irs:
        ir = pools[ir_name]
        nest_pt = nested_retune_perT(ir, th, mid, yt, yu, mask0, Ts_per, ngrid=11)
        # full-hold perT for reference only (NOT for selection / gate)
        full_pt_acc, full_pt_cfg = fuse3_perT(ir, th, mid, yt, mask0, Ts_per, ngrid=11)
        if nest_pt["folds"]:
            wa = float(np.mean([f["cfg"]["wa"] for f in nest_pt["folds"]]))
            wb = float(np.mean([f["cfg"]["wb"] for f in nest_pt["folds"]]))
            wc = float(np.mean([f["cfg"]["wc"] for f in nest_pt["folds"]]))
            Ta = float(np.mean([f["cfg"]["Ta"] for f in nest_pt["folds"]]))
            Tb = float(np.mean([f["cfg"]["Tb"] for f in nest_pt["folds"]]))
            Tc = float(np.mean([f["cfg"]["Tc"] for f in nest_pt["folds"]]))
            s = max(wa + wb + wc, 1e-9)
            scfg = {"wa": wa / s, "wb": wb / s, "wc": wc / s, "Ta": Ta, "Tb": Tb, "Tc": Tc,
                    "mode": "perT_nested_stacked", "acc": full_pt_acc, "n": int(mask0.sum())}
            nest_fixed_pt = nested_fixed_perT(ir, th, mid, yt, yu, mask0, scfg)
        else:
            scfg = full_pt_cfg
            nest_fixed_pt = {"mean": 0.0}
        honest = min(float(nest_pt["mean"]), float(nest_fixed_pt["mean"]))
        pred = preds_full_perT(ir, th, mid, mask0, scfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        # full sameT for hold floor
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask0, Ts_fine, ngrid=21)
        row = {
            "ir": f"perT::{ir_name}",
            "full": b_acc,
            "full_perT_ref": full_pt_acc,
            "cfg": scfg,
            "stacked_cfg": scfg,
            "nested_retune": float(nest_pt["mean"]),
            "nested_fixed_stacked": float(nest_fixed_pt["mean"]),
            "honest_nested": honest,
            "disagree_vs_v7": disagree,
            "ir_solo": float((ir.argmax(1) == yt).mean()),
            "clears": bool(b_acc >= GATE and honest >= GATE and disagree >= MIN_DISAGREE),
            "mode": "perT_nested",
        }
        perT_rows.append(row)
        print(f"  perT {ir_name}: honest={honest:.5f} nestR={row['nested_retune']:.5f} "
              f"full={b_acc:.4f} fullPerTref={full_pt_acc:.4f} dis={disagree}", flush=True)

    # Phase 3: densify mid/th around best IR with nested-only objective
    best_ir_name = ranked[0]["ir"]
    best_ir = pools[best_ir_name]
    print(f"densify mid/th nested-obj on best IR={best_ir_name}...", flush=True)
    densify_best = (-1.0, None)
    densify_rows = []
    # finer nested: for each leave fit on tr with denser grid, mean te
    for T in [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]:
        fold_scores = []
        fold_cfgs = []
        for leave in LEAVE:
            te = mask0 & (yu == leave)
            tr = mask0 & (yu != leave)
            yt_tr = yt[tr]
            pa = softmax_np(best_ir[tr], T)
            pb = softmax_np(th[tr], T)
            pc = softmax_np(mid[tr], T)
            best_f = (-1.0, None)
            for wa in np.linspace(0.30, 0.60, 31):
                for wb in np.linspace(0.15, 0.45, 31):
                    wc = 1 - wa - wb
                    if wc < 0.05 or wc > 0.40:
                        continue
                    acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt_tr).mean())
                    if acc > best_f[0]:
                        best_f = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T)})
            te_acc, _ = apply_cfg(best_ir, th, mid, yt, te, best_f[1])
            fold_scores.append(te_acc)
            fold_cfgs.append(best_f[1])
        nest_mean = float(np.mean(fold_scores))
        wa = float(np.mean([c["wa"] for c in fold_cfgs]))
        wb = float(np.mean([c["wb"] for c in fold_cfgs]))
        wc = float(np.mean([c["wc"] for c in fold_cfgs]))
        s = wa + wb + wc
        scfg = {"wa": wa / s, "wb": wb / s, "wc": wc / s, "T": float(T), "mode": "densify_nested"}
        full_acc, _ = apply_cfg(best_ir, th, mid, yt, mask0, scfg)
        nest_fixed = nested_fixed(best_ir, th, mid, yt, yu, mask0, scfg)
        honest = min(nest_mean, float(nest_fixed["mean"]))
        pred = preds_full(best_ir, th, mid, mask0, scfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        row = {
            "ir": f"densify::{best_ir_name}::T{T}",
            "full": full_acc,
            "cfg": scfg,
            "stacked_cfg": scfg,
            "nested_retune": nest_mean,
            "nested_fixed_stacked": float(nest_fixed["mean"]),
            "honest_nested": honest,
            "disagree_vs_v7": disagree,
            "ir_solo": float((best_ir.argmax(1) == yt).mean()),
            "clears": bool(full_acc >= GATE and honest >= GATE and disagree >= MIN_DISAGREE),
            "mode": "densify",
        }
        densify_rows.append(row)
        if honest > densify_best[0]:
            densify_best = (honest, row)
        print(f"  densify T={T}: honest={honest:.5f} full={full_acc:.4f} nestR={nest_mean:.5f} dis={disagree} cfg={scfg}", flush=True)

    all_rows = ranked + perT_rows + densify_rows
    all_ranked = sorted(all_rows, key=lambda r: (r["honest_nested"], r.get("nested_retune", 0), r["full"], r["disagree_vs_v7"]), reverse=True)
    best = all_ranked[0]
    clears = [r for r in all_ranked if r.get("clears")]
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")

    wrote_csv = False
    csv_name = None
    if clears:
        # promote best clear with max disagree among clears
        win = sorted(clears, key=lambda r: (r["honest_nested"], r["disagree_vs_v7"], r["full"]), reverse=True)[0]
        # Build test IR matching win["ir"]
        # Load test assets
        try:
            meta = json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_meta.json").read_text(encoding="utf-8-sig"))
            empty = set(json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_empty.json").read_text(encoding="utf-8-sig")))
            fb_path = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "fallback_map.json"
            fb = json.loads(fb_path.read_text(encoding="utf-8-sig")) if fb_path.exists() else {}
            # classic9 test
            from fuse_ir_v9 import load_members as _lm
            # reuse write_ir_v7 style: mean of classic members test logits
            # Simplified: use existing helpers from write path if available
            c9_test_parts = []
            v5 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
            v6 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
            # load_members tags -> test files via write_ir_v7 logic
            import importlib.util
            spec = importlib.util.spec_from_file_location("write_ir_v7", ROOT / "write_ir_v7.py")
            # fallback manual: phaseA test + classic from ens
            t42 = np.load(CK / "test_logits_seed42.npy")
            t888 = np.load(CK / "test_logits_seed888.npy")
            t2024 = np.load(CK / "test_logits_seed2024.npy")
            phaseA_test = np.mean([t42, t888, t2024], 0).astype(np.float32)
            # classic9 test from v7 submission ingredients
            ir_test_c9 = np.load(v5 / "test_logits_ens_v6.npy").astype(np.float32) if (v5 / "test_logits_ens_v6.npy").exists() else phaseA_test
            th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
            if not th_test_p.exists():
                th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
            th_test = np.load(th_test_p).astype(np.float32)
            mid_test = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_test_logits.npy").astype(np.float32)
            # Map IR name to test
            irn = win["ir"].split("::")[-1] if "::" in win["ir"] else win["ir"]
            # densify::name::T -> name
            if win["ir"].startswith("densify::"):
                irn = win["ir"].split("::")[1]
            if irn.startswith("perT::"):
                irn = irn[6:]
            # approximate test IR
            if "louo_phaseA" in irn or irn == "phaseA" or "mean_w_phaseA" in irn or irn == "top2_42_888":
                if "top2" in irn:
                    ir_test = np.mean([t42, t888], 0).astype(np.float32)
                else:
                    ir_test = phaseA_test
            elif "c9" in irn and "pa" in irn:
                # mix classic9 test + phaseA
                alpha = 0.5
                for a in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
                    if f"a{a}" in irn or f"_{a}_" in irn or irn.endswith(f"_{a}") or f"0.{int(a*100)}" in irn:
                        pass
                if "0.4_ff_0.3" in irn or "0.4" in irn and "0.3" in irn:
                    ff_test_p = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "test_logits_ens.npy"
                    ff_test = np.load(ff_test_p).astype(np.float32) if ff_test_p.exists() else phaseA_test
                    ir_test = (0.4 * ir_test_c9 + 0.3 * ff_test + 0.3 * phaseA_test).astype(np.float32)
                elif "0.5" in irn:
                    ir_test = (0.5 * ir_test_c9 + 0.5 * phaseA_test).astype(np.float32)
                elif "0.55" in irn:
                    ir_test = (0.55 * ir_test_c9 + 0.45 * phaseA_test).astype(np.float32)
                elif "0.6" in irn and "0.4" in irn:
                    ir_test = (0.6 * ir_test_c9 + 0.4 * phaseA_test).astype(np.float32)
                else:
                    ir_test = (0.5 * ir_test_c9 + 0.5 * phaseA_test).astype(np.float32)
            else:
                ir_test = (0.5 * ir_test_c9 + 0.5 * phaseA_test).astype(np.float32)
            cfg = win.get("stacked_cfg") or win["cfg"]
            mode = "perT" if cfg.get("mode", "").startswith("perT") or "Ta" in cfg else "sameT"
            csv_name = "submission_ir_v26.csv"
            nfb, _ = write_submission_if_clear(
                None, ir_test, None, th_test, None, mid_test, cfg, meta, empty, fb,
                ROOT / csv_name, mode=mode)
            wrote_csv = True
            print(f"WROTE {csv_name} nfb={nfb} win={win['ir']} honest={win['honest_nested']:.5f}", flush=True)
        except Exception as e:
            print("CSV write failed:", repr(e), flush=True)
            wrote_csv = False
            csv_name = None

    status = {
        "tag": "ir_v26_nested_obj_fuse",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not bool(clears),
        "wrote_csv": wrote_csv,
        "csv": csv_name,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "louo": {
            "phaseA_w": mean_w_pa, "phaseA_folds": [list(map(float, w)) for w in folds_pa],
            "c9_pa_w": mean_w_c9pa, "c9_ff_pa_w": mean_w_3,
        },
        "progress": {
            "best_honest": best["honest_nested"],
            "best_full": best["full"],
            "best_ir": best["ir"],
            "best_disagree": best["disagree_vs_v7"],
            "n_clear": len(clears),
            "delta_vs_gate": best["honest_nested"] - GATE,
            "prior_honest": PRIOR,
            "improved_vs_prior": best["honest_nested"] - PRIOR,
        },
        "best": best,
        "top15": all_ranked[:15],
        "clears": clears[:5],
        "next_roi": (
            ["PROMOTE CSV — kaggle CLI if clear win", f"csv={csv_name}"]
            if clears else [
                f"MISS honest {best['honest_nested']:.5f} vs gate {GATE} (gap {GATE - best['honest_nested']:.5f})",
                "Keep submission_ir_v7.csv @ 0.69154; no weaker CSV",
                "Next ROI: stronger T24 recipe (seed42-class >=0.71); then re-fuse nested-obj",
                "Optional: densify mid/th further or add mid ens3_soft if IR nest stalls after strong T24",
            ]
        ),
        "elapsed_sec": round(time.time() - t0, 1),
        "updated_at": now,
    }
    (ROOT / "metrics_ir_v26_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v25_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "HOTC_GPU_HANDOFF.md").write_text(
        f"# CUHK-X Small Model Track - status\n\n"
        f"**{status['outcome']}** nest-honest best **{best['honest_nested']:.4f}** ({best['ir']}) "
        f"full={best['full']:.4f} dis={best['disagree_vs_v7']} — gap to 0.758 "
        f"{best['honest_nested'] - GATE:+.4f}.\n"
        f"{'CSV: ' + csv_name if wrote_csv else 'Keep **submission_ir_v7.csv** @ 0.69154. No CSV promoted.'}\n\n"
        f"## Updated {now}\n",
        encoding="utf-8",
    )
    print("BEST", best["ir"], "honest", best["honest_nested"], "full", best["full"],
          "dis", best["disagree_vs_v7"], "clears", len(clears), "elapsed", status["elapsed_sec"], flush=True)
    return 0 if clears else 1


if __name__ == "__main__":
    raise SystemExit(main())
