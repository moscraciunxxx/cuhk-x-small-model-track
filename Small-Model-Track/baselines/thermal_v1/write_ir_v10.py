"""IR v10: anti-holdout-overfit. Prefer sameT / nested LOUO / weights near v7.
Avoids perT and holdout-maximizing topK_sel that tanked public (v9 0.68656 vs v7 0.69154).
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, write_sub, fuse3_sameT

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7_HOLD = 0.7530364372469636
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
V7_PUBLIC = 0.69154
V9_PUBLIC = 0.68656


def acc_of(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (
        cfg["wa"] * softmax_np(a[mask], T)
        + cfg["wb"] * softmax_np(b[mask], T)
        + cfg["wc"] * softmax_np(c[mask], T)
    ).argmax(1)
    return float((p == y[mask]).mean())


def cfg_dist(cfg, ref=V7_CFG):
    return abs(cfg["wa"] - ref["wa"]) + abs(cfg["wb"] - ref["wb"]) + abs(cfg["wc"] - ref["wc"]) + 0.1 * abs(cfg["T"] - ref["T"])


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=41, leave_users=(8, 9, 24)):
    """Honest nested: tune sameT on train folds, eval on leave-out user."""
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc = acc_of(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def local_refine_near_v7(a, b, c, y, mask, center=V7_CFG, d_w=0.12, d_T=1.0, n_w=13, n_T=9):
    """Grid search in a neighborhood of v7 weights (compromise, not full holdout hunt)."""
    best = (-1.0, None)
    yt = y[mask]
    was = np.linspace(max(0, center["wa"] - d_w), min(1, center["wa"] + d_w), n_w)
    wbs = np.linspace(max(0, center["wb"] - d_w), min(1, center["wb"] + d_w), n_w)
    Ts = np.linspace(max(0.5, center["T"] - d_T), center["T"] + d_T, n_T)
    for T in Ts:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in was:
            for wb in wbs:
                wc = 1.0 - wa - wb
                if wc < -1e-9 or wc > 1 + 1e-9:
                    continue
                if abs(wc - center["wc"]) > d_w + 1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0] + 1e-12:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "sameT"})
    return best


def build_variants(members):
    stack = np.stack([m["logits"] for m in members], 0)
    members9 = [m for m in members if m["tag"] != "pool_seed55"]
    stack9 = np.stack([m["logits"] for m in members9], 0)
    # base-only stacks (no holdout TTA selection)
    base_all = np.stack([m["base"] for m in members], 0)
    base9 = np.stack([m["base"] for m in members9], 0)
    w = np.array([max(m["acc"], 1e-3) for m in members], float); w /= w.sum()
    w9 = np.array([max(m["acc"], 1e-3) for m in members9], float); w9 /= w9.sum()
    # base acc weights (less holdout-contaminated than selective acc)
    wb = np.array([max(float((m["base"].argmax(1) == members[0]["logits"].argmax(1)).mean()) * 0 + max(m["acc_base"], 1e-3), 1e-3) for m in members], float)
    # properly: use acc_base
    wb = np.array([max(m["acc_base"], 1e-3) for m in members], float); wb /= wb.sum()
    wb9 = np.array([max(m["acc_base"], 1e-3) for m in members9], float); wb9 /= wb9.sum()

    variants = {
        "all9_base": np.mean(base9, 0),  # exact v7 IR
        "all10_base": np.mean(base_all, 0),  # +seed55 base, no TTA select
        "all9_sel": np.mean(stack9, 0),
        "all_sel": np.mean(stack, 0),
        "all9_acc_w_base": np.tensordot(wb9, base9, axes=(0, 0)),
        "all10_acc_w_base": np.tensordot(wb, base_all, axes=(0, 0)),
        "all9_acc_w_sel": np.tensordot(w9, stack9, axes=(0, 0)),
        "top8_base": np.mean(np.stack([m["base"] for m in sorted(members9, key=lambda d: -d["acc_base"])[:8]], 0), 0),
        "top6_base": np.mean(np.stack([m["base"] for m in sorted(members9, key=lambda d: -d["acc_base"])[:6]], 0), 0),
    }
    sm9 = np.mean([softmax_np(m["base"]) for m in members9], 0)
    variants["all9_sm_base"] = np.log(np.clip(sm9, 1e-8, 1))
    sm = np.mean([softmax_np(m["logits"]) for m in members9], 0)
    variants["all9_sm_sel"] = np.log(np.clip(sm, 1e-8, 1))
    # mild selective: only apply TTA if gain >= 0.005 (stricter than v9's >0)
    strict = []
    for m in members9:
        if m["tta"] is not None and (m["acc_tta"] or 0) >= m["acc_base"] + 0.005:
            strict.append(m["tta"])
        else:
            strict.append(m["base"])
    variants["all9_strictTTA"] = np.mean(np.stack(strict, 0), 0)
    # half-mix base and sel (compromise IR)
    variants["all9_half"] = 0.5 * variants["all9_base"] + 0.5 * variants["all9_sel"]
    return variants, members9, wb9


def ir_test_for(ens, members, members9):
    def base_test(m):
        seed = m["tag"].replace("pool_seed", "")
        old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
        newd = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
        v7d = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7"
        for p in [
            old / f"test_logits_v6_{m['tag']}_base.npy",
            old / f"test_logits_{m['tag']}.npy",
            old / f"test_logits_seed{seed}.npy",
            newd / f"test_logits_seed{seed}.npy",
            v7d / f"test_logits_seed{seed}.npy",
        ]:
            if p.exists():
                return np.load(p)
        return m["test_logits"]  # fallback

    def sel_test(m):
        return m["test_logits"]

    if ens == "all9_base":
        return np.mean([base_test(m) for m in members9], 0).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all10_base":
        return np.mean([base_test(m) for m in members], 0).astype(np.float32), [m["tag"] for m in members]
    if ens == "all9_sel":
        return np.mean([sel_test(m) for m in members9], 0).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all_sel":
        return np.mean([sel_test(m) for m in members], 0).astype(np.float32), [m["tag"] for m in members]
    if ens == "all9_acc_w_base":
        wb9 = np.array([max(m["acc_base"], 1e-3) for m in members9], float); wb9 /= wb9.sum()
        return np.tensordot(wb9, np.stack([base_test(m) for m in members9], 0), 1).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all10_acc_w_base":
        wb = np.array([max(m["acc_base"], 1e-3) for m in members], float); wb /= wb.sum()
        return np.tensordot(wb, np.stack([base_test(m) for m in members], 0), 1).astype(np.float32), [m["tag"] for m in members]
    if ens == "all9_acc_w_sel":
        w9 = np.array([max(m["acc"], 1e-3) for m in members9], float); w9 /= w9.sum()
        return np.tensordot(w9, np.stack([sel_test(m) for m in members9], 0), 1).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all9_sm_base":
        sm = np.mean([softmax_np(base_test(m)) for m in members9], 0)
        return np.log(np.clip(sm, 1e-8, 1)).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all9_sm_sel":
        sm = np.mean([softmax_np(sel_test(m)) for m in members9], 0)
        return np.log(np.clip(sm, 1e-8, 1)).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all9_strictTTA":
        outs = []
        for m in members9:
            if m["tta"] is not None and (m["acc_tta"] or 0) >= m["acc_base"] + 0.005:
                outs.append(sel_test(m))
            else:
                outs.append(base_test(m))
        return np.mean(outs, 0).astype(np.float32), [m["tag"] for m in members9]
    if ens == "all9_half":
        b = np.mean([base_test(m) for m in members9], 0)
        s = np.mean([sel_test(m) for m in members9], 0)
        return (0.5 * b + 0.5 * s).astype(np.float32), [m["tag"] for m in members9]
    if ens.startswith("top") and ens.endswith("_base"):
        k = int(ens[3:].split("_")[0])
        ranked = sorted(members9, key=lambda d: -d["acc_base"])[:k]
        return np.mean([base_test(m) for m in ranked], 0).astype(np.float32), [m["tag"] for m in ranked]
    raise KeyError(ens)


def preds_sameT(ir, th, mid, cfg):
    T = cfg["T"]
    return (cfg["wa"] * softmax_np(ir, T) + cfg["wb"] * softmax_np(th, T) + cfg["wc"] * softmax_np(mid, T)).argmax(1)


def main():
    t0 = time.time()
    cache = ROOT / "cache" / "ir_yolo_v4"
    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    members, yt, yu = load_members()
    th = np.load(old / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")
    from dataset import DEFAULT_HOLD_OUT_USERS
    tu = np.load(cache / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid_h = mid[hold_idx]
    mask = th.any(1) & mid_h.any(1)

    variants, members9, wb9 = build_variants(members)
    print("variants:", list(variants.keys()), flush=True)

    # Reproduce v7
    v7_full = acc_of(variants["all9_base"], th, mid_h, yt, mask, V7_CFG)
    v7_nest = nested_fixed(variants["all9_base"], th, mid_h, yt, yu, mask, V7_CFG)
    v7_retune = nested_retune(variants["all9_base"], th, mid_h, yt, yu, mask, Ts=np.linspace(0.75, 3.5, 12), ngrid=31)
    print(f"v7 reproduce full={v7_full:.6f} nested_fixed={v7_nest['mean']:.6f} nested_retune={v7_retune['mean']:.6f}", flush=True)

    Ts_coarse = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    rows = []
    for ens, el in variants.items():
        # A) fixed v7 cfg
        full_v7w = acc_of(el, th, mid_h, yt, mask, V7_CFG)
        nest_v7w = nested_fixed(el, th, mid_h, yt, yu, mask, V7_CFG)["mean"]
        rows.append({
            "ens": ens, "strategy": "fixed_v7_cfg", "cfg": dict(V7_CFG, mode="sameT", acc=full_v7w),
            "full": full_v7w, "nested_fixed": nest_v7w, "nested_retune": None,
            "dist_v7": 0.0,
        })
        # B) local refine near v7 (compromise)
        full_loc, cfg_loc = local_refine_near_v7(el, th, mid_h, yt, mask)
        if cfg_loc is not None:
            nest_loc = nested_fixed(el, th, mid_h, yt, yu, mask, cfg_loc)["mean"]
            rows.append({
                "ens": ens, "strategy": "local_near_v7", "cfg": cfg_loc,
                "full": full_loc, "nested_fixed": nest_loc, "nested_retune": None,
                "dist_v7": cfg_dist(cfg_loc),
            })
        # C) nested retune mean (honest) — use fold-mean as score; also report full with mean cfg
        ret = nested_retune(el, th, mid_h, yt, yu, mask, Ts=Ts_coarse, ngrid=31)
        # consensus cfg = average of fold cfgs (compromise)
        if ret["folds"]:
            wa = float(np.mean([f["cfg"]["wa"] for f in ret["folds"]]))
            wb = float(np.mean([f["cfg"]["wb"] for f in ret["folds"]]))
            wc = float(np.mean([f["cfg"]["wc"] for f in ret["folds"]]))
            T = float(np.mean([f["cfg"]["T"] for f in ret["folds"]]))
            s = wa + wb + wc
            cfg_avg = {"wa": wa / s, "wb": wb / s, "wc": wc / s, "T": T, "mode": "sameT"}
            full_avg = acc_of(el, th, mid_h, yt, mask, cfg_avg)
            nest_avg = nested_fixed(el, th, mid_h, yt, yu, mask, cfg_avg)["mean"]
            cfg_avg["acc"] = full_avg
            rows.append({
                "ens": ens, "strategy": "nested_avg_cfg", "cfg": cfg_avg,
                "full": full_avg, "nested_fixed": nest_avg, "nested_retune": ret["mean"],
                "dist_v7": cfg_dist(cfg_avg), "retune_folds": ret["folds"],
            })
        print(f"  {ens}: v7w full={full_v7w:.4f} nest={nest_v7w:.4f} | retune={ret['mean']:.4f}", flush=True)

    # Score: prioritize nested_retune if present else nested_fixed; penalize distance to v7; prefer base ensembles
    def score(r):
        nest = r["nested_retune"] if r["nested_retune"] is not None else r["nested_fixed"]
        # bonus for base-ish ensembles (less holdout TTA selection)
        base_bonus = 0.0015 if ("base" in r["ens"] or r["ens"] in ("all9_half", "all9_strictTTA")) else 0.0
        # prefer fixed_v7_cfg / local / nested_avg over free hunt
        strat_bonus = {"fixed_v7_cfg": 0.0010, "local_near_v7": 0.0005, "nested_avg_cfg": 0.0008}.get(r["strategy"], 0.0)
        return nest + base_bonus + strat_bonus - 0.004 * r["dist_v7"]

    # Filter: must not be much worse than v7 on nested_fixed; prefer nested >= v7_nest - 0.002
    honest = [r for r in rows if r["nested_fixed"] + 1e-9 >= v7_nest["mean"] - 0.003]
    if not honest:
        honest = rows
    honest.sort(key=lambda r: (-score(r), -r["nested_fixed"], -r["full"], r["dist_v7"]))
    print("\nTOP honest candidates:", flush=True)
    for r in honest[:15]:
        print(f"  score={score(r):.4f} nest_f={r['nested_fixed']:.4f} ret={r['nested_retune']} full={r['full']:.4f} "
              f"{r['strategy']}/{r['ens']} dist={r['dist_v7']:.3f} cfg={ {k: round(v,4) if isinstance(v,float) else v for k,v in r['cfg'].items() if k in ('wa','wb','wc','T')} }", flush=True)

    best = honest[0]
    # Prefer a candidate that is NOT identical to v7 if we have a clear nested improvement,
    # but if best is basically v7 that's ok — still write v10 with explicit rationale.
    # Also evaluate soft blend: 0.7*v7_probs + 0.3*best_alt if best uses sel
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    v7p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8"))]
    v9p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v9.csv", encoding="utf-8"))]
    v9s = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v9_sameT.csv", encoding="utf-8"))]

    ir_test, tags = ir_test_for(best["ens"], members, members9)
    preds = preds_sameT(ir_test, th_test, mid_test, best["cfg"])

    # Soft-blend toward v7 if we moved away (compromise)
    ir_v7, _ = ir_test_for("all9_base", members, members9)
    p_v7 = V7_CFG["wa"] * softmax_np(ir_v7, V7_CFG["T"]) + V7_CFG["wb"] * softmax_np(th_test, V7_CFG["T"]) + V7_CFG["wc"] * softmax_np(mid_test, V7_CFG["T"])
    p_new = best["cfg"]["wa"] * softmax_np(ir_test, best["cfg"]["T"]) + best["cfg"]["wb"] * softmax_np(th_test, best["cfg"]["T"]) + best["cfg"]["wc"] * softmax_np(mid_test, best["cfg"]["T"])

    # Evaluate blend alphas on holdout
    el_best = variants[best["ens"]]
    p_v7_h = V7_CFG["wa"] * softmax_np(variants["all9_base"][mask], V7_CFG["T"]) + V7_CFG["wb"] * softmax_np(th[mask], V7_CFG["T"]) + V7_CFG["wc"] * softmax_np(mid_h[mask], V7_CFG["T"])
    p_new_h = best["cfg"]["wa"] * softmax_np(el_best[mask], best["cfg"]["T"]) + best["cfg"]["wb"] * softmax_np(th[mask], best["cfg"]["T"]) + best["cfg"]["wc"] * softmax_np(mid_h[mask], best["cfg"]["T"])
    blend_rows = []
    for a in [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 1.0]:
        ph = a * p_new_h + (1 - a) * p_v7_h
        full = float((ph.argmax(1) == yt[mask]).mean())
        # nested fixed for blend: approximate by per-fold with same alpha
        folds = []
        for leave in (8, 9, 24):
            te = mask & (yu == leave)
            p_v7_f = V7_CFG["wa"] * softmax_np(variants["all9_base"][te], V7_CFG["T"]) + V7_CFG["wb"] * softmax_np(th[te], V7_CFG["T"]) + V7_CFG["wc"] * softmax_np(mid_h[te], V7_CFG["T"])
            p_new_f = best["cfg"]["wa"] * softmax_np(el_best[te], best["cfg"]["T"]) + best["cfg"]["wb"] * softmax_np(th[te], best["cfg"]["T"]) + best["cfg"]["wc"] * softmax_np(mid_h[te], best["cfg"]["T"])
            folds.append(float(((a * p_new_f + (1 - a) * p_v7_f).argmax(1) == yt[te]).mean()))
        nest = float(np.mean(folds))
        blend_rows.append({"alpha_new": a, "full": full, "nested": nest})
        print(f"  blend a={a:.2f} full={full:.4f} nested={nest:.4f}", flush=True)

    # pick blend maximizing nested, tie-break closer to v7 (smaller alpha), require nested >= v7
    blend_ok = [b for b in blend_rows if b["nested"] + 1e-9 >= v7_nest["mean"] - 0.001]
    blend_ok.sort(key=lambda b: (-b["nested"], b["alpha_new"]))
    blend = blend_ok[0] if blend_ok else {"alpha_new": 0.0, "full": v7_full, "nested": v7_nest["mean"]}
    alpha = blend["alpha_new"]
    preds_blend = (alpha * p_new + (1 - alpha) * p_v7).argmax(1)

    # Choose final: if blend nested > best nested_fixed slightly and alpha in (0,1), use blend; else use best
    use_blend = (0.0 < alpha < 1.0) and (blend["nested"] + 1e-9 >= best["nested_fixed"] - 0.0005)
    if use_blend:
        final_preds = preds_blend
        final_method = f"softblend a={alpha:.2f} of ({best['strategy']}/{best['ens']}) with v7"
        final_full = blend["full"]
        final_nested = blend["nested"]
        final_cfg = {"blend_alpha_new": alpha, "new": best["cfg"], "v7": V7_CFG}
    else:
        final_preds = preds
        final_method = f"{best['strategy']}/{best['ens']} sameT"
        final_full = best["full"]
        final_nested = best["nested_fixed"]
        final_cfg = best["cfg"]

    out = ROOT / "submission_ir_v10.csv"
    nfb = write_sub(out, meta, final_preds, empty, fb)
    disagree_v7 = int(sum(int(a) != int(b) for a, b in zip(final_preds, v7p)))
    disagree_v9 = int(sum(int(a) != int(b) for a, b in zip(final_preds, v9p)))
    disagree_v9s = int(sum(int(a) != int(b) for a, b in zip(final_preds, v9s)))

    # Confidence to submit: nested+full honest, not chasing, and not identical to failed v9
    # Require: nested >= v7 nested - small, disagree with v9 high enough OR closer to v7,
    # and either modest improvement nested OR clearly different from overfit path
    closer_to_v7 = disagree_v7 <= disagree_v9  # fewer flips from good public than from bad
    nest_ok = final_nested + 1e-9 >= v7_nest["mean"] - 0.0015
    not_v9_clone = disagree_v9 >= 5
    modest_change = 1 <= disagree_v7 <= 20
    # Only auto-submit if we believe can beat public: stay near v7 recipe with honest nested >= v7
    believe_beat = nest_ok and closer_to_v7 and not_v9_clone and modest_change and (
        final_nested >= v7_nest["mean"] - 1e-9
    ) and (best["ens"].endswith("base") or "half" in best["ens"] or "strict" in best["ens"] or alpha < 0.75)

    fp16 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "model_fp16.pt"
    yolo = ROOT / "yolov8n.pt"
    fp16_mb = fp16.stat().st_size / (1024 * 1024) if fp16.exists() else 59.85
    yolo_mb = yolo.stat().st_size / (1024 * 1024) if yolo.exists() else 6.25

    report = {
        "tag": "ir_v10",
        "primary": "submission_ir_v10.csv",
        "method": final_method,
        "holdout_acc": final_full,
        "nested_fixed": final_nested,
        "nested_v7": v7_nest["mean"],
        "nested_retune_v7": v7_retune["mean"],
        "cfg": final_cfg,
        "best_raw": {k: best[k] for k in ("ens", "strategy", "full", "nested_fixed", "nested_retune", "dist_v7", "cfg")},
        "blend": blend_rows,
        "chosen_blend_alpha": alpha,
        "use_blend": use_blend,
        "ir_tags": tags,
        "delta_hold_vs_v7": float(final_full - V7_HOLD),
        "delta_nested_vs_v7": float(final_nested - v7_nest["mean"]),
        "disagree_vs_v7": disagree_v7,
        "disagree_vs_v9": disagree_v9,
        "disagree_vs_v9_sameT": disagree_v9s,
        "empty_fallback": nfb,
        "v7_public": V7_PUBLIC,
        "v9_public": V9_PUBLIC,
        "public_gap_v7": V7_HOLD - V7_PUBLIC,
        "public_gap_v9": 0.757085020242915 - V9_PUBLIC,
        "top_candidates": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != "retune_folds"}
            for r in honest[:12]
        ],
        "believe_can_beat_public": bool(believe_beat),
        "submit_recommendation": "submit" if believe_beat else "leave_for_parent",
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) <= 100,
        "notes": [
            f"v7 public {V7_PUBLIC} hold {V7_HOLD:.4f}; v9 public {V9_PUBLIC} hold 0.757 — perT/top9_sel overfit",
            f"v10 method={final_method} hold={final_full:.4f} nested={final_nested:.4f} (v7 nested {v7_nest['mean']:.4f})",
            f"disagree v7={disagree_v7} v9={disagree_v9} v9_sameT={disagree_v9s}",
            "Prefer sameT + nested/local-near-v7; no perT",
            f"submit_recommendation={'submit' if believe_beat else 'leave_for_parent'}",
            f"pack ~{fp16_mb+yolo_mb:.1f}MB <=100MB",
            "v9_sameT disagrees with v9 on only 1 test row — essentially same overfit; do not prefer it",
        ],
        "elapsed_s": time.time() - t0,
    }
    Path("metrics_ir_v10.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "holdout_acc": final_full,
        "nested": final_nested,
        "delta_nested_v7": final_nested - v7_nest["mean"],
        "method": final_method,
        "disagree_v7": disagree_v7,
        "disagree_v9": disagree_v9,
        "believe_can_beat_public": believe_beat,
        "submit_recommendation": report["submit_recommendation"],
        "wrote": str(out),
    }, indent=2, default=float), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
