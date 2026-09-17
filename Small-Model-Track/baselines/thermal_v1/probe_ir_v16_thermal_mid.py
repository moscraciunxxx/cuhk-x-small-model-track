"""ir_v16 CPU probe: Thermal / MidFuse side of late fuse with fixed IR pools.
Stay CPU-only. Write metrics_ir_v16_status.json. CSV only if gate clears.
Gate: hold+nested >= ~0.763 AND >=20 disagrees vs ir_v7 preds.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import (
    load_members, softmax_np, nested_fixed, fuse3_sameT, fuse3_geom, fuse3_perT,
    write_sub,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7_HOLD = 0.7530364372469636
GATE = V7_HOLD + 0.01
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}


def apply_cfg(a, b, c, y, mask, cfg):
    mode = cfg.get("mode", "sameT")
    if mode == "perT":
        pa = softmax_np(a[mask], cfg["Ta"])
        pb = softmax_np(b[mask], cfg["Tb"])
        pc = softmax_np(c[mask], cfg["Tc"])
        p = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1)
    elif mode == "geom":
        eps = 1e-8
        T = cfg["T"]
        la = np.log(np.clip(softmax_np(a[mask], T), eps, 1))
        lb = np.log(np.clip(softmax_np(b[mask], T), eps, 1))
        lc = np.log(np.clip(softmax_np(c[mask], T), eps, 1))
        p = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        T = cfg["T"]
        p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
             + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean()), p


def preds_full(a, b, c, mask, cfg):
    """Return length-len(a) pred array; unmasked rows = -1."""
    out = np.full(len(a), -1, dtype=np.int64)
    _, p = apply_cfg(a, b, c, np.zeros(len(a), dtype=np.int64), mask, cfg)
    # recompute with real path
    mode = cfg.get("mode", "sameT")
    if mode == "perT":
        pa = softmax_np(a[mask], cfg["Ta"]); pb = softmax_np(b[mask], cfg["Tb"]); pc = softmax_np(c[mask], cfg["Tc"])
        out[mask] = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1)
    elif mode == "geom":
        eps = 1e-8; T = cfg["T"]
        la = np.log(np.clip(softmax_np(a[mask], T), eps, 1))
        lb = np.log(np.clip(softmax_np(b[mask], T), eps, 1))
        lc = np.log(np.clip(softmax_np(c[mask], T), eps, 1))
        out[mask] = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        T = cfg["T"]
        out[mask] = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
                     + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return out


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=21, leave_users=(8, 9, 24)):
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc, _ = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": acc, "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def clip_key(m):
    return (int(m["user_id"]), int(m["label"]), str(m.get("trial", "")), str(m.get("action_name", "")))


def align_thermal_npz_to_ir(npz_path, ir_meta, ir_len):
    """Map thermal-native hold ensemble (504) onto IR hold order (505) via clip keys."""
    import json
    z = np.load(npz_path, allow_pickle=True)
    th_logits = z["logits"].astype(np.float32)
    th_meta_path = ROOT / "cache" / "thermal_yolo" / "train_meta.json"
    th_users = np.load(ROOT / "cache" / "thermal_yolo" / "train_users.npy")
    th_meta = json.load(open(th_meta_path))
    hold_u = set(DEFAULT_HOLD_OUT_USERS)
    th_hold_idx = [i for i, u in enumerate(th_users) if int(u) in hold_u]
    assert len(th_hold_idx) == len(th_logits), (len(th_hold_idx), len(th_logits))
    key_to_logit = {}
    for j, i in enumerate(th_hold_idx):
        key_to_logit[clip_key(th_meta[i])] = th_logits[j]
    out = np.zeros((ir_len, th_logits.shape[1]), dtype=np.float32)
    matched = 0
    for i, m in enumerate(ir_meta):
        k = clip_key(m)
        if k in key_to_logit:
            out[i] = key_to_logit[k]
            matched += 1
    return out, matched, list(z["tags"]) if "tags" in z.files else [], z


def load_ir_pools(members, yt, yu):
    # ir_v7 used BASE logit-mean (selective-TTA logits actually hurt triple fuse).
    c9 = sorted([m for m in members if m["tag"] != "pool_seed55"], key=lambda d: -d.get("acc_base", d["acc"]))
    def base_of(m):
        return m["base"] if m.get("base") is not None else m["logits"]
    pools = {
        "classic9_base": np.mean([base_of(m) for m in c9], 0).astype(np.float32),
        "classic_top6_base": np.mean([base_of(m) for m in c9[:6]], 0).astype(np.float32),
        "classic9_selTTA": np.mean([m["logits"] for m in c9], 0).astype(np.float32),
        "classic_acc_w_base": None,
    }
    w = np.array([max(m.get("acc_base", m["acc"]), 1e-3) for m in c9], dtype=np.float64); w /= w.sum()
    stack = np.stack([base_of(m) for m in c9], 0)
    pools["classic_acc_w_base"] = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)

    # v15 MV members on disk
    mv_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v15_tta"
    mv = []
    if mv_dir.exists():
        for p in sorted(mv_dir.glob("hold_mv_*.npy")):
            lg = np.load(p).astype(np.float32)
            if lg.shape[0] != len(yt):
                continue
            acc = float((lg.argmax(1) == yt).mean())
            mv.append((p.stem.replace("hold_mv_", ""), lg, acc))
        mv = sorted(mv, key=lambda t: -t[2])
        if mv:
            pools["mv_top6"] = np.mean([x[1] for x in mv[:6]], 0).astype(np.float32)
            pools["mv_top8"] = np.mean([x[1] for x in mv[:8]], 0).astype(np.float32)
            pools["mv_all"] = np.mean([x[1] for x in mv], 0).astype(np.float32)
    # v15 diversity seeds
    v15 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v15"
    extra = []
    for name in ["hold_mv_seed2048.npy", "hold_mv_seed555.npy"]:
        p = v15 / name
        if p.exists():
            lg = np.load(p).astype(np.float32)
            if lg.shape[0] == len(yt):
                extra.append(lg)
    if extra:
        pools["classic9_plus_v15"] = np.mean([pools["classic9_base"]] + extra, 0).astype(np.float32)
        if "mv_top8" in pools:
            pools["mv8_plus_v15"] = np.mean([pools["mv_top8"]] + extra, 0).astype(np.float32)
    return pools, c9, mv


def build_thermal_variants(yt, yu, ir_meta):
    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy"
    th0 = np.load(old).astype(np.float32)
    variants = {"th_v6_v2trio": th0}

    for label, path in [
        ("th_v2_ens_aligned", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "holdout_ensemble_logits.npz"),
        ("th_v3_ens_aligned", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "holdout_ensemble_logits.npz"),
    ]:
        if path.exists():
            aligned, matched, tags, z = align_thermal_npz_to_ir(path, ir_meta, len(yt))
            variants[label] = aligned
            print(f"  {label}: matched={matched}/{len(yt)} tags={tags}", flush=True)
            # also sharpen / temp variants of aligned
            variants[label + "_T0.75"] = (aligned / 0.75).astype(np.float32)
            variants[label + "_T1.5"] = (aligned / 1.5).astype(np.float32)

    # Mix v6 with v3-aligned where both nonzero
    if "th_v3_ens_aligned" in variants:
        a, b = th0, variants["th_v3_ens_aligned"]
        both = a.any(1) & b.any(1)
        mix = a.copy()
        mix[both] = 0.5 * a[both] + 0.5 * b[both]
        variants["th_v6_v3_mean"] = mix
        # prefer stronger-looking v3 on pool side: 0.35 v6 + 0.65 v3
        mix2 = a.copy()
        mix2[both] = 0.35 * a[both] + 0.65 * b[both]
        variants["th_v6_35_v3_65"] = mix2
        mix3 = a.copy()
        mix3[both] = 0.65 * a[both] + 0.35 * b[both]
        variants["th_v6_65_v3_35"] = mix3

    # Softmax-mean of v6 and sharpened
    variants["th_v6_sharp"] = (th0 / 0.7).astype(np.float32)
    variants["th_v6_soft"] = (th0 / 1.8).astype(np.float32)
    return variants


def main():
    t0 = time.time()
    print("ir_v16 Thermal/Mid CPU probe — no GPU", flush=True)
    members, yt, yu = load_members()
    import json
    ir_meta_all = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "train_meta.json"))
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    ir_meta = [ir_meta_all[i] for i in hold_idx]
    assert len(ir_meta) == len(yt)

    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")[hold_idx].astype(np.float32)
    # alt mid from 11n cache (same alignment length?)
    mid_alts = {"mid_ir_v4": mid}
    p11 = ROOT / "cache" / "ir_yolo_11n_v1" / "midfuse_aligned_train_logits.npy"
    if p11.exists():
        m11 = np.load(p11)
        if len(m11) == len(tu):
            mid_alts["mid_ir_11n"] = m11[hold_idx].astype(np.float32)

    pools, c9, mv = load_ir_pools(members, yt, yu)
    print("IR pools:", {k: float((v.argmax(1) == yt).mean()) for k, v in pools.items()}, flush=True)
    if mv:
        print("MV top:", [(t, round(a, 4)) for t, _, a in mv[:8]], flush=True)

    print("Building thermal variants...", flush=True)
    th_vars = build_thermal_variants(yt, yu, ir_meta)
    for k, v in th_vars.items():
        nz = int(v.any(1).sum())
        acc = float((v.argmax(1) == yt)[v.any(1)].mean()) if nz else 0.0
        print(f"  {k}: nz={nz} solo_acc_on_nz={acc:.4f}", flush=True)

    # Reproduce v7 baseline
    ir0 = pools["classic9_base"]
    th0 = th_vars["th_v6_v2trio"]
    mask0 = th0.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(ir0, th0, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(ir0, th0, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(ir0, th0, mid, mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f} mask={int(mask0.sum())}", flush=True)

    Ts_fine = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    Ts_med = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    Ts_per = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = []
    # Focused grid: each IR pool x thermal var x mid alt
    # Use ngrid=31 for speed/quality balance
    for ir_name, ir in pools.items():
        for th_name, th in th_vars.items():
            for mid_name, md in mid_alts.items():
                mask = th.any(1) & md.any(1) & np.ones(len(yt), dtype=bool)
                # skip degenerate
                if mask.sum() < 400:
                    continue
                # sameT
                b_acc, bcfg = fuse3_sameT(ir, th, md, yt, mask, Ts_fine, ngrid=31)
                nest_fixed = nested_fixed(ir, th, md, yt, yu, mask, bcfg)
                nest_rt = nested_retune(ir, th, md, yt, yu, mask, Ts_med, ngrid=17)
                pred = preds_full(ir, th, md, mask, bcfg)
                disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
                row = {
                    "ir": ir_name, "th": th_name, "mid": mid_name,
                    "mode": "sameT", "full": b_acc, "cfg": bcfg,
                    "nested_fixed": float(nest_fixed["mean"]),
                    "nested_retune": float(nest_rt["mean"]),
                    "disagree_vs_v7": disagree,
                    "mask_n": int(mask.sum()),
                    "ens_ir_acc": float((ir.argmax(1) == yt).mean()),
                }
                results.append(row)
                print(
                    f"{ir_name}|{th_name}|{mid_name} sameT full={b_acc:.4f} "
                    f"nestF={row['nested_fixed']:.4f} nestR={row['nested_retune']:.4f} "
                    f"dis={disagree} cfg={bcfg}",
                    flush=True,
                )

    # Extra: geom + perT on best few by nested_fixed
    results_sorted = sorted(results, key=lambda r: (r["nested_fixed"], r["full"]), reverse=True)
    extras = []
    for row in results_sorted[:8]:
        ir, th, md = pools[row["ir"]], th_vars[row["th"]], mid_alts[row["mid"]]
        mask = th.any(1) & md.any(1)
        g_acc, gcfg = fuse3_geom(ir, th, md, yt, mask, Ts_med, ngrid=25)
        nest_g = nested_fixed(ir, th, md, yt, yu, mask, gcfg)
        pred = preds_full(ir, th, md, mask, gcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        extras.append({
            "ir": row["ir"], "th": row["th"], "mid": row["mid"], "mode": "geom",
            "full": g_acc, "cfg": gcfg, "nested_fixed": float(nest_g["mean"]),
            "nested_retune": None, "disagree_vs_v7": disagree, "mask_n": int(mask.sum()),
            "ens_ir_acc": row["ens_ir_acc"],
        })
        print(f"EXTRA geom {row['ir']}|{row['th']} full={g_acc:.4f} nest={nest_g['mean']:.4f} dis={disagree}", flush=True)

        p_acc, pcfg = fuse3_perT(ir, th, md, yt, mask, Ts_per, ngrid=13)
        # nested_fixed for perT via apply
        nest_p_folds = []
        for leave in (8, 9, 24):
            te = mask & (yu == leave)
            if te.sum() < 5:
                continue
            te_acc, _ = apply_cfg(ir, th, md, yt, te, pcfg)
            nest_p_folds.append(te_acc)
        nest_p = float(np.mean(nest_p_folds)) if nest_p_folds else 0.0
        pred = preds_full(ir, th, md, mask, pcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        extras.append({
            "ir": row["ir"], "th": row["th"], "mid": row["mid"], "mode": "perT",
            "full": p_acc, "cfg": pcfg, "nested_fixed": nest_p,
            "nested_retune": None, "disagree_vs_v7": disagree, "mask_n": int(mask.sum()),
            "ens_ir_acc": row["ens_ir_acc"],
            "note": "nested_fixed here is fixed-cfg LOUO (cfg fit on full hold — optimistic)",
        })
        print(f"EXTRA perT {row['ir']}|{row['th']} full={p_acc:.4f} nestFcfg={nest_p:.4f} dis={disagree}", flush=True)

    all_rows = results + extras
    # Rank by honest nested: prefer nested_retune if present else nested_fixed; for sameT prefer min(nest_fixed, nest_retune)
    def score(r):
        nf = r["nested_fixed"] or 0
        nr = r["nested_retune"] if r["nested_retune"] is not None else None
        # perT nested_fixed uses full-hold-fit cfg — optimistic; demote unless retune exists
        if r["mode"] == "sameT" and nr is not None:
            honest = min(nf, nr)
        elif r["mode"] == "perT":
            honest = nf - 0.01  # demote optimistic full-fit LOUO
        else:
            honest = nf
        return (honest, r["full"], r["disagree_vs_v7"])

    ranked = sorted(all_rows, key=score, reverse=True)
    best = ranked[0]
    honest_best = min(best["nested_fixed"], best["nested_retune"]) if best["nested_retune"] is not None else best["nested_fixed"]
    clears = (
        best["full"] >= GATE
        and honest_best >= GATE
        and best["disagree_vs_v7"] >= MIN_DISAGREE
    )

    wrote_csv = False
    csv_path = None
    if clears:
        # rebuild preds and write
        ir, th, md = pools[best["ir"]], th_vars[best["th"]], mid_alts[best["mid"]]
        mask = th.any(1) & md.any(1)
        # Need test logits — only write if we can compose test IR/Thermal/Mid like v7
        # For safety: only auto-write when thermal is th_v6_v2trio (known test logits) and ir is classic9
        if best["th"] in ("th_v6_v2trio", "th_v6_sharp", "th_v6_soft") and best["ir"] in ("classic9_base", "classic_top6_base", "classic_acc_w_base") and best["mid"] == "mid_ir_v4":
            # reuse write path from fuse_ir_v14
            mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
            th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
            th_test = np.load(th_p)
            # IR test from members
            if best["ir"] in ("classic9_base", "classic9_selTTA"):
                # test_logits already selective; for base prefer test_base if present
                def tlog(m):
                    return m["test_base"] if m.get("test_base") is not None else m["test_logits"]
                ir_test = np.mean([tlog(m) for m in c9], 0)
            elif best["ir"] == "classic_top6_base":
                def tlog(m):
                    return m["test_base"] if m.get("test_base") is not None else m["test_logits"]
                ir_test = np.mean([tlog(m) for m in c9[:6]], 0)
            else:
                w = np.array([max(m.get("acc_base", m["acc"]), 1e-3) for m in c9], dtype=np.float64); w /= w.sum()
                def tlog(m):
                    return m["test_base"] if m.get("test_base") is not None else m["test_logits"]
                ir_test = np.tensordot(w, np.stack([tlog(m) for m in c9], 0), axes=(0, 0))
            cfg = best["cfg"]
            if cfg.get("mode") == "perT":
                probs = (cfg["wa"] * softmax_np(ir_test, cfg["Ta"]) + cfg["wb"] * softmax_np(th_test, cfg["Tb"])
                         + cfg["wc"] * softmax_np(mid_test, cfg["Tc"]))
            elif cfg.get("mode") == "geom":
                eps = 1e-8; T = cfg["T"]
                probs = np.exp(cfg["wa"] * np.log(np.clip(softmax_np(ir_test, T), eps, 1))
                               + cfg["wb"] * np.log(np.clip(softmax_np(th_test, T), eps, 1))
                               + cfg["wc"] * np.log(np.clip(softmax_np(mid_test, T), eps, 1)))
            else:
                T = cfg["T"]
                probs = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T)
                         + cfg["wc"] * softmax_np(mid_test, T))
            preds = probs.argmax(1)
            meta = json.load(open(TRACK / "test.csv" if False else ROOT / "cache" / "ir_yolo_v4" / "test_meta.json"))
            empty = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "test_empty.json"))
            # fallback label 0
            csv_path = str(ROOT / "submission_ir_v16.csv")
            write_sub(Path(csv_path), meta, preds, empty, 0)
            wrote_csv = True
            print(f"WROTE {csv_path}", flush=True)
        else:
            print("GATE cleared but CSV skipped (thermal/IR combo lacks verified test logits mapping)", flush=True)

    status = {
        "tag": "ir_v16_thermal_mid_cpu",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": v7_full, "nested": float(v7_nest["mean"])},
        "best": {
            **{k: best[k] for k in ("ir", "th", "mid", "mode", "full", "cfg", "nested_fixed", "nested_retune", "disagree_vs_v7")},
            "honest_nested": honest_best,
            "clears_gate": clears,
        },
        "top10": [
            {
                "ir": r["ir"], "th": r["th"], "mid": r["mid"], "mode": r["mode"],
                "full": r["full"], "nested_fixed": r["nested_fixed"], "nested_retune": r["nested_retune"],
                "disagree_vs_v7": r["disagree_vs_v7"], "cfg": r["cfg"],
            }
            for r in ranked[:10]
        ],
        "inventory": {
            "thermal_ckpts": {
                "v1": "checkpoints/thermal_yolo_r2p1d18",
                "v2_trio": "checkpoints/thermal_yolo_r2p1d18_v2 (v1+seed123+seed7) — used by ir_v7 as hold_thermal_v6",
                "v3": "checkpoints/thermal_yolo_r2p1d18_v3 (pool seeds + alldata test)",
            },
            "thermal_solo_hold_approx": {
                "v2_trio_ens": 0.6032,
                "v3_best_pool_seed2024": 0.5813,
                "midfuse_blend_v2": 0.6640,
                "midfuse_blend_v3_v2trio": 0.6741,
            },
            "ir_pools": {k: float((v.argmax(1) == yt).mean()) for k, v in pools.items()},
            "midfuse_sources": list(mid_alts.keys()),
            "gpu_policy": "CPU-only; LMT holds 3060 for sequence-iterative graft",
        },
        "wrote_csv": wrote_csv,
        "csv": csv_path,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "n_combos_scored": len(all_rows),
    }

    # next_roi heuristics
    best_th_names = {r["th"] for r in ranked[:5]}
    if clears:
        status["next_roi"].append("Gate cleared — consider human Kaggle submit of submission_ir_v16.csv")
    else:
        status["next_roi"].append("Thermal/Mid reweight + existing ens variants did not clear +0.01 nested gate")
        if all(t.startswith("th_v6") for t in best_th_names):
            status["next_roi"].append("Aligned v3 thermal ens did not beat v2_trio in top ranks — train 1-2 stronger Thermal seeds when GPU free (YOLO crop cache ready)")
        else:
            status["next_roi"].append("Some non-v6 thermal variants ranked high — if near gate, CPU-infer missing test logits then retry CSV")
        # check if wb/wc want more thermal
        cfg = best["cfg"]
        if cfg.get("wb", 0) >= 0.4 or cfg.get("wc", 0) >= 0.15:
            status["next_roi"].append("Best cfg leans more Thermal/Mid than v7 — thermal quality is the bottleneck, not IR weight headroom")
        else:
            status["next_roi"].append("Best cfg still IR-heavy like v7 — IR ens may still be limiting; longer IR seed (ep48+) is alternate ROI when GPU free")
        status["next_roi"].append("Do not fight LMT for GPU; wait for nvidia-smi free / parent ping")

    out = ROOT / "metrics_ir_v16_status.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(f"\nOUTCOME={status['outcome']} clears={clears} best_full={best['full']:.4f} honest_nest={honest_best:.4f} dis={best['disagree_vs_v7']}", flush=True)
    print(f"wrote {out} elapsed={status['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()

