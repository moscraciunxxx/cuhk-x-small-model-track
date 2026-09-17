"""ir_v17: fuse new Thermal v4 seeds with classic9 BASE IR + MidFuse.
Honest nested sameT preferred (ir_v16 lesson). CSV only if gate clears.
Gate: hold+nested >= ~0.763 AND >=20 disagrees vs ir_v7. No Kaggle submit.
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
        p = (
            cfg["wa"] * softmax_np(a[mask], T)
            + cfg["wb"] * softmax_np(b[mask], T)
            + cfg["wc"] * softmax_np(c[mask], T)
        ).argmax(1)
    return float((p == y[mask]).mean()), p


def preds_full(a, b, c, mask, cfg):
    out = np.full(len(a), -1, dtype=np.int64)
    mode = cfg.get("mode", "sameT")
    if mode == "perT":
        pa = softmax_np(a[mask], cfg["Ta"])
        pb = softmax_np(b[mask], cfg["Tb"])
        pc = softmax_np(c[mask], cfg["Tc"])
        out[mask] = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1)
    elif mode == "geom":
        eps = 1e-8
        T = cfg["T"]
        la = np.log(np.clip(softmax_np(a[mask], T), eps, 1))
        lb = np.log(np.clip(softmax_np(b[mask], T), eps, 1))
        lc = np.log(np.clip(softmax_np(c[mask], T), eps, 1))
        out[mask] = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        T = cfg["T"]
        out[mask] = (
            cfg["wa"] * softmax_np(a[mask], T)
            + cfg["wb"] * softmax_np(b[mask], T)
            + cfg["wc"] * softmax_np(c[mask], T)
        ).argmax(1)
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
    z = np.load(npz_path, allow_pickle=True)
    th_logits = z["logits"].astype(np.float32)
    th_meta_path = ROOT / "cache" / "thermal_yolo" / "train_meta.json"
    th_users = np.load(ROOT / "cache" / "thermal_yolo" / "train_users.npy")
    th_meta = json.load(open(th_meta_path, encoding="utf-8"))
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
    tags = list(z["tags"]) if "tags" in z.files else []
    return out, matched, tags, z


def align_hold_logits_array(hold_logits, th_hold_idx, th_meta, ir_meta, ir_len):
    key_to_logit = {}
    for j, i in enumerate(th_hold_idx):
        key_to_logit[clip_key(th_meta[i])] = hold_logits[j]
    out = np.zeros((ir_len, hold_logits.shape[1]), dtype=np.float32)
    matched = 0
    for i, m in enumerate(ir_meta):
        k = clip_key(m)
        if k in key_to_logit:
            out[i] = key_to_logit[k]
            matched += 1
    return out, matched


def load_ir_pools(members, yt, yu):
    c9 = sorted([m for m in members if m["tag"] != "pool_seed55"], key=lambda d: -d.get("acc_base", d["acc"]))

    def base_of(m):
        return m["base"] if m.get("base") is not None else m["logits"]

    pools = {
        "classic9_base": np.mean([base_of(m) for m in c9], 0).astype(np.float32),
        "classic_top6_base": np.mean([base_of(m) for m in c9[:6]], 0).astype(np.float32),
        "classic9_selTTA": np.mean([m["logits"] for m in c9], 0).astype(np.float32),
    }
    w = np.array([max(m.get("acc_base", m["acc"]), 1e-3) for m in c9], dtype=np.float64)
    w /= w.sum()
    stack = np.stack([base_of(m) for m in c9], 0)
    pools["classic_acc_w_base"] = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)
    return pools, c9


def build_thermal_variants(yt, ir_meta):
    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy"
    th0 = np.load(old).astype(np.float32)
    variants = {"th_v6_v2trio": th0, "th_v6_soft": (th0 / 1.8).astype(np.float32)}

    th_users = np.load(ROOT / "cache" / "thermal_yolo" / "train_users.npy")
    th_meta = json.load(open(ROOT / "cache" / "thermal_yolo" / "train_meta.json", encoding="utf-8"))
    hold_u = set(DEFAULT_HOLD_OUT_USERS)
    th_hold_idx = [i for i, u in enumerate(th_users) if int(u) in hold_u]

    for label, path in [
        ("th_v2_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "holdout_ensemble_logits.npz"),
        ("th_v3_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "holdout_ensemble_logits.npz"),
        ("th_v4_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v4" / "holdout_ensemble_logits.npz"),
    ]:
        if path.exists():
            aligned, matched, tags, z = align_thermal_npz_to_ir(path, ir_meta, len(yt))
            variants[label] = aligned
            print(f"  {label}: matched={matched}/{len(yt)} tags={tags} solo={(aligned.argmax(1)==yt)[aligned.any(1)].mean():.4f}", flush=True)
            # if stack present, also expose best member and soft
            if "stack" in z.files:
                stack = z["stack"].astype(np.float32)
                scores = z["scores"] if "scores" in z.files else None
                for mi in range(stack.shape[0]):
                    al, mt = align_hold_logits_array(stack[mi], th_hold_idx, th_meta, ir_meta, len(yt))
                    tag = f"{label}_m{mi}"
                    if scores is not None and mi < len(scores):
                        tag = f"{label}_seedscore{float(scores[mi]):.3f}".replace(".", "p")
                    variants[f"{label}_m{mi}"] = al
                    print(f"    member{mi}: matched={mt} solo={(al.argmax(1)==yt)[al.any(1)].mean():.4f}", flush=True)

    # Single v4 ckpts if ens missing but seed files exist
    v4 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v4"
    if v4.exists():
        for ck in sorted(v4.glob("pool_seed*.pt")):
            try:
                import torch
                blob = torch.load(ck, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  skip {ck.name}: {e}", flush=True)
                continue
            hl = blob.get("hold_logits")
            if hl is None:
                continue
            hl = np.asarray(hl, dtype=np.float32)
            al, mt = align_hold_logits_array(hl, th_hold_idx, th_meta, ir_meta, len(yt))
            name = f"th_v4_{ck.stem}"
            variants[name] = al
            print(f"  {name}: matched={mt} val={float(blob.get('val_acc', -1)):.4f} solo={(al.argmax(1)==yt)[al.any(1)].mean():.4f}", flush=True)

    # Mix v4 with v2trio (diversity)
    v4_keys = [k for k in variants if k.startswith("th_v4")]
    for vk in v4_keys:
        a, b = th0, variants[vk]
        both = a.any(1) & b.any(1)
        if both.sum() < 400:
            continue
        for wa, wb, tag in [(0.5, 0.5, "50_50"), (0.35, 0.65, "35_65"), (0.65, 0.35, "65_35")]:
            mix = a.copy()
            mix[both] = wa * a[both] + wb * b[both]
            variants[f"mix_v6_{tag}_{vk}"] = mix

    return variants


def honest_score(r):
    nf = r["nested_fixed"] or 0
    nr = r["nested_retune"] if r["nested_retune"] is not None else None
    if r["mode"] == "sameT" and nr is not None:
        honest = min(nf, nr)
    elif r["mode"] == "perT":
        honest = nf - 0.01
    else:
        honest = nf
    return (honest, r["full"], r["disagree_vs_v7"])


def main():
    t0 = time.time()
    print("ir_v17 Thermal-v4 + classic9 BASE MidFuse probe", flush=True)
    members, yt, yu = load_members()
    ir_meta_all = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "train_meta.json", encoding="utf-8"))
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    ir_meta = [ir_meta_all[i] for i in hold_idx]
    assert len(ir_meta) == len(yt)

    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")[hold_idx].astype(np.float32)
    mid_alts = {"mid_ir_v4": mid}
    p11 = ROOT / "cache" / "ir_yolo_11n_v1" / "midfuse_aligned_train_logits.npy"
    if p11.exists():
        m11 = np.load(p11)
        if len(m11) == len(tu):
            mid_alts["mid_ir_11n"] = m11[hold_idx].astype(np.float32)

    pools, c9 = load_ir_pools(members, yt, yu)
    # Prefer classic9 BASE focus; keep top6/accw for completeness
    focus_irs = ["classic9_base", "classic_top6_base", "classic_acc_w_base"]
    print("IR pools:", {k: float((v.argmax(1) == yt).mean()) for k, v in pools.items()}, flush=True)

    print("Building thermal variants (incl v4)...", flush=True)
    th_vars = build_thermal_variants(yt, ir_meta)
    for k, v in th_vars.items():
        nz = int(v.any(1).sum())
        acc = float((v.argmax(1) == yt)[v.any(1)].mean()) if nz else 0.0
        print(f"  {k}: nz={nz} solo={acc:.4f}", flush=True)

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

    # Prioritize classic9_base and v4 thermals; still score baselines
    results = []
    for ir_name in focus_irs:
        if ir_name not in pools:
            continue
        ir = pools[ir_name]
        for th_name, th in th_vars.items():
            for mid_name, md in mid_alts.items():
                mask = th.any(1) & md.any(1)
                if mask.sum() < 400:
                    continue
                b_acc, bcfg = fuse3_sameT(ir, th, md, yt, mask, Ts_fine, ngrid=31)
                nest_fixed = nested_fixed(ir, th, md, yt, yu, mask, bcfg)
                nest_rt = nested_retune(ir, th, md, yt, yu, mask, Ts_med, ngrid=17)
                pred = preds_full(ir, th, md, mask, bcfg)
                disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
                row = {
                    "ir": ir_name,
                    "th": th_name,
                    "mid": mid_name,
                    "mode": "sameT",
                    "full": b_acc,
                    "cfg": bcfg,
                    "nested_fixed": float(nest_fixed["mean"]),
                    "nested_retune": float(nest_rt["mean"]),
                    "disagree_vs_v7": disagree,
                    "mask_n": int(mask.sum()),
                    "th_solo": float((th.argmax(1) == yt)[th.any(1)].mean()),
                }
                results.append(row)
                print(
                    f"{ir_name}|{th_name}|{mid_name} sameT full={b_acc:.4f} "
                    f"nestF={row['nested_fixed']:.4f} nestR={row['nested_retune']:.4f} "
                    f"dis={disagree} th_solo={row['th_solo']:.4f}",
                    flush=True,
                )

    ranked_same = sorted(results, key=honest_score, reverse=True)
    extras = []
    for row in ranked_same[:6]:
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
            "th_solo": row["th_solo"],
        })
        p_acc, pcfg = fuse3_perT(ir, th, md, yt, mask, Ts_per, ngrid=13)
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
            "th_solo": row["th_solo"],
            "note": "perT nested_fixed optimistic (cfg fit on full hold)",
        })
        print(f"EXTRA {row['ir']}|{row['th']} geom={g_acc:.4f}/{nest_g['mean']:.4f} perT={p_acc:.4f}/{nest_p:.4f}", flush=True)

    all_rows = results + extras
    ranked = sorted(all_rows, key=honest_score, reverse=True)
    best = ranked[0]
    honest_best = min(best["nested_fixed"], best["nested_retune"]) if best["nested_retune"] is not None else best["nested_fixed"]
    clears = best["full"] >= GATE and honest_best >= GATE and best["disagree_vs_v7"] >= MIN_DISAGREE

    # best honest sameT specifically
    sameT_rows = [r for r in results if r["mode"] == "sameT"]
    best_same = sorted(sameT_rows, key=honest_score, reverse=True)[0]
    honest_same = min(best_same["nested_fixed"], best_same["nested_retune"])

    wrote_csv = False
    csv_path = None
    th_test_path = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v4" / "test_logits.npy"
    if clears and best["ir"] == "classic9_base" and th_test_path.exists() and (
        best["th"].startswith("th_v4") or best["th"].startswith("mix_v6")
    ):
        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
        th_test = np.load(th_test_path).astype(np.float32)
        # If mix, blend with v2/v3 test
        if best["th"].startswith("mix_v6"):
            th_v2t = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "test_logits.npy").astype(np.float32)
            # parse weights from name mix_v6_50_50_...
            parts = best["th"].split("_")
            # mix_v6_50_50_th_v4_...
            try:
                wa = int(parts[2]) / 100.0
                wb = int(parts[3]) / 100.0
            except Exception:
                wa, wb = 0.5, 0.5
            th_test = (wa * th_v2t + wb * th_test).astype(np.float32)

        def tlog(m):
            return m["test_base"] if m.get("test_base") is not None else m["test_logits"]

        ir_test = np.mean([tlog(m) for m in c9], 0)
        cfg = best["cfg"]
        T = cfg.get("T", 2.5)
        if cfg.get("mode") == "perT":
            probs = (
                cfg["wa"] * softmax_np(ir_test, cfg["Ta"])
                + cfg["wb"] * softmax_np(th_test, cfg["Tb"])
                + cfg["wc"] * softmax_np(mid_test, cfg["Tc"])
            )
        elif cfg.get("mode") == "geom":
            eps = 1e-8
            probs = np.exp(
                cfg["wa"] * np.log(np.clip(softmax_np(ir_test, T), eps, 1))
                + cfg["wb"] * np.log(np.clip(softmax_np(th_test, T), eps, 1))
                + cfg["wc"] * np.log(np.clip(softmax_np(mid_test, T), eps, 1))
            )
        else:
            probs = (
                cfg["wa"] * softmax_np(ir_test, T)
                + cfg["wb"] * softmax_np(th_test, T)
                + cfg["wc"] * softmax_np(mid_test, T)
            )
        preds = probs.argmax(1)
        meta = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "test_meta.json", encoding="utf-8"))
        empty = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "test_empty.json", encoding="utf-8"))
        csv_path = str(ROOT / "submission_ir_v17.csv")
        write_sub(Path(csv_path), meta, preds, empty, 0)
        wrote_csv = True
        print(f"WROTE {csv_path}", flush=True)
    elif clears:
        print("GATE cleared but CSV skipped (need classic9_base + v4 test logits mapping)", flush=True)

    v4_metrics = {}
    mp = ROOT / "metrics_thermal_v4.json"
    if mp.exists():
        v4_metrics = json.loads(mp.read_text(encoding="utf-8"))

    status = {
        "tag": "ir_v17_thermal_v4_mid",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": v7_full, "nested": float(v7_nest["mean"])},
        "thermal_v4": v4_metrics,
        "best": {
            **{k: best[k] for k in ("ir", "th", "mid", "mode", "full", "cfg", "nested_fixed", "nested_retune", "disagree_vs_v7")},
            "honest_nested": honest_best,
            "clears_gate": clears,
        },
        "best_honest_sameT": {
            **{k: best_same[k] for k in ("ir", "th", "mid", "mode", "full", "cfg", "nested_fixed", "nested_retune", "disagree_vs_v7")},
            "honest_nested": honest_same,
            "clears_gate": bool(best_same["full"] >= GATE and honest_same >= GATE and best_same["disagree_vs_v7"] >= MIN_DISAGREE),
        },
        "top10": [
            {
                "ir": r["ir"], "th": r["th"], "mid": r["mid"], "mode": r["mode"],
                "full": r["full"], "nested_fixed": r["nested_fixed"], "nested_retune": r["nested_retune"],
                "disagree_vs_v7": r["disagree_vs_v7"], "cfg": r["cfg"],
                "th_solo": r.get("th_solo"),
            }
            for r in ranked[:10]
        ],
        "wrote_csv": wrote_csv,
        "csv": csv_path,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "delta_vs_gate": {
            "best_full": best["full"] - GATE,
            "honest_nested": honest_best - GATE,
            "honest_sameT": honest_same - GATE,
        },
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "n_combos_scored": len(all_rows),
    }

    if clears:
        status["next_roi"].append("Gate cleared — human may upload submission_ir_v17.csv (no auto Kaggle)")
    else:
        th_solo_best = max((r.get("th_solo") or 0) for r in results) if results else 0
        status["next_roi"].append(
            f"Thermal v4 best solo~{th_solo_best:.4f} vs v2_trio~0.603; fuse honest_sameT={honest_same:.4f} gate={GATE:.4f}"
        )
        if th_solo_best < 0.60 and honest_same <= V7_HOLD + 0.005:
            status["next_roi"].append(
                "Thermal train looks plateaued (solo<<prior / fuse<=v7) — optional longer IR seed ep48+ on ir_yolo_v4"
            )
        else:
            status["next_roi"].append("If only one v4 seed trained, try second seed; else IR longer seed fallback")
        status["next_roi"].append("Keep ir_v7; never promote weaker")

    out = ROOT / "metrics_ir_v17_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(
        f"\nOUTCOME={status['outcome']} clears={clears} best_full={best['full']:.4f} "
        f"honest_nest={honest_best:.4f} sameT_honest={honest_same:.4f} dis={best['disagree_vs_v7']}",
        flush=True,
    )
    print(f"wrote {out} elapsed={status['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
