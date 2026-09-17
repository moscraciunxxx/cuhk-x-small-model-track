"""IR v10b: distinct from v7, sameT, compromise near-v7. Soft multi-cfg blend for public robustness."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, write_sub
from write_ir_v10 import (
    V7_CFG, V7_HOLD, V7_PUBLIC, V9_PUBLIC, acc_of, cfg_dist, build_variants,
    ir_test_for, local_refine_near_v7, preds_sameT,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")


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
    variants, members9, _ = build_variants(members)

    v7_nest = nested_fixed(variants["all9_base"], th, mid_h, yt, yu, mask, V7_CFG)["mean"]
    print(f"v7 nested={v7_nest:.6f}", flush=True)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")

    # Candidate pool: sameT only, prefer base/half/strict, weights near v7
    pool = []
    for ens in ["all9_base", "all9_half", "all9_strictTTA", "all9_acc_w_base", "all10_base", "all9_sel"]:
        el = variants[ens]
        # fixed v7
        pool.append({"ens": ens, "cfg": dict(V7_CFG), "label": f"{ens}@v7"})
        # local near v7
        full, cfg = local_refine_near_v7(el, th, mid_h, yt, mask)
        if cfg is not None:
            pool.append({"ens": ens, "cfg": cfg, "label": f"{ens}@local"})

    # Dedup by (ens, rounded cfg)
    seen = set()
    cand = []
    for p in pool:
        key = (p["ens"], round(p["cfg"]["wa"], 4), round(p["cfg"]["wb"], 4), round(p["cfg"]["wc"], 4), round(p["cfg"]["T"], 4))
        if key in seen:
            continue
        seen.add(key)
        el = variants[p["ens"]]
        full = acc_of(el, th, mid_h, yt, mask, p["cfg"])
        nest = nested_fixed(el, th, mid_h, yt, yu, mask, p["cfg"])["mean"]
        dist = cfg_dist(p["cfg"])
        p.update({"full": full, "nested": nest, "dist": dist})
        cand.append(p)
        print(f"  {p['label']}: full={full:.4f} nest={nest:.4f} dist={dist:.3f} cfg={p['cfg']}", flush=True)

    # Keep only those within 0.003 nested of v7 and dist<=0.25
    keep = [c for c in cand if c["nested"] + 1e-9 >= v7_nest - 0.003 and c["dist"] <= 0.25]
    keep.sort(key=lambda c: (-c["nested"], c["dist"], -c["full"]))
    print(f"\nKept {len(keep)} near-v7 candidates", flush=True)

    # Build holdout & test prob matrices for soft vote
    def probs_hold(c):
        el = variants[c["ens"]]
        T = c["cfg"]["T"]
        return c["cfg"]["wa"] * softmax_np(el[mask], T) + c["cfg"]["wb"] * softmax_np(th[mask], T) + c["cfg"]["wc"] * softmax_np(mid_h[mask], T)

    def probs_test(c):
        ir, _ = ir_test_for(c["ens"], members, members9)
        T = c["cfg"]["T"]
        return c["cfg"]["wa"] * softmax_np(ir, T) + c["cfg"]["wb"] * softmax_np(th_test, T) + c["cfg"]["wc"] * softmax_np(mid_test, T)

    # Soft-average top-K by nested among kept; force include v7
    v7_c = next(c for c in cand if c["ens"] == "all9_base" and abs(c["cfg"]["wa"] - 0.56) < 1e-9 and abs(c["cfg"]["T"] - 2.5) < 1e-9)
    # Rank: prefer half/strict/base over sel; nest; low dist
    def rank_key(c):
        pref = 0.0
        if "base" in c["ens"]:
            pref += 0.002
        if c["ens"] in ("all9_half", "all9_strictTTA"):
            pref += 0.0015
        if "sel" in c["ens"] and c["ens"] != "all9_half":
            pref -= 0.001
        return -(c["nested"] + pref - 0.003 * c["dist"])

    keep_sorted = sorted(keep, key=rank_key)
    # Build several blend recipes
    recipes = []
    # R1: equal soft mean of top3 distinct ensembles (prefer different ens)
    picked = []
    used_ens = set()
    for c in keep_sorted:
        if c["ens"] in used_ens and c["ens"] != "all9_base":
            continue
        picked.append(c)
        used_ens.add(c["ens"])
        if len(picked) >= 3:
            break
    if v7_c not in picked and not any(c["ens"] == "all9_base" and abs(c["cfg"]["T"] - 2.5) < 1e-9 for c in picked):
        picked = [v7_c] + picked[:2]
    recipes.append(("top3_diverse", picked))

    # R2: v7 + all9_half@best + all9_acc_w_base@v7
    half = [c for c in keep_sorted if c["ens"] == "all9_half"]
    accw = [c for c in keep_sorted if c["ens"] == "all9_acc_w_base"]
    r2 = [v7_c]
    if half:
        r2.append(half[0])
    if accw:
        r2.append(accw[0])
    recipes.append(("v7_half_accw", r2))

    # R3: 0.6*v7 + 0.4*best_non_v7
    non_v7 = [c for c in keep_sorted if not (c["ens"] == "all9_base" and abs(c["cfg"]["wa"] - 0.56) < 1e-9)]
    recipes.append(("v7_60_best40", [v7_c, non_v7[0]] if non_v7 else [v7_c]))

    # R4: all9_half local alone if nest ok
    if half:
        recipes.append(("half_alone", [half[0]]))

    # R5: strictTTA @ v7
    strict = [c for c in keep_sorted if c["ens"] == "all9_strictTTA"]
    if strict:
        recipes.append(("strict_v7w", [strict[0]]))

    def eval_recipe(name, members_c, weights=None):
        if weights is None:
            weights = [1.0 / len(members_c)] * len(members_c)
        # normalize
        s = sum(weights)
        weights = [w / s for w in weights]
        ph = sum(w * probs_hold(c) for w, c in zip(weights, members_c))
        full = float((ph.argmax(1) == yt[mask]).mean())
        folds = []
        for leave in (8, 9, 24):
            te = mask & (yu == leave)
            # rebuild fold probs
            pf = None
            for w, c in zip(weights, members_c):
                el = variants[c["ens"]]
                T = c["cfg"]["T"]
                p = c["cfg"]["wa"] * softmax_np(el[te], T) + c["cfg"]["wb"] * softmax_np(th[te], T) + c["cfg"]["wc"] * softmax_np(mid_h[te], T)
                pf = w * p if pf is None else pf + w * p
            folds.append(float((pf.argmax(1) == yt[te]).mean()))
        nest = float(np.mean(folds))
        return {"name": name, "full": full, "nested": nest, "members": [{"ens": c["ens"], "cfg": c["cfg"], "w": w} for c, w in zip(members_c, weights)],
                "labels": [c["label"] for c in members_c]}

    results = []
    for name, mems in recipes:
        if name == "v7_60_best40" and len(mems) == 2:
            r = eval_recipe(name, mems, [0.6, 0.4])
        else:
            r = eval_recipe(name, mems)
        results.append(r)
        print(f"RECIPE {name}: full={r['full']:.4f} nested={r['nested']:.4f} members={r['labels']}", flush=True)

    # Choose: max nested among those with nested >= v7_nest - 0.0015; prefer diversity (not pure v7) if nest within 0.0005
    results.sort(key=lambda r: (-r["nested"], -r["full"]))
    # Prefer recipes that are not pure all9_base@v7 if nest almost as good
    def recipe_score(r):
        pure_v7 = len(r["members"]) == 1 and r["members"][0]["ens"] == "all9_base" and abs(r["members"][0]["cfg"]["wa"] - 0.56) < 1e-9
        diversity = 0.0 if pure_v7 else 0.0004
        # prefer including half/strict
        labels = ",".join(r["labels"])
        if "half" in labels or "strict" in labels:
            diversity += 0.0003
        return r["nested"] + diversity

    ok = [r for r in results if r["nested"] + 1e-9 >= v7_nest - 0.0015]
    ok.sort(key=lambda r: (-recipe_score(r), -r["full"]))
    best = ok[0]
    print(f"\nCHOSEN {best['name']} full={best['full']:.6f} nested={best['nested']:.6f}", flush=True)

    # Test preds
    weights = [m["w"] for m in best["members"]]
    # map back to cand objects
    mem_objs = []
    for m in best["members"]:
        match = next(c for c in cand if c["ens"] == m["ens"] and abs(c["cfg"]["wa"] - m["cfg"]["wa"]) < 1e-9 and abs(c["cfg"]["T"] - m["cfg"]["T"]) < 1e-9)
        mem_objs.append(match)
    pt = sum(w * probs_test(c) for w, c in zip(weights, mem_objs))
    preds = pt.argmax(1)

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    from fuse_ir_v9 import write_sub
    out = ROOT / "submission_ir_v10.csv"
    nfb = write_sub(out, meta, preds, empty, fb)

    v7p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8"))]
    v9p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v9.csv", encoding="utf-8"))]
    v9s = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v9_sameT.csv", encoding="utf-8"))]
    d7 = int(sum(int(a) != int(b) for a, b in zip(preds, v7p)))
    d9 = int(sum(int(a) != int(b) for a, b in zip(preds, v9p)))
    d9s = int(sum(int(a) != int(b) for a, b in zip(preds, v9s)))

    # Also analyze v9_sameT honesty for submit note
    # v9_sameT: top9_sel sameT 0.52/0.36/0.12@2.25 — hold 0.755 nested 0.754 but ~identical to overfit v9 on test
    sameT_vs_v9 = int(sum(a != b for a, b in zip(v9p, v9s)))

    nest_ok = best["nested"] + 1e-9 >= v7_nest - 0.001
    distinct = d7 >= 2
    closer_v7 = d7 < d9
    not_v9 = d9 >= 5
    # Submit only if distinct from v7, honest nested, closer to v7 than failed v9, and uses compromise (half/blend)
    believe = nest_ok and distinct and closer_v7 and not_v9 and ("half" in best["name"] or "top3" in best["name"] or "60" in best["name"] or "strict" in best["name"])
    # Extra caution: if holdout barely above v7 but we changed many preds, riskier
    if d7 > 25:
        believe = False

    fp16 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "model_fp16.pt"
    yolo = ROOT / "yolov8n.pt"
    fp16_mb = fp16.stat().st_size / (1024 * 1024) if fp16.exists() else 59.85
    yolo_mb = yolo.stat().st_size / (1024 * 1024) if yolo.exists() else 6.25

    report = {
        "tag": "ir_v10",
        "primary": "submission_ir_v10.csv",
        "method": f"sameT soft-blend recipe={best['name']}: {best['labels']}",
        "holdout_acc": best["full"],
        "nested_fixed": best["nested"],
        "nested_v7": v7_nest,
        "recipe": best,
        "all_recipes": results,
        "near_v7_candidates": [{k: c[k] for k in ("label", "ens", "full", "nested", "dist", "cfg")} for c in keep_sorted[:15]],
        "delta_hold_vs_v7": float(best["full"] - V7_HOLD),
        "delta_nested_vs_v7": float(best["nested"] - v7_nest),
        "disagree_vs_v7": d7,
        "disagree_vs_v9": d9,
        "disagree_vs_v9_sameT": d9s,
        "v9_sameT_vs_v9_disagree": sameT_vs_v9,
        "empty_fallback": nfb,
        "v7_public": V7_PUBLIC,
        "v9_public": V9_PUBLIC,
        "public_gap_v7": round(V7_HOLD - V7_PUBLIC, 5),
        "public_gap_v9": round(0.757085020242915 - V9_PUBLIC, 5),
        "believe_can_beat_public": bool(believe),
        "submit_recommendation": "submit" if believe else "leave_for_parent",
        "v9_sameT_submit": "no_essentially_same_as_overfit_v9" if sameT_vs_v9 <= 2 else "maybe",
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) <= 100,
        "notes": [
            f"v7 public {V7_PUBLIC} hold {V7_HOLD:.4f} gap~{V7_HOLD-V7_PUBLIC:.4f}; v9 public {V9_PUBLIC} hold 0.757 gap~{0.757-V9_PUBLIC:.4f} — REGRESS",
            f"v10 recipe={best['name']} hold={best['full']:.4f} nested={best['nested']:.4f} (v7 nested {v7_nest:.4f})",
            f"disagree v7={d7} v9={d9} v9_sameT={d9s}; v9_sameT vs v9 = {sameT_vs_v9} (clone)",
            "sameT only; soft multi-cfg blend / half-mix IR; weights near v7; no perT/top9_sel hunt",
            f"submit_recommendation={'submit' if believe else 'leave_for_parent'}",
        ],
        "elapsed_s": time.time() - t0,
    }
    Path("metrics_ir_v10.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "holdout_acc": best["full"], "nested": best["nested"], "method": report["method"],
        "disagree_v7": d7, "disagree_v9": d9, "believe": believe,
        "submit": report["submit_recommendation"], "v9_sameT_clone": sameT_vs_v9,
        "wrote": str(out),
    }, indent=2, default=float), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
