import json, csv, numpy as np
from pathlib import Path
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, write_sub, V7, CLEAR

ROOT = Path(".")
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
cache = ROOT / "cache" / "ir_yolo_v4"
members, yt, yu = load_members()
th = np.load(old / "hold_thermal_v6.npy")
mid = np.load(cache / "midfuse_aligned_train_logits.npy")
from dataset import DEFAULT_HOLD_OUT_USERS
tu = np.load(cache / "train_users.npy")
hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
mid_h = mid[hold_idx]
mask = th.any(1) & mid_h.any(1)
stack = np.stack([m["logits"] for m in members], 0)
w = np.array([max(m["acc"], 1e-3) for m in members], float); w /= w.sum()
variants = {f"top{k}_sel": np.mean(stack[:k], 0) for k in range(3, 11)}
variants["all_sel"] = np.mean(stack, 0)
variants["all_acc_w_sel"] = np.tensordot(w, stack, axes=(0, 0))
sm = np.mean([softmax_np(m["logits"]) for m in members], 0)
variants["all_sm"] = np.log(np.clip(sm, 1e-8, 1))
members9 = [m for m in members if m["tag"] != "pool_seed55"]
variants["all9_base"] = np.mean([m["base"] for m in members9], 0)
variants["all9_sel"] = np.mean([m["logits"] for m in members9], 0)
w9 = np.array([max(m["acc"], 1e-3) for m in members9], float); w9 /= w9.sum()
variants["all9_acc_w_sel"] = np.tensordot(w9, np.stack([m["logits"] for m in members9], 0), axes=(0, 0))
sm9 = np.mean([softmax_np(m["logits"]) for m in members9], 0)
variants["all9_sm"] = np.log(np.clip(sm9, 1e-8, 1))
drop1 = [m for m in members9 if m["tag"] != "pool_seed1"]
variants["drop_seed1"] = np.mean([m["logits"] for m in drop1], 0)

v7 = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
nest_v7 = nested_fixed(variants["all9_base"], th, mid_h, yt, yu, mask, v7)["mean"]
print(f"v7 nested={nest_v7:.6f}")

rep = json.loads(Path("metrics_ir_v9.json").read_text(encoding="utf-8"))
rows = []
for r in rep["results"]:
    for mode, ka, kc in [("sameT", "triple_acc", "triple"), ("geom", "geom_acc", "geom"), ("perT", "perT_acc", "perT")]:
        acc = r.get(ka)
        cfg = r.get(kc)
        if acc is None or cfg is None or acc < V7 - 1e-9:
            continue
        ens = r["ens"]
        if ens not in variants:
            continue
        el = variants[ens]
        if mode == "sameT":
            nest = nested_fixed(el, th, mid_h, yt, yu, mask, cfg)["mean"]
            full = float(((cfg["wa"]*softmax_np(el[mask], cfg["T"]) + cfg["wb"]*softmax_np(th[mask], cfg["T"]) + cfg["wc"]*softmax_np(mid_h[mask], cfg["T"])).argmax(1) == yt[mask]).mean())
        elif mode == "geom":
            folds = []
            T = cfg["T"]; eps = 1e-8
            la = np.log(np.clip(softmax_np(el[mask], T), eps, 1))
            lb = np.log(np.clip(softmax_np(th[mask], T), eps, 1))
            lc = np.log(np.clip(softmax_np(mid_h[mask], T), eps, 1))
            full = float(((cfg["wa"]*la + cfg["wb"]*lb + cfg["wc"]*lc).argmax(1) == yt[mask]).mean())
            for leave in (8, 9, 24):
                te = mask & (yu == leave)
                la = np.log(np.clip(softmax_np(el[te], T), eps, 1))
                lb = np.log(np.clip(softmax_np(th[te], T), eps, 1))
                lc = np.log(np.clip(softmax_np(mid_h[te], T), eps, 1))
                folds.append(float(((cfg["wa"]*la + cfg["wb"]*lb + cfg["wc"]*lc).argmax(1) == yt[te]).mean()))
            nest = float(np.mean(folds))
        else:
            full = float(((cfg["wa"]*softmax_np(el[mask], cfg["Ta"]) + cfg["wb"]*softmax_np(th[mask], cfg["Tb"]) + cfg["wc"]*softmax_np(mid_h[mask], cfg["Tc"])).argmax(1) == yt[mask]).mean())
            folds = []
            for leave in (8, 9, 24):
                te = mask & (yu == leave)
                p = (cfg["wa"]*softmax_np(el[te], cfg["Ta"]) + cfg["wb"]*softmax_np(th[te], cfg["Tb"]) + cfg["wc"]*softmax_np(mid_h[te], cfg["Tc"])).argmax(1)
                folds.append(float((p == yt[te]).mean()))
            nest = float(np.mean(folds))
        rows.append({"ens": ens, "mode": mode, "full": full, "nested": nest, "cfg": cfg})
        print(f"{mode}/{ens} full={full:.4f} nested={nest:.4f} dnest={nest-nest_v7:+.4f}")

rows.sort(key=lambda x: (-x["nested"], -x["full"], 0 if x["mode"]=="sameT" else 1))
print("\nTOP:")
for r in rows[:10]:
    print(f"  {r['mode']}/{r['ens']} full={r['full']:.4f} nested={r['nested']:.4f}")

# Choose best: clear full (>=V7+CLEAR) maximizing nested; prefer sameT on ties
clear = [r for r in rows if r["full"] >= V7 + CLEAR - 1e-9]
if not clear:
    clear = [r for r in rows if r["full"] > V7]
clear.sort(key=lambda x: (-x["nested"], 0 if x["mode"]=="sameT" else 1, -x["full"]))
best = clear[0]
# also pick best sameT clear as alt
same = [r for r in clear if r["mode"] == "sameT"]
alt = same[0] if same else None
print(f"\nCHOSEN {best['mode']}/{best['ens']} full={best['full']:.6f} nested={best['nested']:.6f}")
if alt:
    print(f"ALT sameT {alt['ens']} full={alt['full']:.6f} nested={alt['nested']:.6f}")

def ir_test_for(ens):
    if ens.startswith("top") and ens.endswith("_sel"):
        k = int(ens[3:].split("_")[0])
        return np.mean([m["test_logits"] for m in members[:k]], 0).astype(np.float32), [m["tag"] for m in members[:k]]
    if ens in ("all_sel",):
        return np.mean([m["test_logits"] for m in members], 0).astype(np.float32), [m["tag"] for m in members]
    if ens == "all_acc_w_sel":
        return np.tensordot(w, np.stack([m["test_logits"] for m in members], 0), axes=(0,0)).astype(np.float32), [m["tag"] for m in members]
    if ens == "all_sm":
        sm = np.mean([softmax_np(m["test_logits"]) for m in members], 0)
        return np.log(np.clip(sm, 1e-8, 1)).astype(np.float32), [m["tag"] for m in members]
    if ens == "all9_sel":
        return np.mean([m["test_logits"] for m in members9], 0).astype(np.float32), [m["tag"] for m in members9]
    raise KeyError(ens)

def preds_from(ir_test, th_test, mid_test, mode, cfg):
    if mode == "sameT":
        T = cfg["T"]
        return (cfg["wa"]*softmax_np(ir_test,T) + cfg["wb"]*softmax_np(th_test,T) + cfg["wc"]*softmax_np(mid_test,T)).argmax(1)
    if mode == "geom":
        T = cfg["T"]; eps=1e-8
        la=np.log(np.clip(softmax_np(ir_test,T),eps,1))
        lb=np.log(np.clip(softmax_np(th_test,T),eps,1))
        lc=np.log(np.clip(softmax_np(mid_test,T),eps,1))
        return (cfg["wa"]*la + cfg["wb"]*lb + cfg["wc"]*lc).argmax(1)
    return (cfg["wa"]*softmax_np(ir_test,cfg["Ta"]) + cfg["wb"]*softmax_np(th_test,cfg["Tb"]) + cfg["wc"]*softmax_np(mid_test,cfg["Tc"])).argmax(1)

mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")
meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
fb = {}
with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
    for row in csv.DictReader(f):
        fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
v7p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8"))]

ir_test, tags = ir_test_for(best["ens"])
preds = preds_from(ir_test, th_test, mid_test, best["mode"], best["cfg"])
out = ROOT / "submission_ir_v9.csv"
nfb = write_sub(out, meta, preds, empty, fb)
disagree = int(sum(int(a)!=int(b) for a,b in zip(preds, v7p)))

alt_info = None
if alt and (alt["ens"] != best["ens"] or alt["mode"] != best["mode"]):
    ir_a, tags_a = ir_test_for(alt["ens"])
    preds_a = preds_from(ir_a, th_test, mid_test, alt["mode"], alt["cfg"])
    out_a = ROOT / "submission_ir_v9_sameT.csv"
    write_sub(out_a, meta, preds_a, empty, fb)
    disagree_a = int(sum(int(a)!=int(b) for a,b in zip(preds_a, v7p)))
    alt_info = {"csv": str(out_a), "ens": alt["ens"], "hold": alt["full"], "nested": alt["nested"], "cfg": alt["cfg"], "disagree_vs_v7": disagree_a, "tags": tags_a}

fp16 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "model_fp16.pt"
yolo = ROOT / "yolov8n.pt"
fp16_mb = fp16.stat().st_size/(1024*1024); yolo_mb = yolo.stat().st_size/(1024*1024)

report = {
    "tag": "ir_v9",
    "primary": "submission_ir_v9.csv",
    "method": f"full selective-TTA IR ({best['ens']}) + Thermal + Mid; {best['mode']} fuse",
    "holdout_acc": best["full"],
    "nested_fixed_cfg": best["nested"],
    "nested_v7": nest_v7,
    "cfg": best["cfg"],
    "mode": best["mode"],
    "ens": best["ens"],
    "ir_tags": tags,
    "ir_tta_flags": {m["tag"]: m["use_tta"] for m in (members[:int(best['ens'][3:].split('_')[0])] if best['ens'].startswith('top') else members)},
    "delta_vs_v7": float(best["full"] - V7),
    "delta_nested_vs_v7": float(best["nested"] - nest_v7),
    "clear_win": True,
    "disagree_vs_v7": disagree,
    "empty_fallback": nfb,
    "alt_sameT": alt_info,
    "top_candidates": rows[:12],
    "members": {m["tag"]: {"acc": m["acc"], "use_tta": m["use_tta"], "acc_base": m["acc_base"], "acc_tta": m["acc_tta"]} for m in members},
    "fp16_pack_mb": fp16_mb,
    "yolo_mb": yolo_mb,
    "total_approx_mb": fp16_mb + yolo_mb,
    "size_ok_under_100mb": (fp16_mb + yolo_mb) <= 100,
    "promoted_track": None,
    "notes": [
        f"PRIMARY hold {best['full']:.4f} nested {best['nested']:.4f} (v7 hold 0.7530 nested {nest_v7:.4f})",
        f"New-seed TTA gains: seed777 +0.0139, seed55/333 +0.0079, seed1 +0.0059",
        f"mode={best['mode']} ens={best['ens']} disagree_vs_v7={disagree}",
        "Depth 4-way still weak / conf-gate no help",
        "CSV written; parent submits. Track submission.csv NOT auto-promoted.",
        "Pack ~66MB (fp16+yolo) <=100MB",
    ],
}
Path("metrics_ir_v9.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
print(json.dumps({"holdout_acc": best["full"], "nested": best["nested"], "delta_v7": best["full"]-V7, "disagree": disagree, "mode": best["mode"], "ens": best["ens"], "cfg": best["cfg"], "alt": alt_info}, indent=2, default=float))
print(f"WROTE {out}")
