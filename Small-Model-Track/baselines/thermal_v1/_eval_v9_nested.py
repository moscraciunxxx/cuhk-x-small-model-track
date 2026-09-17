import json, numpy as np
from pathlib import Path
from fuse_ir_v9 import load_members, softmax_np, nested_fixed

ROOT = Path(".")
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
w_acc = np.array([max(m["acc"], 1e-3) for m in members], float)
w_acc /= w_acc.sum()
members9 = [m for m in members if m["tag"] != "pool_seed55"]
stack9 = np.stack([m["logits"] for m in members9], 0)
variants = {}
for k in range(3, len(members) + 1):
    variants[f"top{k}_sel"] = np.mean(stack[:k], 0)
variants["all_sel"] = np.mean(stack, 0)
variants["all_acc_w_sel"] = np.tensordot(w_acc, stack, axes=(0, 0))
variants["all9_sel"] = np.mean(stack9, 0)
variants["all9_base"] = np.mean([m["base"] for m in members9], 0)
sm = np.mean([softmax_np(m["logits"]) for m in members9], 0)
variants["all9_sm"] = np.log(np.clip(sm, 1e-8, 1))
sm_all = np.mean([softmax_np(m["logits"]) for m in members], 0)
variants["all_sm"] = np.log(np.clip(sm_all, 1e-8, 1))
drop1 = [m for m in members9 if m["tag"] != "pool_seed1"]
variants["drop_seed1"] = np.mean([m["logits"] for m in drop1], 0)

v7 = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
print("V7 nested", nested_fixed(variants["all9_base"], th, mid_h, yt, yu, mask, v7))

rep = json.loads(Path("metrics_ir_v9.json").read_text(encoding="utf-8"))
cands = []
for r in rep["results"]:
    if r["triple_acc"] >= 0.753 - 1e-9:
        cands.append((r["ens"], "sameT", r["triple"], r["triple_acc"]))
    if r["geom_acc"] >= 0.753 - 1e-9:
        cands.append((r["ens"], "geom", r["geom"], r["geom_acc"]))
    if r.get("perT_acc") and r["perT_acc"] >= 0.753 - 1e-9:
        cands.append((r["ens"], "perT", r["perT"], r["perT_acc"]))

rows = []
for ens, mode, cfg, acc in sorted(cands, key=lambda x: -x[3]):
    el = variants[ens]
    if mode == "sameT":
        nest = nested_fixed(el, th, mid_h, yt, yu, mask, cfg)
        mean = nest["mean"]
        folds = nest["folds"]
    elif mode == "geom":
        folds = []
        for leave in (8, 9, 24):
            te = mask & (yu == leave)
            T = cfg["T"]
            eps = 1e-8
            la = np.log(np.clip(softmax_np(el[te], T), eps, 1))
            lb = np.log(np.clip(softmax_np(th[te], T), eps, 1))
            lc = np.log(np.clip(softmax_np(mid_h[te], T), eps, 1))
            pred = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
            folds.append({"leave": int(leave), "te_acc": float((pred == yt[te]).mean()), "n": int(te.sum())})
        mean = float(np.mean([f["te_acc"] for f in folds]))
    else:
        folds = []
        for leave in (8, 9, 24):
            te = mask & (yu == leave)
            pred = (
                cfg["wa"] * softmax_np(el[te], cfg["Ta"])
                + cfg["wb"] * softmax_np(th[te], cfg["Tb"])
                + cfg["wc"] * softmax_np(mid_h[te], cfg["Tc"])
            ).argmax(1)
            folds.append({"leave": int(leave), "te_acc": float((pred == yt[te]).mean()), "n": int(te.sum())})
        mean = float(np.mean([f["te_acc"] for f in folds]))
    rows.append((mean, acc, mode, ens, cfg, folds))
    print(f"{mode}/{ens} full={acc:.4f} nested={mean:.4f} delta_nest_v7={mean-0.751962:+.4f}")

rows.sort(key=lambda x: (-x[0], -x[1]))
print("\nTOP by nested:")
for mean, acc, mode, ens, cfg, folds in rows[:12]:
    print(f"  nested={mean:.4f} full={acc:.4f} {mode}/{ens} cfg={cfg}")
