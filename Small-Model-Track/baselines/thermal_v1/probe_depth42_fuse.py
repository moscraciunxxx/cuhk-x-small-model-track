"""Quick mid-train fuse probe with depth seed42 (IR-box) vs ir_v7."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from write_ir_v11 import fuse4_sameT, nested_fixed4, nested_retune4, align_depth_hold

ROOT = Path(__file__).resolve().parent
V7_HOLD, V7_NESTED = 0.7530364372469636, 0.7519623092355898
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
MIN_D, MIN_DIS = 0.01, 20

members, yt, yu = load_members()
old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
cache = ROOT / "cache" / "ir_yolo_v4"
th = np.load(old / "hold_thermal_v6.npy")
mid = np.load(cache / "midfuse_aligned_train_logits.npy")
tu = np.load(cache / "train_users.npy")
from dataset import DEFAULT_HOLD_OUT_USERS
hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
mid_h = mid[hold_idx] if len(mid) == len(tu) else mid
if len(yu) != len(yt):
    yu = tu[hold_idx]
mask = th.any(1) & mid_h.any(1)
ir = np.mean([m["base"] for m in members if m["tag"] != "pool_seed55"], 0).astype(np.float32)

# depth from ckpt hold_logits
blob = torch.load(ROOT / "checkpoints" / "depth_yolo_r2p1d18_v12" / "pool_seed42.pt",
                  map_location="cpu", weights_only=False)
dlog = np.asarray(blob["hold_logits"], dtype=np.float32)
dy = np.asarray(blob["hold_y"])
assert np.array_equal(dy, yt), "hold y mismatch — may need align"
print(f"depth42={(dlog.argmax(1)==yt).mean():.4f} ir={(ir.argmax(1)==yt).mean():.4f}", flush=True)

# v7 recon
T = 2.5
v7p = V7_CFG["wa"]*softmax_np(ir,T)+V7_CFG["wb"]*softmax_np(th,T)+V7_CFG["wc"]*softmax_np(mid_h,T)
print(f"v7 recon={(v7p[mask].argmax(1)==yt[mask]).mean():.4f}", flush=True)

# 4way fuse
acc4, cfg4 = fuse4_sameT(ir, th, mid_h, dlog, yt, mask, [1.5, 2.0, 2.5, 3.0], ngrid=21)
nest4 = nested_fixed4(ir, th, mid_h, dlog, yt, yu, mask, cfg4)
print("4way", cfg4, "nested", nest4, flush=True)

# 3way IR+Th+Depth (drop mid)
acc3, cfg3 = fuse3_sameT(ir, th, dlog, yt, mask, [1.5, 2.0, 2.5, 3.0], ngrid=41)
nest3 = nested_fixed(ir, th, dlog, yt, yu, mask, cfg3)["mean"]
print("3way ir+th+dep", cfg3, "nested", nest3, flush=True)

# confidence gate: v7 for high conf, blend depth when low
pdep = softmax_np(dlog, 2.5)
attempts = []
for thr in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for w in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        out = v7p.copy()
        low = v7p.max(1) < thr
        out[low] = (1-w)*v7p[low] + w*pdep[low]
        acc = float((out[mask].argmax(1)==yt[mask]).mean())
        # nested
        folds=[]
        for u in (8,9,24):
            te=mask&(yu==u)
            folds.append(float((out[te].argmax(1)==yt[te]).mean()))
        nest=float(np.mean(folds))
        dis=int((out[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
        attempts.append({"thr":thr,"w":w,"acc":acc,"nested":nest,"dis":dis,
                         "clear":acc>=V7_HOLD+MIN_D and nest>=V7_NESTED+MIN_D and dis>=MIN_DIS})
attempts=sorted(attempts,key=lambda d:(-d["clear"],-d["nested"],-d["acc"]))
print("top gate", attempts[:5], flush=True)

# also when depth conf > v7 conf and v7 maxp low
for thr in [0.3,0.4,0.5]:
    out=v7p.copy()
    swap=(v7p.max(1)<thr)&(pdep.max(1)>v7p.max(1))
    out[swap]=pdep[swap]
    acc=float((out[mask].argmax(1)==yt[mask]).mean())
    folds=[float((out[mask&(yu==u)].argmax(1)==yt[mask&(yu==u)]).mean()) for u in (8,9,24)]
    nest=float(np.mean(folds)); dis=int((out[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
    print(f"swap_if_dep_conf thr={thr} n={int(swap.sum())} acc={acc:.4f} nest={nest:.4f} dis={dis}", flush=True)

# disagree IR vs depth
ir_p, dep_p = ir.argmax(1), dlog.argmax(1)
dis = ir_p!=dep_p
print(f"IR vs Dep42 disagree={dis.sum()} on_dis IR={(ir_p[dis]==yt[dis]).mean():.3f} Dep={(dep_p[dis]==yt[dis]).mean():.3f}", flush=True)
v7w = v7p.argmax(1)!=yt
print(f"depth corrects v7-wrong: {int((dep_p[mask&v7w]==yt[mask&v7w]).sum())}/{int((mask&v7w).sum())}", flush=True)

out={
  "depth42": float((dlog.argmax(1)==yt).mean()),
  "fuse4": {"cfg":cfg4,"nested":nest4,"delta_h":cfg4["acc"]-V7_HOLD,"delta_n":nest4-V7_NESTED},
  "fuse3_ir_th_dep": {"cfg":cfg3,"nested":nest3,"delta_h":cfg3["acc"]-V7_HOLD,"delta_n":nest3-V7_NESTED},
  "best_gate": attempts[0],
  "any_clear": any(a["clear"] for a in attempts) or (
      cfg4["acc"]>=V7_HOLD+MIN_D and nest4>=V7_NESTED+MIN_D),
}
(ROOT/"metrics_probe_depth42_v12.json").write_text(json.dumps(out,indent=2,default=str))
print(json.dumps(out,indent=2,default=str), flush=True)
