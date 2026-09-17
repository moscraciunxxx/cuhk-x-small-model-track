"""3-seed Depth IR-box ens fuse+gate probe vs ir_v7 (CPU; training can continue)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from write_ir_v11 import fuse4_sameT, nested_fixed4

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

ckpt_dir = ROOT / "checkpoints" / "depth_yolo_r2p1d18_v12"
seeds, logs, scores = [], [], []
for s in [42, 123, 7]:
    p = ckpt_dir / f"pool_seed{s}.pt"
    if not p.exists():
        continue
    blob = torch.load(p, map_location="cpu", weights_only=False)
    lg = np.asarray(blob["hold_logits"], dtype=np.float32)
    assert np.array_equal(np.asarray(blob["hold_y"]), yt)
    acc = float((lg.argmax(1) == yt).mean())
    seeds.append(s); logs.append(lg); scores.append(acc)
    print(f"seed{s}={acc:.4f}", flush=True)
stack = np.stack(logs, 0)
dens = stack.mean(0)
print(f"ens3={((dens.argmax(1)==yt).mean()):.4f} members={list(zip(seeds,scores))}", flush=True)

T = 2.5
v7p = V7_CFG["wa"]*softmax_np(ir,T)+V7_CFG["wb"]*softmax_np(th,T)+V7_CFG["wc"]*softmax_np(mid_h,T)
print(f"v7={(v7p[mask].argmax(1)==yt[mask]).mean():.4f}", flush=True)

acc4, cfg4 = fuse4_sameT(ir, th, mid_h, dens, yt, mask, [1.5,2,2.5,3], ngrid=21)
nest4 = nested_fixed4(ir, th, mid_h, dens, yt, yu, mask, cfg4)
print("4way", cfg4, "nested", nest4, flush=True)

acc3, cfg3 = fuse3_sameT(ir, th, dens, yt, mask, [1.5,2,2.5,3], ngrid=41)
nest3 = nested_fixed(ir, th, dens, yt, yu, mask, cfg3)["mean"]
print("3way ir+th+dep", cfg3, "nested", nest3, flush=True)

# also IR+dep+mid
acc3b, cfg3b = fuse3_sameT(ir, dens, mid_h, yt, mask, [1.5,2,2.5,3], ngrid=41)
nest3b = nested_fixed(ir, dens, mid_h, yt, yu, mask, cfg3b)["mean"]
print("3way ir+dep+mid", cfg3b, "nested", nest3b, flush=True)

pdep = softmax_np(dens, 2.5)
# gates on v7
best_gate = None
for thr in np.linspace(0.2, 0.55, 8):
    for w in np.linspace(0.15, 1.0, 8):
        out = v7p.copy()
        low = v7p.max(1) < thr
        out[low] = (1-w)*v7p[low] + w*pdep[low]
        acc = float((out[mask].argmax(1)==yt[mask]).mean())
        folds = [float((out[mask&(yu==u)].argmax(1)==yt[mask&(yu==u)]).mean()) for u in (8,9,24)]
        nest = float(np.mean(folds))
        dis = int((out[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
        row = {"thr":float(thr),"w":float(w),"acc":acc,"nested":nest,"dis":dis,
               "clear": acc>=V7_HOLD+MIN_D and nest>=V7_NESTED+MIN_D and dis>=MIN_DIS}
        if best_gate is None or (row["clear"], row["nested"], row["acc"]) > (best_gate["clear"], best_gate["nested"], best_gate["acc"]):
            best_gate = row
print("best_gate", best_gate, flush=True)

# seed-agree gate on IR -> depth
bases = [m["base"] for m in members if m["tag"]!="pool_seed55"]
preds = np.stack([b.argmax(1) for b in bases],0)
agree = (preds == ir.argmax(1)[None,:]).mean(0)
best_ag = None
for thr in [0.4,0.5,0.55,0.6,0.67,0.75]:
    for w in [0.2,0.3,0.5,0.7,1.0]:
        out=v7p.copy(); low=agree<thr
        out[low]=(1-w)*v7p[low]+w*pdep[low]
        acc=float((out[mask].argmax(1)==yt[mask]).mean())
        folds=[float((out[mask&(yu==u)].argmax(1)==yt[mask&(yu==u)]).mean()) for u in (8,9,24)]
        nest=float(np.mean(folds)); dis=int((out[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
        row={"signal":"agree","thr":thr,"w":w,"acc":acc,"nested":nest,"dis":dis,
             "clear":acc>=V7_HOLD+MIN_D and nest>=V7_NESTED+MIN_D and dis>=MIN_DIS}
        if best_ag is None or (row["clear"],row["nested"],row["acc"])>(best_ag["clear"],best_ag["nested"],best_ag["acc"]):
            best_ag=row
print("best_agree_gate", best_ag, flush=True)

dep_p = dens.argmax(1)
v7w = v7p.argmax(1)!=yt
print(f"dep ens corrects v7-wrong: {int((dep_p[mask&v7w]==yt[mask&v7w]).sum())}/{int((mask&v7w).sum())}", flush=True)
dis = ir.argmax(1)!=dep_p
print(f"IR vs Dep disagree={dis.sum()} on_dis IR={(ir.argmax(1)[dis]==yt[dis]).mean():.3f} Dep={(dep_p[dis]==yt[dis]).mean():.3f}", flush=True)

out={
  "members": {f"s{s}": sc for s,sc in zip(seeds,scores)},
  "ens3": float((dens.argmax(1)==yt).mean()),
  "fuse4": {"cfg":cfg4,"nested":nest4,"dh":cfg4["acc"]-V7_HOLD,"dn":nest4-V7_NESTED},
  "fuse3_ir_th_dep": {"cfg":cfg3,"nested":nest3,"dh":cfg3["acc"]-V7_HOLD,"dn":nest3-V7_NESTED},
  "fuse3_ir_dep_mid": {"cfg":cfg3b,"nested":nest3b,"dh":cfg3b["acc"]-V7_HOLD,"dn":nest3b-V7_NESTED},
  "best_gate": best_gate, "best_agree_gate": best_ag,
  "any_clear": any([
      cfg4["acc"]>=V7_HOLD+MIN_D and nest4>=V7_NESTED+MIN_D,
      best_gate["clear"], best_ag["clear"],
  ]),
}
(ROOT/"metrics_probe_depth3_v12.json").write_text(json.dumps(out,indent=2,default=str))
print(json.dumps(out,indent=2,default=str), flush=True)
