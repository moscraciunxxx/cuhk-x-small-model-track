"""Refine class-conditional Depth routing with proper LOUO nested + hold report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from fuse_ir_v9 import load_members, softmax_np

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
logs=[np.asarray(torch.load(ROOT/"checkpoints"/"depth_yolo_r2p1d18_v12"/f"pool_seed{s}.pt", map_location="cpu", weights_only=False)["hold_logits"], np.float32) for s in [42,123,7]]
dens=np.mean(logs,0)
T=2.5
pir,pth,pmid,pdep=[softmax_np(z,T) for z in (ir,th,mid_h,dens)]
v7p=V7_CFG["wa"]*pir+V7_CFG["wb"]*pth+V7_CFG["wc"]*pmid

def fit_wc(tr_idx, scale, mode="pred_class"):
    """Per-class blend weight from train fold."""
    w_c=np.zeros(40)
    for c in range(40):
        if mode=="pred_class":
            m=tr_idx & (v7p.argmax(1)==c)
        else:
            m=tr_idx & (yt==c)
        if m.sum()<5:
            continue
        a_v=float((v7p[m].argmax(1)==yt[m]).mean())
        a_d=float((pdep[m].argmax(1)==yt[m]).mean())
        # also consider blend
        best=0.0
        for w in np.linspace(0,1,11):
            a=float((((1-w)*v7p[m]+w*pdep[m]).argmax(1)==yt[m]).mean())
            best=max(best, a-a_v, 0)
        w_c[c]=np.clip(best*scale, 0, 1)
    return w_c

def apply_wc(w_c, idx=None):
    out=v7p.copy()
    sel=np.arange(len(yt)) if idx is None else np.where(idx)[0]
    for i in sel:
        c=int(v7p[i].argmax())
        w=float(w_c[c])
        if w>0:
            out[i]=(1-w)*v7p[i]+w*pdep[i]
    return out

results=[]
for scale in [1,2,3,4,5,6,8]:
    for mode in ["pred_class","true_class"]:
        # nested
        folds=[]
        for leave in (8,9,24):
            te=mask&(yu==leave); tr=mask&(yu!=leave)
            wc=fit_wc(tr, scale, mode=mode)
            out=apply_wc(wc)
            folds.append({"leave":int(leave),"te":float((out[te].argmax(1)==yt[te]).mean()),
                          "n_pos":int((wc>0).sum()), "w_mean":float(wc[wc>0].mean()) if (wc>0).any() else 0})
        nest=float(np.mean([f["te"] for f in folds]))
        # optimistic hold: fit on all mask
        wc_all=fit_wc(mask, scale, mode=mode)
        out_h=apply_wc(wc_all)
        hold=float((out_h[mask].argmax(1)==yt[mask]).mean())
        dis=int((out_h[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
        # fixed-cfg nested using wc_all (optimistic nested)
        nest_fix=[]
        for leave in (8,9,24):
            te=mask&(yu==leave)
            nest_fix.append(float((out_h[te].argmax(1)==yt[te]).mean()))
        nest_fixed=float(np.mean(nest_fix))
        row={"scale":scale,"mode":mode,"nested_retune":nest,"nested_fixed_opt":nest_fixed,
             "hold_opt":hold,"dis_opt":dis,"folds":folds,
             "clear_retune": nest>=V7_NESTED+MIN_D and hold>=V7_HOLD+MIN_D and dis>=MIN_DIS}
        results.append(row)
        print(f"scale={scale} mode={mode} nest_rt={nest:.4f} hold_opt={hold:.4f} nest_fix={nest_fixed:.4f} dis={dis}", flush=True)

results=sorted(results,key=lambda d:(-d["clear_retune"],-d["nested_retune"],-d["hold_opt"]))
print("BEST", {k:results[0][k] for k in results[0] if k!="folds"}, flush=True)
print("folds", results[0]["folds"], flush=True)

# Also: only override when depth alone beats v7 by margin on that class AND depth conf high
best2=None
for scale in [2,3,4,5]:
    for dthr in [0.25,0.3,0.35,0.4]:
        folds=[]
        for leave in (8,9,24):
            te=mask&(yu==leave); tr=mask&(yu!=leave)
            wc=fit_wc(tr, scale, "pred_class")
            out=v7p.copy()
            for i in np.where(te)[0]:
                c=int(v7p[i].argmax()); w=float(wc[c])
                if w>0 and pdep[i].max()>=dthr:
                    out[i]=(1-w)*v7p[i]+w*pdep[i]
            folds.append(float((out[te].argmax(1)==yt[te]).mean()))
        nest=float(np.mean(folds))
        if best2 is None or nest>best2[0]:
            best2=(nest, scale, dthr, folds)
print("best_with_dthr", best2, flush=True)

out={"v7_hold":V7_HOLD,"v7_nested":V7_NESTED,"results":[{k:v for k,v in r.items() if k!="folds"} for r in results[:12]],
     "best":{k:results[0][k] for k in results[0] if k!="folds"},"best_folds":results[0]["folds"],
     "best_dthr":{"nest":best2[0],"scale":best2[1],"dthr":best2[2],"folds":best2[3]},
     "any_clear":any(r["clear_retune"] for r in results)}
(ROOT/"metrics_probe_classcond_v12.json").write_text(json.dumps(out,indent=2))
print("any_clear", out["any_clear"], flush=True)
