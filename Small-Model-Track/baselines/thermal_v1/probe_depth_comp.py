"""Deeper complementarity: stack / margin / thermal-agree gates with depth ens3."""
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
logs=[]
for s in [42,123,7]:
    blob=torch.load(ROOT/"checkpoints"/"depth_yolo_r2p1d18_v12"/f"pool_seed{s}.pt", map_location="cpu", weights_only=False)
    logs.append(np.asarray(blob["hold_logits"],np.float32))
dens=np.mean(logs,0)
T=2.5
pir,pth,pmid,pdep=[softmax_np(z,T) for z in (ir,th,mid_h,dens)]
v7p=V7_CFG["wa"]*pir+V7_CFG["wb"]*pth+V7_CFG["wc"]*pmid

def eval_out(out, name):
    acc=float((out[mask].argmax(1)==yt[mask]).mean())
    folds=[float((out[mask&(yu==u)].argmax(1)==yt[mask&(yu==u)]).mean()) for u in (8,9,24)]
    nest=float(np.mean(folds)); dis=int((out[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
    clear=acc>=V7_HOLD+MIN_D and nest>=V7_NESTED+MIN_D and dis>=MIN_DIS
    print(f"{name}: acc={acc:.4f} nest={nest:.4f} dis={dis} clear={clear}", flush=True)
    return {"name":name,"acc":acc,"nested":nest,"dis":dis,"clear":clear}

rows=[]
# margin gate: when v7 top1-top2 small, blend depth
part=np.partition(v7p,-2,axis=1)
marg=part[:,-1]-part[:,-2]
for mthr in [0.02,0.05,0.08,0.1,0.15,0.2]:
    for w in [0.2,0.3,0.5,0.7,1.0]:
        out=v7p.copy(); low=marg<mthr
        out[low]=(1-w)*v7p[low]+w*pdep[low]
        rows.append(eval_out(out, f"margin<{mthr}_w{w}"))

# only swap when depth agrees with thermal
for thr in [0.25,0.35,0.45]:
    out=v7p.copy()
    low=v7p.max(1)<thr
    agree_th=(pdep.argmax(1)==pth.argmax(1)) & low
    out[agree_th]=0.5*pdep[agree_th]+0.5*pth[agree_th]
    rows.append(eval_out(out, f"low{thr}_dep=th"))

# only swap when depth conf high
for dthr in [0.35,0.4,0.45,0.5]:
    for vthr in [0.3,0.4,0.5]:
        out=v7p.copy()
        swap=(v7p.max(1)<vthr)&(pdep.max(1)>=dthr)
        out[swap]=pdep[swap]
        rows.append(eval_out(out, f"v7<{vthr}_dep>={dthr}"))

# class-conditional weight: estimate per-class depth lift on LOUO nested style
# For each leave-user: fit class weights on other users, eval on leave
def nested_class_w():
    folds=[]
    for leave in (8,9,24):
        te=mask&(yu==leave); tr=mask&(yu!=leave)
        # per-class: if depth better than v7 on train fold, use depth when v7 predicts that class or when low conf
        w_c=np.zeros(40)
        for c in range(40):
            m=tr&(v7p.argmax(1)==c)
            if m.sum()<3: 
                w_c[c]=0; continue
            # accuracy if use depth vs v7 on these
            a_v=float((v7p[m].argmax(1)==yt[m]).mean())
            a_d=float((pdep[m].argmax(1)==yt[m]).mean())
            w_c[c]=max(0, a_d-a_v)
        out=v7p.copy()
        for i in np.where(te)[0]:
            c=int(v7p[i].argmax())
            w=float(np.clip(w_c[c]*3,0,0.7))  # scale
            if w>0:
                out[i]=(1-w)*v7p[i]+w*pdep[i]
        folds.append(float((out[te].argmax(1)==yt[te]).mean()))
    return float(np.mean(folds)), folds

nest_cw, folds_cw = nested_class_w()
print(f"class_cond_nested={nest_cw:.4f} folds={folds_cw}", flush=True)

# simple stack: features = [v7_max, dep_max, v7_ent, dep_ent, agree, margin] -> pick dep if score
# fit on 2 users predict 1
def entropy(p):
    return -(p*np.log(np.clip(p,1e-8,1))).sum(1)

def nested_stack():
    folds=[]
    feats_base = np.stack([
        v7p.max(1), pdep.max(1), entropy(v7p), entropy(pdep),
        (v7p.argmax(1)==pdep.argmax(1)).astype(np.float64),
        marg, (pth.argmax(1)==pdep.argmax(1)).astype(np.float64),
    ],1)
    # label: 1 if depth right and v7 wrong
    y_swap = ((pdep.argmax(1)==yt) & (v7p.argmax(1)!=yt)).astype(np.float64)
    for leave in (8,9,24):
        te=mask&(yu==leave); tr=mask&(yu!=leave)
        Xtr, ytr = feats_base[tr], y_swap[tr]
        # ridge-ish linear: use mean of features when y=1 vs 0 as prototype
        if ytr.sum()<3:
            folds.append(float((v7p[te].argmax(1)==yt[te]).mean())); continue
        mu1=Xtr[ytr>0.5].mean(0); mu0=Xtr[ytr<0.5].mean(0)
        # score = dist to mu0 - dist to mu1 (higher => swap)
        def score(X):
            return ((X-mu0)**2).sum(1) - ((X-mu1)**2).sum(1)
        # threshold on train
        s_tr=score(Xtr)
        best_t, best_a = 0, -1
        for t in np.percentile(s_tr, np.linspace(50,99,20)):
            out=v7p.copy()
            sw=np.zeros(len(yt),dtype=bool); sw[np.where(tr)[0]] = s_tr>=t
            out[sw]=pdep[sw]
            a=float((out[tr].argmax(1)==yt[tr]).mean())
            if a>best_a: best_a, best_t=a,t
        s_te=score(feats_base[te])
        out=v7p.copy(); out[te]=np.where((s_te>=best_t)[:,None], pdep[te], v7p[te])
        folds.append(float((out[te].argmax(1)==yt[te]).mean()))
    return float(np.mean(folds)), folds

nest_st, folds_st = nested_stack()
print(f"stack_nested={nest_st:.4f} folds={folds_st}", flush=True)

rows=sorted(rows,key=lambda d:(-d["clear"],-d["nested"],-d["acc"]))
print("TOP5:", flush=True)
for r in rows[:5]:
    print(r, flush=True)

out={"top":rows[:10],"class_cond_nested":nest_cw,"stack_nested":nest_st,
     "v7":V7_HOLD,"any_clear":any(r["clear"] for r in rows) or nest_cw>=V7_NESTED+MIN_D or nest_st>=V7_NESTED+MIN_D}
(ROOT/"metrics_probe_depth_comp_v12.json").write_text(json.dumps(out,indent=2))
print("any_clear", out["any_clear"], flush=True)
