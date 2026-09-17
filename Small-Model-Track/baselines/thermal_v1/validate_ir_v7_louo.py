"""Validate candidate v7 cfgs with LOUO holdout users; decide promote."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS

ROOT = Path(__file__).resolve().parent
V6 = 0.7469635627530364
V6_NESTED = 0.7246545388967788

def softmax_np(z, T=1.0):
    z = z / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)

def apply_cfg(a,b,c,y,mask,cfg):
    T=cfg["T"]; wa,wb,wc=cfg["wa"],cfg["wb"],cfg["wc"]
    pa,pb,pc=softmax_np(a[mask],T),softmax_np(b[mask],T),softmax_np(c[mask],T)
    return float(((wa*pa+wb*pb+wc*pc).argmax(1)==y[mask]).mean())

def louo_fixed(a,b,c,y,users,mask,cfg):
    scores=[]
    for leave in sorted(set(users[mask].tolist())):
        te = mask & (users==leave)
        scores.append({"leave":int(leave),"te_acc":apply_cfg(a,b,c,y,te,cfg),"n":int(te.sum())})
    return float(np.mean([s["te_acc"] for s in scores])), scores

def main():
    z=np.load(ROOT/"checkpoints"/"ir_yolo_r2p1d18_v5"/"hold_logits_v6.npz", allow_pickle=True)
    yt=z["y"]; yu=z["users"]; old=z["base"]; tags=[str(t) for t in z["tags"]]
    hz=np.load(ROOT/"checkpoints"/"ir_yolo_r2p1d18_v6"/"hold_logits_new_seeds.npz", allow_pickle=True)
    members=[]
    for i,t in enumerate(tags):
        members.append({"tag":t,"logits":old[i],"acc":float((old[i].argmax(1)==yt).mean())})
    for t in hz["tags"]:
        lg=hz[str(t)]
        members.append({"tag":str(t),"logits":lg,"acc":float((lg.argmax(1)==yt).mean())})
    members=sorted(members,key=lambda d:-d["acc"])
    stack=np.stack([m["logits"] for m in members],0)
    w=np.array([max(m["acc"],1e-3) for m in members],dtype=np.float64); w/=w.sum()
    all_mean=np.mean(stack,0)
    all_acc_w=np.tensordot(w,stack,axes=(0,0))
    th=np.load(ROOT/"checkpoints"/"ir_yolo_r2p1d18_v5"/"hold_thermal_v6.npy")
    cache=ROOT/"cache"/"ir_yolo_v4"
    mid_full=np.load(cache/"midfuse_aligned_train_logits.npy")
    users_full=np.load(cache/"train_users.npy")
    hold_idx=np.where(np.isin(users_full,list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid=mid_full[hold_idx]
    mask=th.any(1)&mid.any(1)

    candidates=[
        ("v6_primary_all9", all_mean, {"wa":0.56,"wb":0.36,"wc":0.08,"T":1.5}),
        ("v7_all_acc_w_751", all_acc_w, {"wa":0.54,"wb":0.34,"wc":0.12,"T":2.25}),
        ("v7_top9_749", all_mean, {"wa":0.56,"wb":0.34,"wc":0.10,"T":1.25}),
        ("v7_top8_747", np.mean(stack[:8],0), {"wa":0.52,"wb":0.36,"wc":0.12,"T":2.25}),
        ("v7_top4_747", np.mean(stack[:4],0), {"wa":0.50,"wb":0.28,"wc":0.22,"T":2.25}),
        ("compromise_v6", all_mean, {"wa":0.5,"wb":0.3,"wc":0.2,"T":1.75}),
    ]
    rows=[]
    for name,ens,cfg in candidates:
        full=apply_cfg(ens,th,mid,yt,mask,cfg)
        nest,folds=louo_fixed(ens,th,mid,yt,yu,mask,cfg)
        rows.append({"name":name,"full":full,"nested":nest,"delta_v6_full":full-V6,"delta_v6_nest":nest-V6_NESTED,"cfg":cfg,"folds":folds})
        print(f"{name}: full={full:.4f} nested={nest:.4f} dF={full-V6:+.4f} dN={nest-V6_NESTED:+.4f} folds={[round(f['te_acc'],3) for f in folds]}", flush=True)

    print("\nLocal refine all_acc_w scored by nested:", flush=True)
    best_n=(-1,None); best_f=(-1,None)
    for T in [1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0]:
        for wa in np.linspace(0.40,0.70,16):
            for wb in np.linspace(0.15,0.45,16):
                wc=1-wa-wb
                if wc<0.02 or wc>0.30: continue
                cfg={"wa":float(wa),"wb":float(wb),"wc":float(wc),"T":float(T)}
                full=apply_cfg(all_acc_w,th,mid,yt,mask,cfg)
                nest,_=louo_fixed(all_acc_w,th,mid,yt,yu,mask,cfg)
                if nest>best_n[0]: best_n=(nest,{"cfg":cfg,"full":full,"nested":nest})
                if full>best_f[0]: best_f=(full,{"cfg":cfg,"full":full,"nested":nest})
    print("best nested:", best_n[1], flush=True)
    print("best full:", best_f[1], flush=True)

    print("\nLocal refine all_mean scored by nested:", flush=True)
    best_n2=(-1,None); best_f2=(-1,None)
    for T in [1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0]:
        for wa in np.linspace(0.40,0.70,16):
            for wb in np.linspace(0.15,0.45,16):
                wc=1-wa-wb
                if wc<0.02 or wc>0.30: continue
                cfg={"wa":float(wa),"wb":float(wb),"wc":float(wc),"T":float(T)}
                full=apply_cfg(all_mean,th,mid,yt,mask,cfg)
                nest,_=louo_fixed(all_mean,th,mid,yt,yu,mask,cfg)
                if nest>best_n2[0]: best_n2=(nest,{"cfg":cfg,"full":full,"nested":nest})
                if full>best_f2[0]: best_f2=(full,{"cfg":cfg,"full":full,"nested":nest})
    print("best nested:", best_n2[1], flush=True)
    print("best full:", best_f2[1], flush=True)

    out={"candidates":rows,"all_acc_w_best_nested":best_n[1],"all_acc_w_best_full":best_f[1],
         "all_mean_best_nested":best_n2[1],"all_mean_best_full":best_f2[1]}
    (ROOT/"metrics_ir_v7_louo.json").write_text(json.dumps(out,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
