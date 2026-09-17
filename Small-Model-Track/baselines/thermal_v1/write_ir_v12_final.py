"""Final v12 fuse: Depth IR-box ens5 + best seed11; linear + gated vs ir_v7 gate."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from write_ir_v11 import fuse4_sameT, nested_fixed4, nested_retune4, align_depth_hold

ROOT = Path(__file__).resolve().parent
V7_HOLD, V7_NESTED = 0.7530364372469636, 0.7519623092355898
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
MIN_D, MIN_DIS = 0.01, 20

t0=time.time()
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

dp = np.load(ROOT/"checkpoints"/"depth_yolo_r2p1d18_v12"/"hold_logits_v12.npz", allow_pickle=True)
# ens may need align
try:
    dens = align_depth_hold(dp, yt, yu, cache/"train_meta.json",
                            ROOT/"cache"/"depth_color_yolo_v4_irbox"/"train_meta.json")
except Exception as e:
    print("align fallback", e, flush=True)
    dens = dp["ens"].astype(np.float32)
    assert len(dens)==len(yt) and np.array_equal(dp["y"], yt)

# also best member from stack
stack = dp["logits"].astype(np.float32)  # S,N,C
seeds = list(dp["seeds"]); scores=list(dp["scores"])
best_i = int(np.argmax(scores))
dbest = stack[best_i]
print(f"depth ens={(dens.argmax(1)==yt).mean():.4f} best_seed={seeds[best_i]}={(dbest.argmax(1)==yt).mean():.4f}", flush=True)
print(f"members={list(zip([int(s) for s in seeds],[float(x) for x in scores]))}", flush=True)

T=2.5
v7p=V7_CFG["wa"]*softmax_np(ir,T)+V7_CFG["wb"]*softmax_np(th,T)+V7_CFG["wc"]*softmax_np(mid_h,T)
print(f"v7={(v7p[mask].argmax(1)==yt[mask]).mean():.4f}", flush=True)

candidates=[]
for name, dlog in [("ens5", dens), ("best11", dbest), ("top2", stack[np.argsort(scores)[-2:]].mean(0)),
                   ("top3", stack[np.argsort(scores)[-3:]].mean(0))]:
    acc4,cfg4=fuse4_sameT(ir,th,mid_h,dlog,yt,mask,[1.5,2,2.5,3],ngrid=21)
    nest4=nested_fixed4(ir,th,mid_h,dlog,yt,yu,mask,cfg4)
    nest_rt=nested_retune4(ir,th,mid_h,dlog,yt,yu,mask,[1.5,2,2.5,3],ngrid=13)
    # disagree vs v7 preds
    T4=cfg4["T"]
    p=(cfg4["wa"]*softmax_np(ir,T4)+cfg4["wb"]*softmax_np(th,T4)+cfg4["wc"]*softmax_np(mid_h,T4)+cfg4["wd"]*softmax_np(dlog,T4))
    dis=int((p[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
    row={"name":f"4way_{name}","cfg":cfg4,"nested_fixed":nest4,"nested_retune":nest_rt["mean"],
         "dh":cfg4["acc"]-V7_HOLD,"dn":nest4-V7_NESTED,"dis":dis,
         "clear":cfg4["acc"]>=V7_HOLD+MIN_D and nest4>=V7_NESTED+MIN_D and dis>=MIN_DIS,
         "depth_acc":float((dlog.argmax(1)==yt).mean())}
    candidates.append(row)
    print(row["name"], "hold", round(cfg4["acc"],4), "wd", round(cfg4["wd"],4), "nest", round(nest4,4), "dis", dis, flush=True)

    # gates
    pdep=softmax_np(dlog,2.5)
    for thr in [0.25,0.35,0.45]:
        for w in [0.2,0.4,0.6,1.0]:
            out=v7p.copy(); low=v7p.max(1)<thr
            out[low]=(1-w)*v7p[low]+w*pdep[low]
            acc=float((out[mask].argmax(1)==yt[mask]).mean())
            folds=[float((out[mask&(yu==u)].argmax(1)==yt[mask&(yu==u)]).mean()) for u in (8,9,24)]
            nest=float(np.mean(folds)); dis=int((out[mask].argmax(1)!=v7p[mask].argmax(1)).sum())
            candidates.append({"name":f"gate_{name}_t{thr}_w{w}","acc":acc,"nested_fixed":nest,"dis":dis,
                               "dh":acc-V7_HOLD,"dn":nest-V7_NESTED,
                               "clear":acc>=V7_HOLD+MIN_D and nest>=V7_NESTED+MIN_D and dis>=MIN_DIS})

# complementarity
for name,dlog in [("ens5",dens),("best11",dbest)]:
    dep_p=dlog.argmax(1); v7w=v7p.argmax(1)!=yt
    print(f"{name} corrects v7-wrong: {int((dep_p[mask&v7w]==yt[mask&v7w]).sum())}/{int((mask&v7w).sum())}", flush=True)
    dis=ir.argmax(1)!=dep_p
    print(f"  IR vs {name} disagree={dis.sum()} on_dis IR={(ir.argmax(1)[dis]==yt[dis]).mean():.3f} Dep={(dep_p[dis]==yt[dis]).mean():.3f}", flush=True)

cands=sorted(candidates,key=lambda d:(-d.get("clear",False), -d.get("nested_fixed",0), -d.get("acc",d.get("cfg",{}).get("acc",0))))
print("TOP clear/nested:", flush=True)
for c in cands[:8]:
    print({k:c[k] for k in c if k!="cfg"}, flush=True)

any_clear=any(c.get("clear") for c in candidates)
report={
  "tag":"ir_v12_final_probe",
  "depth":{"ens":float((dens.argmax(1)==yt).mean()),
           "members":{f"s{int(s)}":float(a) for s,a in zip(seeds,scores)},
           "best":float((dbest.argmax(1)==yt).mean())},
  "v7":{"hold":V7_HOLD,"nested":V7_NESTED,"public":0.69154},
  "top":[{k:(v if not isinstance(v,dict) else v) for k,v in c.items()} for c in cands[:15]],
  "any_clear":any_clear,
  "wrote_csv":False,
  "submit_recommendation":"leave_for_parent",
  "gpu":"RTX3060 free vs LMT throughout; depth IR-box train used GPU exclusively",
  "elapsed_s":time.time()-t0,
  "notes":[
    "Depth IR-box: detect~99% via IR transfer; ens hold 0.665 (was 0.582 v11); best seed11 0.681",
    "Linear 4way still wd->0; gated/agree/classcond do not clear +0.01 hold&nested with dis>=20",
    "Depth corrects only ~20/122 v7 errors even at 0.66-0.68; errors correlated with IR",
    "NO submission_ir_v12.csv — keep ir_v7 public 0.69154",
  ],
}
# serialize safely
def conv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, dict): return {k:conv(v) for k,v in o.items()}
    if isinstance(o, list): return [conv(x) for x in o]
    return o
(ROOT/"metrics_ir_v12.json").write_text(json.dumps(conv(report), indent=2))
print("any_clear", any_clear, "wrote metrics_ir_v12.json", flush=True)
print("NO CSV", flush=True)
