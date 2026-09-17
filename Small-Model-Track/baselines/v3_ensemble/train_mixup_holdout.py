
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DEFAULT_HOLD_OUT_USERS, CachedDualDataset, load_skel_train_cache
from model import build_model, count_parameters

ROOT = Path(__file__).resolve().parent

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=28)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    args=p.parse_args(); set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "mixup", args.mixup, flush=True)
    cache=ROOT/"cache"
    X,y,users,meta=load_skel_train_cache(cache)
    imu=np.load(cache/"imu_train.npz"); Xi,has=imu["X"],imu["has_imu"].astype(bool)
    hold=set(DEFAULT_HOLD_OUT_USERS)
    tr=np.where(~np.isin(users,list(hold)))[0]; va=np.where(np.isin(users,list(hold)))[0]
    tr_ds=CachedDualDataset(X,Xi,y,users,tr,has,augment=True,seed=args.seed)
    va_ds=CachedDualDataset(X,Xi,y,users,va,has,augment=False,seed=args.seed)
    tr_ld=DataLoader(tr_ds,batch_size=args.batch_size,shuffle=True)
    va_ld=DataLoader(va_ds,batch_size=args.batch_size,shuffle=False)
    model=build_model("midfuse",num_classes=40).to(device)
    print("params", count_parameters(model), flush=True)
    counts=np.bincount(tr_ds.labels,minlength=40).astype(np.float64); counts=np.maximum(counts,1)
    cw=torch.tensor(np.clip(counts.sum()/(40*counts),0.25,8.0),dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=cw,label_smoothing=args.label_smoothing)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=-1.0; patience=args.patience; ckpt=ROOT/"checkpoints_mixup"/"best_holdout.pt"
    ckpt.parent.mkdir(exist_ok=True)
    for ep in range(1,args.epochs+1):
        model.train(); tl=tc=tn=0.0
        for xs,xi,y_b,_,flag in tr_ld:
            xs,xi,y_b,flag=xs.to(device),xi.to(device),y_b.to(device),flag.to(device)
            opt.zero_grad(set_to_none=True)
            if args.mixup>0 and random.random()<args.mixup_prob:
                lam=float(np.random.beta(args.mixup,args.mixup))
                idx=torch.randperm(xs.size(0),device=xs.device)
                xs2=lam*xs+(1-lam)*xs[idx]; xi2=lam*xi+(1-lam)*xi[idx]
                flag2=torch.maximum(flag,flag[idx])
                logits=model(xs2,xi2,flag2)
                loss=lam*crit(logits,y_b)+(1-lam)*crit(logits,y_b[idx])
            else:
                logits=model(xs,xi,flag); loss=crit(logits,y_b)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            bs=y_b.size(0); tl+=loss.item()*bs; tc+=(logits.argmax(1)==y_b).sum().item(); tn+=bs
        model.eval(); vl=vc=vn=0.0
        with torch.no_grad():
            for xs,xi,y_b,_,flag in va_ld:
                xs,xi,y_b,flag=xs.to(device),xi.to(device),y_b.to(device),flag.to(device)
                logits=model(xs,xi,flag); loss=crit(logits,y_b)
                bs=y_b.size(0); vl+=loss.item()*bs; vc+=(logits.argmax(1)==y_b).sum().item(); vn+=bs
        sch.step()
        tra,vaa=tc/tn,vc/vn
        print(f"[mixup-hold] ep {ep}/{args.epochs} train={tra:.4f} val={vaa:.4f}", flush=True)
        if vaa>=best:
            best=vaa; patience=args.patience
            torch.save({"model_state":model.state_dict(),"model_name":"midfuse","num_classes":40,
                        "val_acc":best,"epoch":ep,"dual":True,"mixup":args.mixup}, ckpt)
        else:
            patience-=1
            if patience<=0:
                print("early stop", ep, flush=True); break
    out={"holdout_best":best,"ckpt":str(ckpt),"mixup":args.mixup,"epochs":args.epochs}
    Path("metrics_mixup_holdout.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("DONE", out, flush=True)

if __name__=="__main__":
    main()
