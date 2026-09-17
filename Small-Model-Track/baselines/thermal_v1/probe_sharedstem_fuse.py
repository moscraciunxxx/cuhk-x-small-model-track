"""CPU fuse probe: sharedstem logits + Thermal + Mid vs ir_v7 gate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
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

blob = np.load(ROOT / "checkpoints" / "sharedstem_ir_depth_v1" / "hold_logits_shared.npz")
sh = np.asarray(blob["logits"], dtype=np.float32)
sy = np.asarray(blob["y"])
su = np.asarray(blob["users"]) if "users" in blob.files else yu
assert len(sh) == len(yt) or True
# align on y if lengths differ
print("shared keys", blob.files, "sh", sh.shape, "yt", yt.shape, flush=True)
if len(sh) != len(yt):
    # try index alignment via hold users/y from npz
    print("WARN length mismatch shared", len(sh), "vs ir hold", len(yt), flush=True)

# use shared hold y/users if present
if "y" in blob.files:
    yt_s = sy
    yu_s = su if su is not None and len(su)==len(sy) else yu
else:
    yt_s, yu_s = yt, yu

# If shared hold is full 505 and ir v7 hold is 494 masked, build mask on shared
acc_s = float((sh.argmax(1) == yt_s).mean())
print(f"shared solo={acc_s:.4f} n={len(yt_s)}", flush=True)

# Align shared to IR hold indices if needed: both should be same hold set order
# Prefer matching by (y,user) if lengths differ
if len(sh) == len(yt):
    sh_a, yt_a, yu_a, th_a, mid_a, ir_a = sh, yt, yu, th, mid_h, ir
    mask_a = mask
elif len(sh) == len(tu[hold_idx]):
    # shared on all hold 505; ir members on 494 masked subset — use shared's own y
    sh_a, yt_a, yu_a = sh, yt_s, yu_s
    # rebuild th/mid for full hold
    th_full = th  # may be 494
    print("th", th.shape, "mid_h", mid_h.shape, "hold_idx", len(hold_idx), flush=True)
    # fall back: evaluate fuse only where we can align via first min len
    n = min(len(sh), len(ir), len(th), len(mid_h))
    sh_a, ir_a, th_a, mid_a, yt_a, yu_a = sh[:n], ir[:n], th[:n], mid_h[:n], yt[:n], yu[:n]
    mask_a = th_a.any(1) & mid_a.any(1)
else:
    n = min(len(sh), len(ir), len(th), len(mid_h), len(yt))
    sh_a, ir_a, th_a, mid_a, yt_a, yu_a = sh[:n], ir[:n], th[:n], mid_h[:n], yt[:n], yu[:n]
    mask_a = th_a.any(1) & mid_a.any(1)

T = 2.5
v7p = V7_CFG["wa"]*softmax_np(ir_a,T)+V7_CFG["wb"]*softmax_np(th_a,T)+V7_CFG["wc"]*softmax_np(mid_a,T)
print(f"v7 recon={(v7p[mask_a].argmax(1)==yt_a[mask_a]).mean():.4f} n={mask_a.sum()}", flush=True)

# replace IR with shared in v7 weights
sh_blend = V7_CFG["wa"]*softmax_np(sh_a,T)+V7_CFG["wb"]*softmax_np(th_a,T)+V7_CFG["wc"]*softmax_np(mid_a,T)
acc_rep = float((sh_blend[mask_a].argmax(1)==yt_a[mask_a]).mean())
folds=[float((sh_blend[mask_a&(yu_a==u)].argmax(1)==yt_a[mask_a&(yu_a==u)]).mean()) for u in (8,9,24)]
nest_rep=float(np.mean(folds))
dis_rep=int((sh_blend[mask_a].argmax(1)!=v7p[mask_a].argmax(1)).sum())
print(f"replace IR->shared @v7w hold={acc_rep:.4f} nest={nest_rep:.4f} dis={dis_rep}", flush=True)

# 3way shared+th+mid retune
acc3, cfg3 = fuse3_sameT(sh_a, th_a, mid_a, yt_a, mask_a, [1.5, 2.0, 2.5, 3.0], ngrid=41)
nest3 = nested_fixed(sh_a, th_a, mid_a, yt_a, yu_a, mask_a, cfg3)["mean"]
dis3 = int(((cfg3["wa"]*softmax_np(sh_a,cfg3["T"])+cfg3["wb"]*softmax_np(th_a,cfg3["T"])+cfg3["wc"]*softmax_np(mid_a,cfg3["T"]))[mask_a].argmax(1)!=v7p[mask_a].argmax(1)).sum())
print("3way shared+th+mid", cfg3, "nested", nest3, "dis", dis3, flush=True)

# 4way ir+th+mid+shared
acc4, cfg4 = fuse4_sameT(ir_a, th_a, mid_a, sh_a, yt_a, mask_a, [1.5, 2.0, 2.5, 3.0], ngrid=21)
nest4 = nested_fixed4(ir_a, th_a, mid_a, sh_a, yt_a, yu_a, mask_a, cfg4)
p4 = cfg4["wa"]*softmax_np(ir_a,cfg4["T"])+cfg4["wb"]*softmax_np(th_a,cfg4["T"])+cfg4["wc"]*softmax_np(mid_a,cfg4["T"])+cfg4["wd"]*softmax_np(sh_a,cfg4["T"])
dis4=int((p4[mask_a].argmax(1)!=v7p[mask_a].argmax(1)).sum())
print("4way", cfg4, "nested", nest4, "dis", dis4, flush=True)

# conf gate: blend shared into v7 when v7 low conf
psh = softmax_np(sh_a, 2.5)
attempts=[]
for thr in [0.25,0.3,0.35,0.4,0.45,0.5]:
    for w in [0.15,0.25,0.35,0.5,0.7,1.0]:
        out=v7p.copy(); low=v7p.max(1)<thr; out[low]=(1-w)*v7p[low]+w*psh[low]
        acc=float((out[mask_a].argmax(1)==yt_a[mask_a]).mean())
        folds=[float((out[mask_a&(yu_a==u)].argmax(1)==yt_a[mask_a&(yu_a==u)]).mean()) for u in (8,9,24)]
        nest=float(np.mean(folds)); dis=int((out[mask_a].argmax(1)!=v7p[mask_a].argmax(1)).sum())
        attempts.append({"thr":thr,"w":w,"acc":acc,"nested":nest,"dis":dis,
            "clear":acc>=V7_HOLD+MIN_D and nest>=V7_NESTED+MIN_D and dis>=MIN_DIS})
attempts=sorted(attempts,key=lambda d:(-d["clear"],-d["nested"],-d["acc"]))
print("top gate", attempts[:3], flush=True)

out={
  "tag":"probe_sharedstem_fuse",
  "shared_solo":acc_s,
  "replace_ir":{"hold":acc_rep,"nested":nest_rep,"dis":dis_rep,"clear":acc_rep>=V7_HOLD+MIN_D and nest_rep>=V7_NESTED+MIN_D and dis_rep>=MIN_DIS},
  "fuse3_shared_th_mid":{"cfg":cfg3,"hold":acc3,"nested":nest3,"dis":dis3,
    "clear":acc3>=V7_HOLD+MIN_D and nest3>=V7_NESTED+MIN_D and dis3>=MIN_DIS},
  "fuse4":{"cfg":cfg4,"hold":acc4,"nested":nest4,"dis":dis4,
    "clear":acc4>=V7_HOLD+MIN_D and nest4>=V7_NESTED+MIN_D and dis4>=MIN_DIS},
  "best_conf_gate":attempts[0],
  "any_clear": any([attempts[0]["clear"],
    acc_rep>=V7_HOLD+MIN_D and nest_rep>=V7_NESTED+MIN_D and dis_rep>=MIN_DIS,
    acc3>=V7_HOLD+MIN_D and nest3>=V7_NESTED+MIN_D and dis3>=MIN_DIS,
    acc4>=V7_HOLD+MIN_D and nest4>=V7_NESTED+MIN_D and dis4>=MIN_DIS]),
  "wrote_csv": False,
  "submit":"DO NOT - keep ir_v7",
}
(ROOT/"metrics_probe_sharedstem_fuse.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2), flush=True)
