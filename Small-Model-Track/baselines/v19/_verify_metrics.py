import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, "baselines/v18")
sys.path.insert(0, "baselines/skeleton_imu_v2")
from v18_fuse_kd import softmax, acc, nested_family, hold_family, final_cfg
from dataset import DEFAULT_HOLD_OUT_USERS, load_skel_train_cache

keys = ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]
X, y, users, _ = load_skel_train_cache(Path("baselines/skeleton_imu_v2/cache"))
y = np.asarray(y)
users = np.asarray(users)
hold = set(DEFAULT_HOLD_OUT_USERS)
nh_idx = np.where(np.array([int(u) not in hold for u in users]))[0]
P, H = {}, {}
yh = None
for k in keys:
    o = np.load(f"baselines/v18/oof_{k}.npz")
    h = np.load(f"baselines/v18/holdout_{k}.npz")
    key = [x for x in o.files if x not in ("y", "users")][0]
    P[k] = softmax(o[key][nh_idx])
    H[k] = softmax(h[key])
    yh = h["y"]
yt = y[nh_idx]
us = users[nh_idx]
print("conf4 nested", acc(nested_family(keys, P, yt, us, "conf").argmax(1), yt))
print("conf4 hold", acc(hold_family(keys, P, yt, H, "conf").argmax(1), yh))
print("conf4 cfg", final_cfg(keys, P, yt, "conf"))
keys2 = ["kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]
P2 = {k: P[k] for k in keys2}
H2 = {k: H[k] for k in keys2}
for fam in ["eq", "pow", "conf"]:
    print(fam, "nested", round(acc(nested_family(keys2, P2, yt, us, fam).argmax(1), yt), 4),
          "hold", round(acc(hold_family(keys2, P2, yt, H2, fam).argmax(1), yh), 4),
          "cfg", final_cfg(keys2, P2, yt, fam))
