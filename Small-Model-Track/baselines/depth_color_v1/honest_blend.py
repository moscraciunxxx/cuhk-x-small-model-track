"""Build honest MidFuse OOF + holdout logits; softmax-blend with Depth video on honest holdout; write fused CSV."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V2 = TRACK / "baselines" / "skeleton_imu_v2"
DC = TRACK / "baselines" / "depth_color_v1"
HOLD = {8, 9, 24}

sys.path.insert(0, str(V2))
import model as v2m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = V2 / "cache"
skel_tr = np.load(cache / "skel_train.npz", allow_pickle=True)
imu_tr = np.load(cache / "imu_train.npz", allow_pickle=True)
Xs, Xi, has = skel_tr["X"], imu_tr["X"], imu_tr["has_imu"]
meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
y = np.array([m["label"] for m in meta], dtype=np.int64)
users = np.array([m["user_id"] for m in meta], dtype=np.int64)
assert len(y) == len(Xs) == 2931

def load_mid(ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = v2m.build_model(blob.get("model_name", "midfuse"), num_classes=int(blob.get("num_classes", 40)))
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model

@torch.no_grad()
def predict(model, idxs, bs=64):
    out = np.zeros((len(idxs), 40), dtype=np.float32)
    for i in range(0, len(idxs), bs):
        ii = idxs[i:i+bs]
        xs = torch.from_numpy(Xs[ii]).float().to(device)
        xi = torch.from_numpy(Xi[ii]).float().to(device)
        f = torch.from_numpy(has[ii].astype(np.float32)).to(device)
        out[i:i+len(ii)] = model(xs, xi, f).cpu().numpy()
    return out

# Honest OOF on pool via fold ckpts
ckpt_dir = V2 / "checkpoints_midfuse_s123"
hold_idx = np.where(np.isin(users, list(HOLD)))[0]
pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
oof = np.zeros((len(y), 40), dtype=np.float32)
gkf = GroupKFold(n_splits=5)
for fold, (tr, va) in enumerate(gkf.split(pool_idx, y[pool_idx], groups=users[pool_idx])):
    ck = ckpt_dir / f"best_fold{fold}.pt"
    if not ck.exists():
        ck = V2 / "checkpoints" / f"best_fold{fold}.pt"
    print("fold", fold, ck, "va", len(va))
    model = load_mid(ck)
    oof[pool_idx[va]] = predict(model, pool_idx[va])
    del model
    torch.cuda.empty_cache()

# Honest holdout logits from best_holdout.pt
ckh = ckpt_dir / "best_holdout.pt"
if not ckh.exists():
    ckh = V2 / "checkpoints" / "best_holdout.pt"
model = load_mid(ckh)
oof[hold_idx] = predict(model, hold_idx)
del model
torch.cuda.empty_cache()

mid_acc_hold = float((oof[hold_idx].argmax(1) == y[hold_idx]).mean())
mid_acc_oof = float((oof[pool_idx].argmax(1) == y[pool_idx]).mean())
print(f"HONEST MidFuse holdout_acc={mid_acc_hold:.4f} pool_oof_acc={mid_acc_oof:.4f}")
np.save(DC / "cache" / "midfuse_train_logits_honest.npy", oof)

# Depth video holdout logits from trained cnn_gru
import sys as _sys
_sys.path.insert(0, str(DC))
# reload local
for m in list(sys.modules):
    if m in ("dataset", "model"):
        del sys.modules[m]
sys.path.insert(0, str(DC))
from model import build_model
from dataset import CachedClipDataset
from torch.utils.data import DataLoader

y_dc = np.load(DC / "cache" / "depth_color" / "train_y.npy")
u_dc = np.load(DC / "cache" / "depth_color" / "train_users.npy")
X = np.memmap(DC / "cache" / "depth_color" / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y_dc),16,112,112,3))
assert np.all(y_dc == y) and np.all(u_dc == users)

blob = torch.load(DC / "checkpoints" / "depth_color" / "holdout_train.pt", map_location="cpu", weights_only=False)
vid = build_model(40).to(device)
vid.load_state_dict(blob["model"])
vid.eval()
loader = DataLoader(CachedClipDataset(X, y_dc, u_dc, hold_idx, train=False), batch_size=32, shuffle=False, num_workers=0)
vid_logits = np.zeros((len(hold_idx), 40), dtype=np.float32)
pos = 0
with torch.no_grad():
    for xb, yb, uu, ii in loader:
        out = vid(xb.to(device)).cpu().numpy()
        vid_logits[pos:pos+len(out)] = out
        pos += len(out)
vid_acc = float((vid_logits.argmax(1) == y[hold_idx]).mean())
print(f"Depth CNN-GRU holdout_acc={vid_acc:.4f} (ckpt val_f1={blob.get('val_f1')})")

mid_h = oof[hold_idx]
# tune temperature softmax blend weight
def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

best = (-1, None)
for T in [0.5, 1.0, 1.5, 2.0]:
    pm, pv = softmax(mid_h, T), softmax(vid_logits, T)
    for w in np.linspace(0, 1, 21):
        pred = (w * pv + (1 - w) * pm).argmax(1)
        acc = float((pred == y[hold_idx]).mean())
        if acc > best[0]:
            best = (acc, {"w": float(w), "T": float(T)})
print("best blend holdout", best)

# Also MidFuse-only and video-only already printed
# Build TEST fused submission with best w,T
mid_test = np.load(DC / "cache" / "midfuse_test_logits.npy")
vid_test = np.load(DC / "checkpoints" / "depth_color" / "test_logits.npy")
w, T = best[1]["w"], best[1]["T"]
ptest = (w * softmax(vid_test, T) + (1 - w) * softmax(mid_test, T)).argmax(1)
test_meta = json.loads((DC / "cache" / "depth_color" / "test_meta.json").read_text(encoding="utf-8"))
out = DC / "submission_blend_depth_midfuse.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["path", "prediction"])
    for meta, pred in zip(test_meta, fused):
        path = meta["path"] if meta["path"].endswith("/") else meta["path"] + "/"
        wr.writerow([path, int(pred)])
print("wrote", out, "w", w, "T", T, "holdout_acc", best[0])

# Compare to midfuse ensemble CSV agreement
ens = {}
with (V2 / "submission_skeleton_imu_v2_ensemble.csv").open() as f:
    r = csv.DictReader(f)
    for row in r:
        ens[row["path"].rstrip("/") + "/"] = int(row["prediction"])
agree = sum(1 for meta, pred in zip(test_meta, fused) if ens.get(meta["path"] if meta["path"].endswith("/") else meta["path"]+"/") == pred)
print(f"agree with midfuse ensemble {agree}/405")

report = {
    "honest_midfuse_holdout_acc": mid_acc_hold,
    "honest_midfuse_pool_oof_acc": mid_acc_oof,
    "depth_cnn_gru_holdout_acc": vid_acc,
    "best_blend": best[1],
    "best_blend_holdout_acc": best[0],
}
(DC / "logs" / "honest_blend_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
