"""Dump MidFusePlus OOF logits from existing fold checkpoints; merge later."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
sys.path.insert(0, str(V2)); sys.path.insert(0, str(ROOT))
from dataset import CachedDualDataset, load_skel_train_cache, DEFAULT_HOLD_OUT_USERS
from model import MidFusePlus

@torch.no_grad()
def predict(model, ds, device, bs=48):
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    outs = []
    model.eval()
    for xs, xi, y, _u, flag in loader:
        outs.append(model(xs.to(device), xi.to(device), flag.to(device)).float().cpu().numpy())
    return np.concatenate(outs, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints_mfp"))
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--out", default=str(ROOT / "oof_mfp_partial.npz"))
    ap.add_argument("--cache-dir", default=str(V2 / "cache"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_skel, y, users, _ = load_skel_train_cache(Path(args.cache_dir))
    imu = np.load(Path(args.cache_dir) / "imu_train.npz")
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    oof = np.zeros((len(y), 40), np.float32)
    fold_set = {int(x) for x in args.folds.split(",") if x.strip() != ""}
    gkf = GroupKFold(n_splits=5)
    for fi, (tr, va) in enumerate(gkf.split(np.arange(len(y)), y, users)):
        if fi not in fold_set:
            continue
        ck = Path(args.ckpt_dir) / f"best_fold{fi}.pt"
        assert ck.exists(), ck
        state = torch.load(ck, map_location=device, weights_only=False)
        m = MidFusePlus(num_classes=40, use_velocity=state.get("use_velocity", True)).to(device)
        m.load_state_dict(state["model_state"]); m.eval()
        ds = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False)
        oof[va] = predict(m, ds, device)
        print(f"fold{fi} val_acc_ckpt={state.get('val_acc')} oof_acc={float((oof[va].argmax(1)==y[va]).mean()):.4f}", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(args.out, midfuse_plus=oof, y=y.astype(np.int64), users=users.astype(np.int64), folds=sorted(fold_set))
    print("wrote", args.out)

if __name__ == "__main__":
    main()
