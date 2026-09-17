"""Holdout probe: TinyTempCNN on Depth+IR, optional late-fuse with MidFuse probs.

Gate: only recommend full CV if fused holdout clearly beats MidFuse 0.537.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(r"D:\CUHK-X\Small-Model-Track\baselines\v5")
ROOT_V2 = Path(r"D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT_V2))

from tiny_vision import TinyTempCNN, count_parameters, ckpt_mb
from dataset import CachedDualDataset, load_skel_train_cache
from model import build_model as build_midfuse


class VisionDS(Dataset):
    def __init__(self, X, y, users, indices=None, augment=False, seed=42):
        self.X = X
        self.y = y
        self.users = users
        self.indices = np.arange(len(y)) if indices is None else np.asarray(indices)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.X[idx].astype(np.float32) / 255.0  # (2,T,H,W)
        if self.augment:
            if self.rng.rand() < 0.5:
                x = x + self.rng.randn(*x.shape).astype(np.float32) * 0.02
            if self.rng.rand() < 0.5:
                # temporal roll
                shift = self.rng.randint(-2, 3)
                x = np.roll(x, shift, axis=1)
            if self.rng.rand() < 0.3:
                x = x[:, :, :, ::-1].copy()  # horizontal flip
        return torch.from_numpy(x), int(self.y[idx]), int(self.users[idx])

    @property
    def labels(self):
        return np.asarray(self.y[self.indices], dtype=np.int64)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_correct = total_n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _u in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += bs
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def predict_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y, _u in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def midfuse_holdout_probs(device):
    """Reload MidFuse holdout ckpt and get probs on holdout indices aligned to vision_holdout orig_idx."""
    cache = ROOT_V2 / "cache"
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y, dtype=np.int64)
    users = np.asarray(users, dtype=np.int64)

    hold = np.load(ROOT / "cache" / "vision_holdout.npz", allow_pickle=False)
    orig = hold["orig_idx"]

    ckpt_path = ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_midfuse(ckpt.get("model_name", "midfuse"), num_classes=ckpt.get("num_classes", 40))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ds = CachedDualDataset(X_skel, X_imu, y, users, orig, has_imu, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for xs, xi, yy, _u, flag in loader:
            xs = xs.to(device)
            xi = xi.to(device)
            flag = flag.to(device)
            logits = model(xs, xi, flag)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs), y[orig]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    if args.cpu or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device", device, flush=True)

    tr = np.load(ROOT / "cache" / "vision_trainusers.npz", allow_pickle=False)
    ho = np.load(ROOT / "cache" / "vision_holdout.npz", allow_pickle=False)
    Xtr, ytr, utr = tr["X"], tr["y"], tr["users"]
    Xho, yho, uho = ho["X"], ho["y"], ho["users"]
    print(f"train_users {Xtr.shape} holdout {Xho.shape}", flush=True)

    train_ds = VisionDS(Xtr, ytr, utr, augment=True, seed=args.seed)
    val_ds = VisionDS(Xho, yho, uho, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # class weights from train
    counts = np.bincount(ytr, minlength=40).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (40 * counts)
    w = np.clip(w, 0.25, 8.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=0.05)

    model = TinyTempCNN(in_ch=2, num_classes=40, width=args.width).to(device)
    nparams = count_parameters(model)
    print(f"params={nparams} ~{ckpt_mb(model):.2f}MB", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = -1.0
    best_state = None
    bad = 0
    history = []
    t0 = time.time()
    ckpt_dir = ROOT / "checkpoints_vision"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, opt, device, False)
        sched.step()
        history.append({"ep": ep, "tr_loss": tr_loss, "tr_acc": tr_acc, "va_loss": va_loss, "va_acc": va_acc})
        print(f"ep{ep:02d} tr={tr_acc:.3f}/{tr_loss:.3f} va={va_acc:.3f}/{va_loss:.3f}", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save({"model_state": best_state, "val_acc": best_acc, "epoch": ep, "width": args.width}, ckpt_dir / "best_holdout.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break

    model.load_state_dict(best_state)
    vis_probs, ys = predict_probs(model, val_loader, device)
    vis_acc = float((vis_probs.argmax(1) == ys).mean())
    print("vision holdout best", vis_acc, flush=True)

    # MidFuse probs + fusion sweep
    mf_device = torch.device("cpu")  # keep GPU free / avoid OOM with DAM4SAM
    mf_probs, mf_y = midfuse_holdout_probs(mf_device)
    assert np.array_equal(ys, mf_y)
    mf_acc = float((mf_probs.argmax(1) == ys).mean())
    print("midfuse holdout", mf_acc, flush=True)

    fuse_results = []
    best_fuse = {"acc": -1}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # alpha = weight on MidFuse
        blend = alpha * mf_probs + (1 - alpha) * vis_probs
        acc = float((blend.argmax(1) == ys).mean())
        row = {"alpha_midfuse": alpha, "acc": acc}
        fuse_results.append(row)
        if acc > best_fuse["acc"]:
            best_fuse = row
        print(f"fuse alpha_mf={alpha:.1f} acc={acc:.4f}", flush=True)

    midfuse_baseline = 0.5366336633663367
    delta = best_fuse["acc"] - midfuse_baseline
    clearly_helps = bool(best_fuse["acc"] > midfuse_baseline + 0.005)  # +0.5pp gate
    gate = {
        "vision_alone_holdout": vis_acc,
        "midfuse_holdout": mf_acc,
        "best_fuse": best_fuse,
        "delta_vs_0.537": delta,
        "clearly_helps_gate_+0.5pp": clearly_helps,
        "recommend_full_cv": clearly_helps,
        "n_params": nparams,
        "ckpt_mb_est": ckpt_mb(model),
        "epochs_ran": len(history),
        "elapsed_sec": time.time() - t0,
        "fuse_curve": fuse_results,
        "history": history,
        "device": str(device),
    }
    out = ROOT / "vision_holdout_probe.json"
    out.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({k: gate[k] for k in gate if k not in ("history", "fuse_curve")}, indent=2), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
