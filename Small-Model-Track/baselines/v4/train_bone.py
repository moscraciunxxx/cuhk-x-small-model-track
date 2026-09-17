"""Train BoneMidFuse / TripleFuse with GroupKFold by user (v4)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    DEFAULT_T,
    load_skel_train_cache,
)
from bones import bone_features, bone_dim, precompute_bone_cache
from model import build_model, count_parameters

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "cache"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    w = np.clip(w, 0.25, 8.0)
    return torch.tensor(w, dtype=torch.float32)


class BoneDualDataset(Dataset):
    def __init__(self, X_bone, X_imu, y, users, indices=None, has_imu=None,
                 X_skel=None, augment=False, seed=42):
        if indices is None:
            indices = np.arange(len(y))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.X_bone = X_bone
        self.X_imu = X_imu
        self.X_skel = X_skel
        self.y = y
        self.users = users
        self.has_imu = has_imu if has_imu is not None else np.ones(len(y), dtype=np.bool_)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        xb = np.asarray(self.X_bone[idx], dtype=np.float32).copy()
        xi = np.asarray(self.X_imu[idx], dtype=np.float32).copy()
        xs = None
        if self.X_skel is not None:
            xs = np.asarray(self.X_skel[idx], dtype=np.float32).copy()
        if self.augment:
            if self.rng.rand() < 0.5:
                xb += self.rng.randn(*xb.shape).astype(np.float32) * 0.02
                xi += self.rng.randn(*xi.shape).astype(np.float32) * 0.02
                if xs is not None:
                    xs += self.rng.randn(*xs.shape).astype(np.float32) * 0.02
            if self.rng.rand() < 0.5:
                shift = self.rng.randint(-4, 5)
                xb = np.roll(xb, shift, axis=0)
                xi = np.roll(xi, shift, axis=0)
                if xs is not None:
                    xs = np.roll(xs, shift, axis=0)
            if self.rng.rand() < 0.3:
                t0 = self.rng.randint(0, xb.shape[0])
                w = self.rng.randint(1, max(2, xb.shape[0] // 8))
                xb[t0:t0 + w] = 0
                xi[t0:t0 + w] = 0
                if xs is not None:
                    xs[t0:t0 + w] = 0
        y = int(self.y[idx])
        flag = float(self.has_imu[idx])
        if xs is not None:
            return (torch.from_numpy(xs), torch.from_numpy(xb), torch.from_numpy(xi), y, flag)
        return (torch.from_numpy(xb), torch.from_numpy(xi), y, flag)

    @property
    def labels(self):
        return np.asarray(self.y[self.indices], dtype=np.int64)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, triple: bool):
    model.train(train)
    total_loss = total_correct = total_n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if triple:
                xs, xb, xi, y, flag = batch
                xs = xs.to(device, non_blocking=True)
                xb = xb.to(device, non_blocking=True)
                xi = xi.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                flag = flag.to(device, non_blocking=True)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(xs, xb, xi, flag)
            else:
                xb, xi, y, flag = batch
                xb = xb.to(device, non_blocking=True)
                xi = xi.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                flag = flag.to(device, non_blocking=True)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(xb, xi, flag)
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


def train_one(train_ds, val_ds, args, device, tag, triple, val_users=None):
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    model = build_model(args.model, num_classes=args.num_classes).to(device)
    n_params = count_parameters(model)
    print(f"[{tag}] params={n_params} model={args.model} bone_dim={bone_dim()}", flush=True)

    cw = class_weights_from_labels(train_ds.labels, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_val = -1.0
    best_path = Path(args.ckpt_dir) / f"best_{tag}.pt"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    patience_left = args.patience
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True, triple)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False, triple)
        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        })
        print(
            f"[{tag}] epoch {epoch}/{args.epochs} train_acc={tr_acc:.4f} "
            f"val_acc={va_acc:.4f} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}",
            flush=True,
        )
        if va_acc >= best_val:
            best_val = va_acc
            patience_left = args.patience
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "num_classes": args.num_classes,
                "T": args.T,
                "val_acc": best_val,
                "epoch": epoch,
                "n_params": n_params,
                "tag": tag,
                "triple": triple,
                "bone_dim": bone_dim(),
                "hold_out_users": list(val_users or []),
            }, best_path)
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break

    return {
        "tag": tag,
        "best_val_acc": float(best_val),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_params": n_params,
        "ckpt": str(best_path),
        "history": history,
        "elapsed_sec": time.time() - t0,
        "val_users": list(val_users or []),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    p.add_argument("--ckpt-dir", type=str, default=str(ROOT / "checkpoints_bone"))
    p.add_argument("--metrics-out", type=str, default=str(ROOT / "metrics_bone.json"))
    p.add_argument("--model", type=str, default="bonemidfuse",
                   choices=["bonemidfuse", "triple"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--hold-out-users", type=int, nargs="+", default=list(DEFAULT_HOLD_OUT_USERS))
    p.add_argument("--mode", type=str, default="cv", choices=["holdout", "cv", "smoke"])
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"gpu={torch.cuda.get_device_name(0)} free={free/1e9:.2f}G", flush=True)

    cache = Path(args.cache_dir)
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)

    bone_path = cache / "bone_train.npz"
    if bone_path.exists():
        X_bone = np.load(bone_path)["X"]
        print(f"loaded bone cache {X_bone.shape}", flush=True)
    else:
        print("precomputing bone features...", flush=True)
        X_bone = precompute_bone_cache(X_skel)
        np.savez_compressed(bone_path, X=X_bone)
        print(f"wrote {bone_path} shape={X_bone.shape}", flush=True)

    triple = args.model == "triple"
    augment = not args.no_augment

    def make_ds(indices, train_mode):
        return BoneDualDataset(
            X_bone, X_imu, y, users, indices, has_imu,
            X_skel=X_skel if triple else None,
            augment=augment and train_mode, seed=args.seed,
        )

    all_idx = np.arange(len(y))
    hold = set(args.hold_out_users)
    tr_h = np.where(~np.isin(users, list(hold)))[0]
    va_h = np.where(np.isin(users, list(hold)))[0]

    metrics = {
        "mode": args.mode, "seed": args.seed, "model": args.model,
        "bone_dim": bone_dim(), "device": str(device),
    }

    if args.mode == "smoke":
        args.epochs = 2
        args.patience = 0
        r = train_one(make_ds(tr_h, True), make_ds(va_h, False), args, device, "smoke", triple, args.hold_out_users)
        metrics["folds"] = [r]
        metrics["mean_val_acc"] = r["best_val_acc"]

    elif args.mode == "holdout":
        r = train_one(make_ds(tr_h, True), make_ds(va_h, False), args, device, "holdout", triple, args.hold_out_users)
        metrics["folds"] = [r]
        metrics["mean_val_acc"] = r["best_val_acc"]
        metrics["holdout"] = r
        shutil.copy2(r["ckpt"], Path(args.ckpt_dir) / "best.pt")

    else:  # cv
        gkf = GroupKFold(n_splits=args.cv_splits)
        fold_results = []
        for fi, (tr, va) in enumerate(gkf.split(all_idx, y, users)):
            vu = sorted(set(int(users[i]) for i in va))
            print(f"=== fold {fi} n_train={len(tr)} n_val={len(va)} val_users={vu} ===", flush=True)
            fold_results.append(
                train_one(make_ds(tr, True), make_ds(va, False), args, device, f"fold{fi}", triple, vu)
            )
        accs = [r["best_val_acc"] for r in fold_results]
        metrics["folds"] = fold_results
        metrics["mean_val_acc"] = float(np.mean(accs))
        metrics["std_val_acc"] = float(np.std(accs))
        print(f"CV mean+/-std = {metrics['mean_val_acc']:.4f} +/- {metrics['std_val_acc']:.4f}", flush=True)

        hold = train_one(make_ds(tr_h, True), make_ds(va_h, False), args, device, "holdout", triple, args.hold_out_users)
        metrics["holdout"] = hold

        all_r = train_one(make_ds(all_idx, True), make_ds(va_h, False), args, device, "all_train", triple, args.hold_out_users)
        metrics["all_train"] = all_r
        shutil.copy2(all_r["ckpt"], Path(args.ckpt_dir) / "best.pt")
        metrics["primary_ckpt"] = str(Path(args.ckpt_dir) / "best.pt")

    metrics["finished_at_unix"] = time.time()
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.metrics_out}", flush=True)
    print(f"mean_val_acc={metrics.get('mean_val_acc')} holdout={metrics.get('holdout',{}).get('best_val_acc')}", flush=True)


if __name__ == "__main__":
    main()
