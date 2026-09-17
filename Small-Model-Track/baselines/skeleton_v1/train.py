"""Train skeleton HAR baseline with cross-subject hold-out / GroupKFold."""
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
from torch.utils.data import DataLoader

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    DEFAULT_T,
    CachedSkeletonDataset,
    SkeletonSequenceDataset,
    discover_train_samples,
    group_kfold_indices,
    load_train_cache,
    split_by_users,
)
from model import build_model, count_parameters

ROOT = Path(__file__).resolve().parent
DEFAULT_SKELETON = Path(
    r"D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\Skeleton"
)
DEFAULT_CKPT_DIR = ROOT / "checkpoints"
DEFAULT_METRICS = ROOT / "metrics.json"
DEFAULT_CACHE = ROOT / "cache"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _user in loader:
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


def make_loaders(train_ds, val_ds, args, device):
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def train_one(train_ds, val_ds, args, device, tag: str, val_users=None, train_users=None):
    train_loader, val_loader = make_loaders(train_ds, val_ds, args, device)
    model = build_model(args.model, num_classes=args.num_classes).to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    history = []
    best_val = -1.0
    best_path = Path(args.ckpt_dir) / f"best_{tag}.pt"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[{tag}] epoch {epoch}/{args.epochs}  "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f}",
            flush=True,
        )
        if va_acc >= best_val:
            best_val = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "num_classes": args.num_classes,
                    "T": args.T,
                    "val_acc": best_val,
                    "epoch": epoch,
                    "n_params": n_params,
                    "tag": tag,
                    "hold_out_users": list(val_users or []),
                },
                best_path,
            )
    elapsed = time.time() - t0
    return {
        "tag": tag,
        "best_val_acc": best_val,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_params": n_params,
        "ckpt": str(best_path),
        "history": history,
        "elapsed_sec": elapsed,
        "val_users": list(val_users or []),
        "train_users": list(train_users or []),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Skeleton HAR baseline train")
    p.add_argument("--skeleton-root", type=str, default=str(DEFAULT_SKELETON))
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    p.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    p.add_argument("--metrics-out", type=str, default=str(DEFAULT_METRICS))
    p.add_argument("--model", type=str, default="conv1d",
                   choices=["conv1d", "gru", "transformer"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--hold-out-users",
        type=int,
        nargs="+",
        default=list(DEFAULT_HOLD_OUT_USERS),
    )
    p.add_argument(
        "--mode",
        type=str,
        default="holdout",
        choices=["holdout", "cv", "smoke", "full"],
    )
    p.add_argument("--cv-splits", type=int, default=3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def indices_by_users(users_arr, hold_out):
    hold = set(int(u) for u in hold_out)
    train_idx = np.where(~np.isin(users_arr, list(hold)))[0]
    val_idx = np.where(np.isin(users_arr, list(hold)))[0]
    return train_idx, val_idx


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    cache_npz = Path(args.cache_dir) / "train.npz"
    use_cache = (not args.no_cache) and cache_npz.exists()

    if use_cache:
        print(f"Using cache {cache_npz}", flush=True)
        X, y, users, meta = load_train_cache(Path(args.cache_dir))
        if args.max_samples and args.max_samples > 0:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(y), size=min(args.max_samples, len(y)), replace=False)
            idx = np.sort(idx)
            X, y, users = X[idx], y[idx], users[idx]
            meta = [meta[i] for i in idx]
        print(f"cached trials={len(y)} users={sorted(set(int(u) for u in users))}", flush=True)

        def ds_from_idx(indices):
            return CachedSkeletonDataset(X, y, users, indices)

        def split_holdout():
            tr, va = indices_by_users(users, args.hold_out_users)
            return tr, va

        all_idx = np.arange(len(y))
    else:
        print("No cache — loading JSON on the fly (slow). Run build_cache.py first.", flush=True)
        samples = discover_train_samples(Path(args.skeleton_root))
        print(f"discovered {len(samples)} trials", flush=True)
        if args.max_samples and args.max_samples > 0:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(samples), size=min(args.max_samples, len(samples)), replace=False)
            samples = [samples[i] for i in sorted(idx)]
        users_list = [s["user_id"] for s in samples]
        print(f"users ({len(set(users_list))}): {sorted(set(users_list))}", flush=True)
        X = y = users = meta = None
        all_idx = np.arange(len(samples))

        def ds_from_idx(indices):
            subset = [samples[i] for i in indices]
            return SkeletonSequenceDataset(subset, T=args.T, normalize=True)

        def split_holdout():
            hold = set(args.hold_out_users)
            tr = np.array([i for i, s in enumerate(samples) if s["user_id"] not in hold])
            va = np.array([i for i, s in enumerate(samples) if s["user_id"] in hold])
            return tr, va

    if args.mode == "smoke":
        args.epochs = min(args.epochs, 3)
        tr, va = split_holdout()
        if len(va) == 0:
            cut = max(1, int(0.9 * len(all_idx)))
            tr, va = all_idx[:cut], all_idx[cut:]
        train_users = sorted(set(int(users[i]) for i in tr)) if use_cache else None
        val_users = sorted(set(int(users[i]) for i in va)) if use_cache else list(args.hold_out_users)
        result = train_one(
            ds_from_idx(tr), ds_from_idx(va), args, device, tag="smoke",
            val_users=val_users, train_users=train_users,
        )
        metrics = {
            "mode": "smoke",
            "seed": args.seed,
            "model": args.model,
            "T": args.T,
            "device": str(device),
            "used_cache": use_cache,
            "folds": [result],
            "mean_val_acc": result["best_val_acc"],
        }
    elif args.mode in ("holdout", "full"):
        tr, va = split_holdout()
        print(f"holdout users={args.hold_out_users} n_train={len(tr)} n_val={len(va)}", flush=True)
        train_users = sorted(set(int(users[i]) for i in tr)) if use_cache else None
        val_users = sorted(set(int(users[i]) for i in va)) if use_cache else list(args.hold_out_users)
        result = train_one(
            ds_from_idx(tr), ds_from_idx(va), args, device, tag="holdout",
            val_users=val_users, train_users=train_users,
        )
        metrics = {
            "mode": args.mode,
            "seed": args.seed,
            "model": args.model,
            "T": args.T,
            "device": str(device),
            "used_cache": use_cache,
            "hold_out_users": list(args.hold_out_users),
            "folds": [result],
            "mean_val_acc": result["best_val_acc"],
        }
        if args.mode == "full":
            print("Retraining on ALL samples (monitor on hold-out)...", flush=True)
            all_result = train_one(
                ds_from_idx(all_idx), ds_from_idx(va), args, device, tag="all_train",
                val_users=val_users,
                train_users=sorted(set(int(u) for u in (users if use_cache else users_list))),
            )
            metrics["all_train"] = all_result
            src = Path(all_result["ckpt"])
            dst = Path(args.ckpt_dir) / "best.pt"
            if src.exists():
                shutil.copy2(src, dst)
                metrics["primary_ckpt"] = str(dst)
    else:  # cv
        if use_cache:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=args.cv_splits)
            folds_iter = list(gkf.split(all_idx, y, users))
        else:
            folds_iter = group_kfold_indices(samples, n_splits=args.cv_splits, seed=args.seed)
        fold_results = []
        for fi, (tr, va) in enumerate(folds_iter):
            print(f"=== fold {fi} n_train={len(tr)} n_val={len(va)} ===", flush=True)
            vu = sorted(set(int(users[i]) for i in va)) if use_cache else None
            tu = sorted(set(int(users[i]) for i in tr)) if use_cache else None
            fold_results.append(
                train_one(
                    ds_from_idx(tr), ds_from_idx(va), args, device, tag=f"fold{fi}",
                    val_users=vu, train_users=tu,
                )
            )
        mean_acc = float(np.mean([r["best_val_acc"] for r in fold_results]))
        metrics = {
            "mode": "cv",
            "seed": args.seed,
            "model": args.model,
            "T": args.T,
            "device": str(device),
            "used_cache": use_cache,
            "cv_splits": args.cv_splits,
            "folds": fold_results,
            "mean_val_acc": mean_acc,
        }
        tr, va = split_holdout()
        primary = train_one(
            ds_from_idx(tr), ds_from_idx(va), args, device, tag="holdout",
            val_users=list(args.hold_out_users),
        )
        metrics["holdout"] = primary
        shutil.copy2(primary["ckpt"], Path(args.ckpt_dir) / "best.pt")
        metrics["primary_ckpt"] = str(Path(args.ckpt_dir) / "best.pt")

    if "primary_ckpt" not in metrics:
        folds = metrics.get("folds") or []
        if folds:
            src = Path(folds[0]["ckpt"])
            dst = Path(args.ckpt_dir) / "best.pt"
            if src.exists():
                shutil.copy2(src, dst)
                metrics["primary_ckpt"] = str(dst)

    metrics["finished_at_unix"] = time.time()
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics -> {out}", flush=True)
    print(f"mean_val_acc={metrics.get('mean_val_acc')}", flush=True)


if __name__ == "__main__":
    main()
