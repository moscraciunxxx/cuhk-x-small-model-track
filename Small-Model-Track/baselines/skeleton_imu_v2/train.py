"""Train skeleton / skeleton+IMU v2 with GroupKFold by user."""
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    DEFAULT_T,
    CachedDualDataset,
    CachedSkelDataset,
    load_skel_train_cache,
)
from model import build_model, count_parameters

ROOT = Path(__file__).resolve().parent
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


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    # clip extreme weights
    w = np.clip(w, 0.25, 8.0)
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    counts = np.maximum(counts, 1)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool, dual: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if dual:
                xs, xi, y, _u, flag = batch
                xs = xs.to(device, non_blocking=True)
                xi = xi.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                flag = flag.to(device, non_blocking=True)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(xs, xi, flag)
            else:
                x, y, _u = batch
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


def train_one(train_ds, val_ds, args, device, tag: str, dual: bool, val_users=None, train_users=None):
    labels = train_ds.labels
    if args.balanced_sampler:
        sampler = make_sampler(labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
    else:
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

    model = build_model(args.model, num_classes=args.num_classes).to(device)
    n_params = count_parameters(model)
    print(f"[{tag}] params={n_params} model={args.model}", flush=True)

    if args.weighted_ce:
        cw = class_weights_from_labels(labels, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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
    patience_left = args.patience
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, True, dual
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, False, dual
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
            patience_left = args.patience
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
                    "dual": dual,
                    "hold_out_users": list(val_users or []),
                },
                best_path,
            )
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
        "train_users": list(train_users or []),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    p.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    p.add_argument("--metrics-out", type=str, default=str(DEFAULT_METRICS))
    p.add_argument(
        "--model",
        type=str,
        default="midfuse",
        choices=["midfuse", "bone_midfuse", "bone_tcn", "deepconv", "gru_attn", "transformer", "compact", "compact_fuse"],
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--hold-out-users", type=int, nargs="+", default=list(DEFAULT_HOLD_OUT_USERS))
    p.add_argument(
        "--mode",
        type=str,
        default="cv",
        choices=["holdout", "cv", "smoke", "full"],
    )
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--weighted-ce", action="store_true", default=True)
    p.add_argument("--no-weighted-ce", action="store_true")
    p.add_argument("--balanced-sampler", action="store_true", default=True)
    p.add_argument("--no-balanced-sampler", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--skeleton-only", action="store_true")
    return p.parse_args()


def indices_by_users(users_arr, hold_out):
    hold = set(int(u) for u in hold_out)
    train_idx = np.where(~np.isin(users_arr, list(hold)))[0]
    val_idx = np.where(np.isin(users_arr, list(hold)))[0]
    return train_idx, val_idx


def main():
    args = parse_args()
    if args.no_weighted_ce:
        args.weighted_ce = False
    if args.no_balanced_sampler:
        args.balanced_sampler = False
    if args.no_augment:
        args.augment = False

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(
        f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"cuda_mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB", flush=True)

    cache = Path(args.cache_dir)
    X_skel, y, users, meta = load_skel_train_cache(cache)
    print(f"skel cache n={len(y)} shape={X_skel.shape}", flush=True)

    dual = False
    X_imu = None
    has_imu = None
    imu_path = cache / "imu_train.npz"
    dual_models = {"midfuse", "bone_midfuse", "bone_tcn", "compact_fuse"}
    if (not args.skeleton_only) and args.model in dual_models and imu_path.exists():
        imu = np.load(imu_path, allow_pickle=False)
        X_imu = imu["X"]
        has_imu = imu["has_imu"].astype(bool)
        dual = True
        print(f"imu cache n={len(X_imu)} has={has_imu.sum()} model={args.model}", flush=True)
    elif args.model in dual_models and not args.skeleton_only:
        print(f"WARNING: {args.model} requested but no imu_train.npz - falling back to deepconv", flush=True)
        args.model = "deepconv"
        dual = False

    all_idx = np.arange(len(y))

    def make_ds(indices, train_mode: bool):
        aug = bool(args.augment and train_mode)
        if dual:
            return CachedDualDataset(
                X_skel, X_imu, y, users, indices, has_imu, augment=aug, seed=args.seed
            )
        return CachedSkelDataset(X_skel, y, users, indices, augment=aug, seed=args.seed)

    def split_holdout():
        return indices_by_users(users, args.hold_out_users)

    metrics = {
        "mode": args.mode,
        "seed": args.seed,
        "model": args.model,
        "T": args.T,
        "device": str(device),
        "dual": dual,
        "weighted_ce": args.weighted_ce,
        "balanced_sampler": args.balanced_sampler,
        "label_smoothing": args.label_smoothing,
        "augment": args.augment,
    }

    if args.mode == "smoke":
        args.epochs = min(args.epochs, 3)
        args.patience = 0
        tr, va = split_holdout()
        result = train_one(
            make_ds(tr, True),
            make_ds(va, False),
            args,
            device,
            "smoke",
            dual,
            val_users=sorted(set(int(users[i]) for i in va)),
            train_users=sorted(set(int(users[i]) for i in tr)),
        )
        metrics["folds"] = [result]
        metrics["mean_val_acc"] = result["best_val_acc"]
        metrics["std_val_acc"] = 0.0

    elif args.mode == "holdout":
        tr, va = split_holdout()
        print(f"holdout users={args.hold_out_users} n_train={len(tr)} n_val={len(va)}", flush=True)
        result = train_one(
            make_ds(tr, True),
            make_ds(va, False),
            args,
            device,
            "holdout",
            dual,
            val_users=list(args.hold_out_users),
            train_users=sorted(set(int(users[i]) for i in tr)),
        )
        metrics["folds"] = [result]
        metrics["mean_val_acc"] = result["best_val_acc"]
        metrics["std_val_acc"] = 0.0
        shutil.copy2(result["ckpt"], Path(args.ckpt_dir) / "best.pt")
        metrics["primary_ckpt"] = str(Path(args.ckpt_dir) / "best.pt")

    elif args.mode == "cv":
        gkf = GroupKFold(n_splits=args.cv_splits)
        fold_results = []
        for fi, (tr, va) in enumerate(gkf.split(all_idx, y, users)):
            vu = sorted(set(int(users[i]) for i in va))
            tu = sorted(set(int(users[i]) for i in tr))
            print(f"=== fold {fi} n_train={len(tr)} n_val={len(va)} val_users={vu} ===", flush=True)
            fold_results.append(
                train_one(
                    make_ds(tr, True),
                    make_ds(va, False),
                    args,
                    device,
                    f"fold{fi}",
                    dual,
                    val_users=vu,
                    train_users=tu,
                )
            )
        accs = [r["best_val_acc"] for r in fold_results]
        metrics["folds"] = fold_results
        metrics["cv_splits"] = args.cv_splits
        metrics["mean_val_acc"] = float(np.mean(accs))
        metrics["std_val_acc"] = float(np.std(accs))
        print(
            f"CV mean±std = {metrics['mean_val_acc']:.4f} ± {metrics['std_val_acc']:.4f}",
            flush=True,
        )
        # also holdout for apples-to-apples vs v1 0.479
        tr, va = split_holdout()
        hold = train_one(
            make_ds(tr, True),
            make_ds(va, False),
            args,
            device,
            "holdout",
            dual,
            val_users=list(args.hold_out_users),
            train_users=sorted(set(int(users[i]) for i in tr)),
        )
        metrics["holdout"] = hold
        # retrain on ALL for submission
        print("Retraining on ALL samples (monitor hold-out)...", flush=True)
        all_result = train_one(
            make_ds(all_idx, True),
            make_ds(va, False),
            args,
            device,
            "all_train",
            dual,
            val_users=list(args.hold_out_users),
            train_users=sorted(set(int(u) for u in users)),
        )
        metrics["all_train"] = all_result
        shutil.copy2(all_result["ckpt"], Path(args.ckpt_dir) / "best.pt")
        metrics["primary_ckpt"] = str(Path(args.ckpt_dir) / "best.pt")

    else:  # full = holdout + all_train
        tr, va = split_holdout()
        result = train_one(
            make_ds(tr, True),
            make_ds(va, False),
            args,
            device,
            "holdout",
            dual,
            val_users=list(args.hold_out_users),
        )
        metrics["folds"] = [result]
        metrics["mean_val_acc"] = result["best_val_acc"]
        metrics["std_val_acc"] = 0.0
        all_result = train_one(
            make_ds(all_idx, True),
            make_ds(va, False),
            args,
            device,
            "all_train",
            dual,
            val_users=list(args.hold_out_users),
        )
        metrics["all_train"] = all_result
        shutil.copy2(all_result["ckpt"], Path(args.ckpt_dir) / "best.pt")
        metrics["primary_ckpt"] = str(Path(args.ckpt_dir) / "best.pt")

    metrics["finished_at_unix"] = time.time()
    out = Path(args.metrics_out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics -> {out}", flush=True)
    print(
        f"mean_val_acc={metrics.get('mean_val_acc')} ± {metrics.get('std_val_acc')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
