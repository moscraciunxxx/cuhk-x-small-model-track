"""Multi-seed MidFuse bag (reuse v2 MidFuseNet) + optional holdout probe."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# reuse v2 training pieces via path
V2 = Path(__file__).resolve().parent.parent / "skeleton_imu_v2"
sys.path.insert(0, str(V2))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_skel_train_cache,
)
from model import build_model, count_parameters  # noqa: E402  (v2 model)

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights_from_labels(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = np.clip(counts.sum() / (num_classes * counts), 0.25, 8.0)
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = total_correct = total_n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xs, xi, y, _u, flag in loader:
            xs, xi, y, flag = xs.to(device), xi.to(device), y.to(device), flag.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xs, xi, flag)
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


def train_one(train_ds, val_ds, args, device, tag, seed):
    set_seed(seed)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    pin_memory=device.type == "cuda")
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    pin_memory=device.type == "cuda")
    model = build_model("midfuse", num_classes=40).to(device)
    n_params = count_parameters(model)
    cw = class_weights_from_labels(train_ds.labels, 40).to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1.0
    best_path = Path(args.ckpt_dir) / f"best_{tag}_seed{seed}.pt"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    patience = args.patience
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_l, tr_a = run_epoch(model, tl, crit, opt, device, True)
        va_l, va_a = run_epoch(model, vl, crit, opt, device, False)
        sch.step()
        hist.append({"epoch": ep, "train_acc": tr_a, "val_acc": va_a})
        print(f"[{tag} seed={seed}] ep {ep}/{args.epochs} train={tr_a:.4f} val={va_a:.4f}", flush=True)
        if va_a >= best:
            best = va_a
            patience = args.patience
            torch.save({
                "model_state": model.state_dict(),
                "model_name": "midfuse",
                "num_classes": 40,
                "val_acc": best,
                "epoch": ep,
                "n_params": n_params,
                "dual": True,
                "seed": seed,
                "tag": tag,
            }, best_path)
        else:
            patience -= 1
            if patience <= 0:
                print(f"[{tag} seed={seed}] early stop @ {ep}", flush=True)
                break
    return {"tag": tag, "seed": seed, "best_val_acc": float(best), "ckpt": str(best_path),
            "n_params": n_params, "elapsed_sec": time.time() - t0, "history": hist}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints_seeds"))
    p.add_argument("--metrics-out", default=str(ROOT / "metrics_seeds.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--mode", choices=["holdout", "full"], default="full",
                   help="holdout=train on non-holdout; full=all-data (monitor holdout)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seeds={args.seeds}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"free={free/1e9:.2f}G", flush=True)

    cache = Path(args.cache_dir)
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz")
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_h = np.where(~np.isin(users, list(hold)))[0]
    va_h = np.where(np.isin(users, list(hold)))[0]
    all_idx = np.arange(len(y))

    results = []
    for seed in args.seeds:
        if args.mode == "holdout":
            train_idx, tag = tr_h, "holdout"
        else:
            train_idx, tag = all_idx, "all_train"
        train_ds = CachedDualDataset(X_skel, X_imu, y, users, train_idx, has_imu, augment=True, seed=seed)
        val_ds = CachedDualDataset(X_skel, X_imu, y, users, va_h, has_imu, augment=False, seed=seed)
        results.append(train_one(train_ds, val_ds, args, device, tag, seed))
        # also copy seed42 all as best.pt convenience
        if seed == args.seeds[0]:
            shutil.copy2(results[-1]["ckpt"], Path(args.ckpt_dir) / "best.pt")

    metrics = {
        "mode": args.mode,
        "seeds": args.seeds,
        "results": results,
        "mean_monitor_val": float(np.mean([r["best_val_acc"] for r in results])),
        "finished_at_unix": time.time(),
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("mode", "seeds", "mean_monitor_val")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
