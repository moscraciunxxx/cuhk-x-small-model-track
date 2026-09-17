"""Train MS two-stream ST-GCN with GroupKFold; dump OOF + holdout logits."""
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
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
V7 = ROOT.parent / "v7_stgcn"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V7))
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    DEFAULT_T,
    CachedDualDataset,
    load_skel_train_cache,
)
from model_ms_stgcn import MSTwoStreamSTGCNFuse, count_parameters  # noqa: E402


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


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xs, xi, y, _u, flag in loader:
            xs = xs.to(device, non_blocking=True)
            xi = xi.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            flag = flag.to(device, non_blocking=True)
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


@torch.no_grad()
def predict_logits(model, ds, device, batch_size: int = 20):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    outs = []
    model.eval()
    for xs, xi, y, _u, flag in loader:
        logits = model(
            xs.to(device, non_blocking=True),
            xi.to(device, non_blocking=True),
            flag.to(device, non_blocking=True),
        )
        outs.append(logits.float().cpu().numpy())
    return np.concatenate(outs, 0)


def train_one(train_ds, val_ds, args, device, tag: str, val_users=None, train_users=None):
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

    model = MSTwoStreamSTGCNFuse(num_classes=args.num_classes).to(device)
    n_params = count_parameters(model)
    print(f"[{tag}] params={n_params} model=ms_stgcn_2s", flush=True)

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
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
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
                    "model_name": "ms_stgcn_2s",
                    "num_classes": args.num_classes,
                    "T": args.T,
                    "val_acc": best_val,
                    "epoch": epoch,
                    "n_params": n_params,
                    "tag": tag,
                    "dual": True,
                    "hold_out_users": list(val_users or []),
                },
                best_path,
            )
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break

    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

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
        "model": model,
    }


def indices_by_users(users_arr, hold_out):
    hold = set(int(u) for u in hold_out)
    train_idx = np.where(~np.isin(users_arr, list(hold)))[0]
    val_idx = np.where(np.isin(users_arr, list(hold)))[0]
    return train_idx, val_idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default=str(V2 / "cache"))
    p.add_argument("--ckpt-dir", type=str, default=str(ROOT / "checkpoints_ms2s"))
    p.add_argument("--metrics-out", type=str, default=str(ROOT / "metrics_ms2s.json"))
    p.add_argument("--oof-out", type=str, default=str(ROOT / "oof_ms2s.npz"))
    p.add_argument("--holdout-logits-out", type=str, default=str(ROOT / "holdout_ms2s.npz"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--num-classes", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--hold-out-users", type=int, nargs="+", default=list(DEFAULT_HOLD_OUT_USERS))
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-holdout-train", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device} torch={torch.__version__}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"cuda_mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB", flush=True)

    cache = Path(args.cache_dir)
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    print(f"skel n={len(y)} imu has={has_imu.sum()}", flush=True)

    def make_ds(indices, train_mode: bool):
        return CachedDualDataset(
            X_skel, X_imu, y, users, indices, has_imu, augment=bool(train_mode), seed=args.seed
        )

    all_idx = np.arange(len(y))
    oof = np.zeros((len(y), args.num_classes), dtype=np.float32)
    fold_results = []

    gkf = GroupKFold(n_splits=args.cv_splits)
    for fi, (tr, va) in enumerate(gkf.split(all_idx, y, users)):
        vu = sorted(set(int(users[i]) for i in va))
        tu = sorted(set(int(users[i]) for i in tr))
        print(f"=== fold {fi} n_train={len(tr)} n_val={len(va)} val_users={vu} ===", flush=True)
        result = train_one(
            make_ds(tr, True),
            make_ds(va, False),
            args,
            device,
            f"fold{fi}",
            val_users=vu,
            train_users=tu,
        )
        logits = predict_logits(result["model"], make_ds(va, False), device, args.batch_size)
        oof[va] = logits
        del result["model"]
        fold_results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accs = [r["best_val_acc"] for r in fold_results]
    metrics = {
        "mode": "cv",
        "seed": args.seed,
        "model": "ms_stgcn_2s",
        "T": args.T,
        "device": str(device),
        "dual": True,
        "weighted_ce": True,
        "balanced_sampler": False,
        "label_smoothing": args.label_smoothing,
        "augment": True,
        "mixup": False,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "folds": fold_results,
        "cv_splits": args.cv_splits,
        "mean_val_acc": float(np.mean(accs)),
        "std_val_acc": float(np.std(accs)),
        "n_params": fold_results[0]["n_params"],
    }
    print(f"CV mean+/-std = {metrics['mean_val_acc']:.4f} +/- {metrics['std_val_acc']:.4f}", flush=True)
    hold = set(int(u) for u in args.hold_out_users)
    nh = np.array([int(u) not in hold for u in users])
    oof_acc = float((oof[nh].argmax(1) == y[nh]).mean())
    metrics["oof_acc_nonholdout"] = oof_acc
    print(f"OOF non-holdout acc={oof_acc:.4f}", flush=True)
    np.savez_compressed(
        args.oof_out,
        ms_stgcn_2s=oof,
        y=y.astype(np.int64),
        users=users.astype(np.int64),
    )
    print(f"Wrote OOF -> {args.oof_out}", flush=True)

    if not args.skip_holdout_train:
        tr, va = indices_by_users(users, args.hold_out_users)
        print(f"=== holdout users={args.hold_out_users} n_train={len(tr)} n_val={len(va)} ===", flush=True)
        hold_r = train_one(
            make_ds(tr, True),
            make_ds(va, False),
            args,
            device,
            "holdout",
            val_users=list(args.hold_out_users),
            train_users=sorted(set(int(users[i]) for i in tr)),
        )
        h_logits = predict_logits(hold_r["model"], make_ds(va, False), device, args.batch_size)
        del hold_r["model"]
        metrics["holdout"] = hold_r
        np.savez_compressed(
            args.holdout_logits_out,
            ms_stgcn_2s=h_logits,
            y=y[va].astype(np.int64),
            users=users[va].astype(np.int64),
        )
        print(f"holdout_acc={hold_r['best_val_acc']:.4f} wrote {args.holdout_logits_out}", flush=True)

    metrics["finished_at_unix"] = time.time()
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics -> {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
