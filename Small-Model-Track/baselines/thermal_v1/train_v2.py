"""Train temporal video HAR; arch=cnn_gru|r2p1d."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(labels), True)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(np.clip(w, 0.25, 8.0), dtype=torch.float32)


def build(arch: str, in_ch: int = 3):
    if arch == "r2p1d":
        from model_r2p1d import build_model_r2p1d, count_parameters, model_size_mb
        return build_model_r2p1d(NUM_CLASSES, in_ch=in_ch, base=64), count_parameters, model_size_mb
    from model import build_model, count_parameters, model_size_mb
    return build_model(NUM_CLASSES, in_ch=in_ch), count_parameters, model_size_mb


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        logits_all.append(logits.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc": float((y_true == y_pred).mean()),
        "logits": np.concatenate(logits_all),
        "y": y_true,
        "pred": y_pred,
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _u, _i in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if scaler is not None and train:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


def load_cache(cache_dir: Path, t: int, size: int, in_ch: int):
    y = np.load(cache_dir / "train_y.npy")
    users = np.load(cache_dir / "train_users.npy")
    X = np.memmap(cache_dir / f"train_x_t{t}_s{size}.npy", dtype=np.uint8, mode="r", shape=(len(y), t, size, size, in_ch))
    return X, y, users


def train_one(X, y, users, train_idx, val_idx, device, args, ckpt_path: Path, tag: str, build_fn, size_fn):
    train_ds = CachedClipDataset(X, y, users, train_idx, train=True, seed=args.seed)
    val_ds = CachedClipDataset(X, y, users, val_idx, train=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=make_sampler(y[train_idx]), num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = build_fn().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_idx]).to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_acc, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True, scaler)
        metrics = evaluate(model, val_loader, device)
        sched.step()
        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc, "val_f1": metrics["macro_f1"], "val_acc": metrics["acc"]})
        print(f"[{tag}] ep{epoch:03d} loss={tr_loss:.4f} acc={tr_acc:.3f} val_f1={metrics['macro_f1']:.4f} val_acc={metrics['acc']:.3f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "val_acc": best_acc, "val_f1": metrics["macro_f1"], "epoch": epoch, "args": vars(args), "tag": tag, "arch": args.arch}, ckpt_path)
        best_ep = max((h["epoch"] for h in history if abs(h["val_acc"] - best_acc) < 1e-9), default=epoch)
        if args.patience > 0 and epoch - best_ep >= args.patience:
            print(f"[{tag}] early stop at ep{epoch}, best_acc={best_acc:.4f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, device)
    print(f"[{tag}] BEST val_acc={best_acc:.4f} size_mb={size_fn(model):.2f} took={time.time()-t0:.1f}s", flush=True)
    return {"best_f1": metrics["macro_f1"], "best_acc": best_acc, "history": history, "logits": metrics["logits"], "y": metrics["y"], "pred": metrics["pred"], "ckpt": str(ckpt_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color")
    ap.add_argument("--arch", default="cnn_gru", choices=["cnn_gru", "r2p1d"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--holdout-only", action="store_true")
    ap.add_argument("--cache-dir", type=str, default="")
    args = ap.parse_args()
    set_seed(args.seed)

    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "cache" / args.modality.lower()
    ckpt_dir = ROOT / "checkpoints" / f"{args.modality.lower()}_{args.arch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, users = load_cache(cache_dir, args.t, args.size, args.in_ch)
    hold_set = set(DEFAULT_HOLD_OUT_USERS)
    hold_idx = np.where(np.isin(users, list(hold_set)))[0]
    pool_idx = np.where(~np.isin(users, list(hold_set)))[0]
    proto, count_parameters, model_size_mb = build(args.arch, args.in_ch)
    print(f"device={device} arch={args.arch} n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} params~{count_parameters(proto)}", flush=True)
    del proto

    def build_fn():
        m, _, _ = build(args.arch, args.in_ch)
        return m

    results = {"args": vars(args), "holdout_users": list(DEFAULT_HOLD_OUT_USERS)}
    if args.holdout_only:
        out = train_one(X, y, users, pool_idx, hold_idx, device, args, ckpt_dir / "holdout_train.pt", "holdout_train", build_fn, model_size_mb)
        results["holdout"] = {"macro_f1": out["best_f1"], "acc": out.get("best_acc"), "ckpt": out["ckpt"]}
        np.savez(ckpt_dir / "holdout_preds.npz", logits=out["logits"], y=out["y"], pred=out["pred"], idx=hold_idx)
    else:
        gkf = GroupKFold(n_splits=args.folds)
        oof_logits = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
        oof_mask = np.zeros(len(y), dtype=bool)
        fold_scores = []
        for fold, (tr, va) in enumerate(gkf.split(pool_idx, y[pool_idx], groups=users[pool_idx])):
            out = train_one(X, y, users, pool_idx[tr], pool_idx[va], device, args, ckpt_dir / f"fold{fold}.pt", f"fold{fold}", build_fn, model_size_mb)
            oof_logits[pool_idx[va]] = out["logits"]
            oof_mask[pool_idx[va]] = True
            fold_scores.append(out["best_f1"])
        oof_f1 = float(f1_score(y[oof_mask], oof_logits[oof_mask].argmax(1), average="macro", zero_division=0))
        results["oof_macro_f1"] = oof_f1
        results["fold_scores"] = fold_scores
        np.savez(ckpt_dir / "oof_logits.npz", logits=oof_logits, mask=oof_mask, y=y, users=users)
        print(f"OOF macro_f1={oof_f1:.4f} folds={fold_scores}", flush=True)
        out = train_one(X, y, users, pool_idx, hold_idx, device, args, ckpt_dir / "holdout_train.pt", "holdout_train", build_fn, model_size_mb)
        results["holdout"] = {"macro_f1": out["best_f1"], "acc": out.get("best_acc"), "ckpt": out["ckpt"]}
        np.savez(ckpt_dir / "holdout_preds.npz", logits=out["logits"], y=out["y"], pred=out["pred"], idx=hold_idx)
        print(f"HOLDOUT macro_f1={out['best_f1']:.4f}", flush=True)

    results["submit_ckpt"] = str(ckpt_dir / "holdout_train.pt")
    (ROOT / f"metrics_{args.modality.lower()}_{args.arch}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("wrote metrics", flush=True)


if __name__ == "__main__":
    main()
