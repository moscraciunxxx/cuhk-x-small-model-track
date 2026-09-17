"""Train Depth_Color / Thermal temporal CNN with GroupKFold + honest holdout."""
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
from model import build_model, count_parameters, model_size_mb

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(labels), replacement=True)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = np.clip(w, 0.25, 8.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.numpy())
        preds.append(pred)
        logits_all.append(logits.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = float((y_true == y_pred).mean())
    return {"macro_f1": float(macro), "acc": acc, "logits": np.concatenate(logits_all), "y": y_true, "pred": y_pred}


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
    x_path = cache_dir / f"train_x_t{t}_s{size}.npy"
    y = np.load(cache_dir / "train_y.npy")
    users = np.load(cache_dir / "train_users.npy")
    X = np.memmap(x_path, dtype=np.uint8, mode="r", shape=(len(y), t, size, size, in_ch))
    return X, y, users


def train_one(
    X, y, users, train_idx, val_idx, device, args, ckpt_path: Path, tag: str
):
    train_ds = CachedClipDataset(X, y, users, train_idx, train=True, seed=args.seed)
    val_ds = CachedClipDataset(X, y, users, val_idx, train=False, seed=args.seed)
    sampler = make_sampler(y[train_idx])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = build_model(NUM_CLASSES, in_ch=args.in_ch).to(device)
    cw = class_weights(y[train_idx]).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_f1, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True, scaler)
        metrics = evaluate(model, val_loader, device)
        sched.step()
        row = {
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "val_f1": metrics["macro_f1"],
            "val_acc": metrics["acc"],
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[{tag}] ep{epoch:03d} loss={tr_loss:.4f} acc={tr_acc:.3f} "
            f"val_f1={metrics['macro_f1']:.4f} val_acc={metrics['acc']:.3f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "val_f1": best_f1,
                    "epoch": epoch,
                    "args": vars(args),
                    "tag": tag,
                },
                ckpt_path,
            )
        if args.patience > 0:
            best_ep = max((h["epoch"] for h in history if abs(h["val_f1"] - best_f1) < 1e-9), default=epoch)
            if epoch - best_ep >= args.patience:
                print(f"[{tag}] early stop at ep{epoch}, best_f1={best_f1:.4f}")
                break

    # reload best for OOF logits
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, device)
    print(f"[{tag}] BEST val_f1={best_f1:.4f} size_mb={model_size_mb(model):.2f} took={time.time()-t0:.1f}s")
    return {
        "best_f1": best_f1,
        "history": history,
        "val_idx": val_idx.tolist(),
        "logits": metrics["logits"],
        "y": metrics["y"],
        "pred": metrics["pred"],
        "ckpt": str(ckpt_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color")
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--holdout-only", action="store_true", help="Fast path: train all non-holdout, val=holdout")
    ap.add_argument("--full-train-after", action="store_true", help="After CV, train on all non-holdout for submit")
    ap.add_argument("--cache-dir", type=str, default="")
    args = ap.parse_args()
    set_seed(args.seed)

    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "cache" / args.modality.lower()
    ckpt_dir = ROOT / "checkpoints" / args.modality.lower()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, users = load_cache(cache_dir, args.t, args.size, args.in_ch)
    hold_set = set(DEFAULT_HOLD_OUT_USERS)
    hold_idx = np.where(np.isin(users, list(hold_set)))[0]
    pool_idx = np.where(~np.isin(users, list(hold_set)))[0]
    print(
        f"device={device} n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} "
        f"params~{count_parameters(build_model())}"
    )

    results = {"args": vars(args), "holdout_users": list(DEFAULT_HOLD_OUT_USERS)}

    if args.holdout_only:
        tag = "holdout_train"
        out = train_one(
            X, y, users, pool_idx, hold_idx, device, args, ckpt_dir / f"{tag}.pt", tag
        )
        results["holdout"] = {
            "macro_f1": out["best_f1"],
            "ckpt": out["ckpt"],
            "history": out["history"],
        }
        # also dump holdout logits
        np.savez(ckpt_dir / "holdout_preds.npz", logits=out["logits"], y=out["y"], pred=out["pred"], idx=hold_idx)
    else:
        # GroupKFold on pool users
        gkf = GroupKFold(n_splits=args.folds)
        oof_logits = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
        oof_mask = np.zeros(len(y), dtype=bool)
        fold_scores = []
        for fold, (tr, va) in enumerate(gkf.split(pool_idx, y[pool_idx], groups=users[pool_idx])):
            tr_idx = pool_idx[tr]
            va_idx = pool_idx[va]
            tag = f"fold{fold}"
            out = train_one(
                X, y, users, tr_idx, va_idx, device, args, ckpt_dir / f"{tag}.pt", tag
            )
            oof_logits[va_idx] = out["logits"]
            oof_mask[va_idx] = True
            fold_scores.append(out["best_f1"])
            print(f"FOLD {fold} f1={out['best_f1']:.4f}")

        oof_pred = oof_logits[oof_mask].argmax(1)
        oof_f1 = float(f1_score(y[oof_mask], oof_pred, average="macro", zero_division=0))
        results["oof_macro_f1"] = oof_f1
        results["fold_scores"] = fold_scores
        np.savez(ckpt_dir / "oof_logits.npz", logits=oof_logits, mask=oof_mask, y=y, users=users)
        print(f"OOF macro_f1={oof_f1:.4f} folds={fold_scores}")

        # Honest holdout: train on all pool, eval holdout
        tag = "holdout_train"
        out = train_one(
            X, y, users, pool_idx, hold_idx, device, args, ckpt_dir / f"{tag}.pt", tag
        )
        results["holdout"] = {"macro_f1": out["best_f1"], "ckpt": out["ckpt"]}
        np.savez(ckpt_dir / "holdout_preds.npz", logits=out["logits"], y=out["y"], pred=out["pred"], idx=hold_idx)
        print(f"HOLDOUT macro_f1={out['best_f1']:.4f}")

    if args.full_train_after or args.holdout_only:
        # final submit ckpt = best holdout_train (already saved)
        results["submit_ckpt"] = str(ckpt_dir / "holdout_train.pt")

    metrics_path = ROOT / f"metrics_{args.modality.lower()}.json"
    # strip huge history optionally keep
    metrics_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("wrote", metrics_path)


if __name__ == "__main__":
    main()
