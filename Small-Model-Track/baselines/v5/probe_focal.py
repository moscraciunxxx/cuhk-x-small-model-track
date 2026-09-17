"""Holdout-only MidFuse focal-loss tweak on hard classes from error analysis.

Train on non-holdout users, eval users {8,9,24}. Compare to 0.537 baseline.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_V2 = Path(r"D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2")
OUT = Path(r"D:\CUHK-X\Small-Model-Track\baselines\v5")
sys.path.insert(0, str(ROOT_V2))

from dataset import DEFAULT_HOLD_OUT_USERS, CachedDualDataset, load_skel_train_cache
from model import build_model, count_parameters


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # CE with label smoothing to get soft nll, then focal modulate with hard CE pt
        log_probs = F.log_softmax(logits, dim=1)
        n_class = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        ce = -(true_dist * log_probs).sum(dim=1)
        pt = torch.exp(-ce.detach())  # approx
        # better: use non-smoothed pt for focal
        probs = log_probs.exp()
        pt_hard = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
        focal = (1 - pt_hard) ** self.gamma * ce
        if self.weight is not None:
            focal = focal * self.weight[target]
        return focal.mean()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
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
            total_loss += float(loss.item()) * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += bs
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"cuda free={free/1e9:.2f}GB total={total/1e9:.2f}GB", flush=True)
        if free < 1.2e9:
            print("low VRAM, falling back to CPU", flush=True)
            device = torch.device("cpu")
    print("device", device, flush=True)

    cache = ROOT_V2 / "cache"
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y, dtype=np.int64)
    users = np.asarray(users, dtype=np.int64)

    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]
    print(f"n_train={len(tr_idx)} n_val={len(va_idx)}", flush=True)

    # Boost worst OOF classes from error analysis
    hard = {25, 26, 19, 18, 35, 16, 22, 14, 24, 8, 10, 2}  # worst recall OOF
    counts = np.bincount(y[tr_idx], minlength=40).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (40 * counts)
    w = np.clip(w, 0.25, 8.0)
    for c in hard:
        w[c] *= 1.5
    w = np.clip(w, 0.25, 10.0)
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    train_ds = CachedDualDataset(X_skel, X_imu, y, users, tr_idx, has_imu, augment=True, seed=42)
    val_ds = CachedDualDataset(X_skel, X_imu, y, users, va_idx, has_imu, augment=False)
    train_loader = DataLoader(train_ds, batch_size=40, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False, num_workers=0)

    model = build_model("midfuse", num_classes=40).to(device)
    print("params", count_parameters(model), flush=True)
    criterion = FocalLoss(weight=weight, gamma=2.0, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    epochs = 40
    patience = 12
    best_acc = -1.0
    best_state = None
    bad = 0
    history = []
    t0 = time.time()
    ckpt_dir = OUT / "checkpoints_focal"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, opt, device, False)
        history.append({"ep": ep, "tr_acc": tr_acc, "va_acc": va_acc, "tr_loss": tr_loss, "va_loss": va_loss})
        print(f"ep{ep:02d} tr={tr_acc:.3f} va={va_acc:.3f}", flush=True)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            torch.save({"model_state": best_state, "val_acc": best_acc, "epoch": ep, "model_name": "midfuse", "dual": True, "num_classes": 40}, ckpt_dir / "best_holdout_focal.pt")
        else:
            bad += 1
            if bad >= patience:
                print("early stop", flush=True)
                break

    baseline = 0.5366336633663367
    result = {
        "best_holdout_acc": best_acc,
        "baseline_midfuse_v2b": baseline,
        "delta": best_acc - baseline,
        "beats_baseline": bool(best_acc > baseline + 1e-6),
        "gamma": 2.0,
        "hard_class_boost": sorted(hard),
        "epochs_ran": len(history),
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "history": history,
    }
    (OUT / "focal_holdout_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
