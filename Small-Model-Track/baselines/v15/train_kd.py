"""Knowledge distill: compact student on v13-selected ensemble soft labels (OOF-honest)."""
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
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
V10 = ROOT.parent / "v10"
V11 = ROOT.parent / "v11"
V13 = ROOT.parent / "v13"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_skel_train_cache,
)
from model import MidFusePlus, count_parameters  # noqa: E402 — student can be MidFusePlus or Compact

# Import CompactMidFuse from v2
import importlib.util
spec = importlib.util.spec_from_file_location("v2_model_kd", V2 / "model.py")
v2m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2m)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def build_teacher_probs(y, users):
    """v13-selected-style soft labels from OOF logits (bases are OOF-honest)."""
    z = np.load(V10 / "oof_logits_v10.npz")
    L = {k: z[k] for k in ["midfuse", "midfuse2", "midfuse3", "stgcn", "gru", "gru2", "deepconv", "cfuse"]}
    s2s = np.load(V13 / "oof_stgcn2s.npz")
    L["s2s"] = s2s["stgcn_2s"]
    P = {k: softmax_np(v) for k, v in L.items()}
    P["mf_avg"] = softmax_np((L["midfuse"] + L["midfuse2"] + L["midfuse3"]) / 3.0)
    P["gru_avg"] = softmax_np(0.5 * (L["gru"] + L["gru2"]))
    # Classic A/B/G equal-ish blends used in v13 — use fixed equal for teacher stability
    # A ~ pow mf_avg,stgcn,gru,cfuse — approximate with equal for soft labels
    A = (P["mf_avg"] + P["stgcn"] + P["gru"] + P["cfuse"]) / 4.0
    B = (P["mf_avg"] + P["stgcn"] + P["gru_avg"]) / 3.0
    G = B.copy()
    eq_mf_st_s2s = (P["mf_avg"] + P["stgcn"] + P["s2s"]) / 3.0
    # v13 selected: (2A+B+G)/4 then + eq_mf_st_s2s x1 -> /5
    base = (2 * A + B + G) / 4.0
    teacher = (base * 4.0 + 1.0 * eq_mf_st_s2s) / 5.0
    teacher = teacher / np.maximum(teacher.sum(1, keepdims=True), 1e-12)
    return teacher.astype(np.float32)


class DualSoftDataset(Dataset):
    def __init__(self, X_skel, X_imu, y, users, soft, indices, has_imu, augment=False, seed=42):
        self.base = CachedDualDataset(X_skel, X_imu, y, users, indices, has_imu, augment=augment, seed=seed)
        self.soft = soft
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        xs, xi, y, user, flag = self.base[i]
        idx = int(self.indices[i])
        return xs, xi, y, user, flag, torch.from_numpy(self.soft[idx])

    @property
    def labels(self):
        return self.base.labels


def kd_loss(logits, y, soft, T=2.0, alpha=0.7):
    """alpha * CE(hard) + (1-alpha) * T^2 * KL(student||teacher)."""
    ce = F.cross_entropy(logits, y, label_smoothing=0.05)
    log_p = F.log_softmax(logits / T, dim=1)
    q = soft  # already probs
    # soft may be probs; convert to soft targets at temp by renormalizing log
    # Use KL(student || teacher) with teacher as probs softened:
    q_t = F.softmax(torch.log(q.clamp_min(1e-8)) / T, dim=1)
    kl = F.kl_div(log_p, q_t, reduction="batchmean") * (T * T)
    return alpha * ce + (1.0 - alpha) * kl


def run_epoch(model, loader, optimizer, device, train, T, alpha):
    model.train(train)
    total_loss = total_correct = total_n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xs, xi, y, _u, flag, soft in loader:
            xs, xi, y, flag, soft = (
                xs.to(device),
                xi.to(device),
                y.to(device),
                flag.to(device),
                soft.to(device),
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xs, xi, flag)
            loss = kd_loss(logits, y, soft, T=T, alpha=alpha)
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
def predict_logits(model, ds, device, bs=48):
    # ds yields 6-tuples; wrap
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    outs = []
    model.eval()
    for batch in loader:
        xs, xi, y, _u, flag = batch[:5]
        outs.append(model(xs.to(device), xi.to(device), flag.to(device)).float().cpu().numpy())
    return np.concatenate(outs, 0)


def build_student(name, num_classes=40):
    if name == "compact":
        return v2m.CompactMidFuse(num_classes=num_classes)
    if name == "midfuse":
        return v2m.MidFuseNet(num_classes=num_classes)
    if name == "mfp":
        return MidFusePlus(num_classes=num_classes, use_velocity=True)
    raise ValueError(name)


def train_one(train_ds, val_ds, args, device, tag):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_student(args.student, args.num_classes).to(device)
    n_params = count_parameters(model) if hasattr(model, "parameters") else sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params} student={args.student}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_val = -1.0
    best_path = Path(args.ckpt_dir) / f"best_{tag}.pt"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    patience = args.patience
    hist = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device, True, args.T, args.alpha)
        va_loss, va_acc = run_epoch(model, val_loader, opt, device, False, args.T, args.alpha)
        sch.step()
        hist.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": va_acc, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"[{tag}] epoch {epoch}/{args.epochs} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}", flush=True)
        if va_acc >= best_val:
            best_val = va_acc
            patience = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.student,
                    "num_classes": args.num_classes,
                    "val_acc": best_val,
                    "epoch": epoch,
                    "n_params": n_params,
                    "tag": tag,
                    "kd_T": args.T,
                    "kd_alpha": args.alpha,
                },
                best_path,
            )
        else:
            patience -= 1
            if args.patience > 0 and patience <= 0:
                print(f"[{tag}] early stop at epoch {epoch}", flush=True)
                break
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return {
        "tag": tag,
        "best_val_acc": float(best_val),
        "n_params": n_params,
        "ckpt": str(best_path),
        "history": hist,
        "elapsed_sec": time.time() - t0,
        "model": model,
    }


def indices_by_users(users_arr, hold_out):
    hold = set(int(u) for u in hold_out)
    train_idx = np.where(~np.isin(users_arr, list(hold)))[0]
    val_idx = np.where(np.isin(users_arr, list(hold)))[0]
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="compact", choices=["compact", "midfuse", "mfp"])
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=40)
    ap.add_argument("--folds-only", type=str, default="")
    ap.add_argument("--skip-holdout-train", action="store_true")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints_kd"))
    ap.add_argument("--metrics-out", default=str(ROOT / "metrics_kd.json"))
    ap.add_argument("--oof-out", default=str(ROOT / "oof_kd.npz"))
    ap.add_argument("--holdout-logits-out", default=str(ROOT / "holdout_kd.npz"))
    ap.add_argument("--cache-dir", default=str(V2 / "cache"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} student={args.student} T={args.T} alpha={args.alpha}", flush=True)

    cache = Path(args.cache_dir)
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz")
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    soft = build_teacher_probs(y, users)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    nh = np.array([int(u) not in hold for u in users])
    print(f"teacher soft argmax acc (all)={(soft.argmax(1)==y).mean():.4f} nh={(soft[nh].argmax(1)==y[nh]).mean():.4f}", flush=True)

    def make_ds(indices, train_mode):
        return DualSoftDataset(X_skel, X_imu, y, users, soft, indices, has_imu, augment=bool(train_mode), seed=args.seed)

    all_idx = np.arange(len(y))
    oof = np.zeros((len(y), args.num_classes), np.float32)
    fold_filter = None
    if args.folds_only.strip():
        fold_filter = {int(x) for x in args.folds_only.split(",") if x.strip() != ""}
    fold_results = []
    gkf = GroupKFold(n_splits=5)
    for fi, (tr, va) in enumerate(gkf.split(all_idx, y, users)):
        if fold_filter is not None and fi not in fold_filter:
            print(f"skip fold{fi}", flush=True)
            continue
        print(f"=== fold {fi} n_train={len(tr)} n_val={len(va)} ===", flush=True)
        result = train_one(make_ds(tr, True), make_ds(va, False), args, device, f"fold{fi}")
        oof[va] = predict_logits(result["model"], make_ds(va, False), device, args.batch_size)
        del result["model"]
        fold_results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accs = [r["best_val_acc"] for r in fold_results]
    metrics = {
        "model": f"kd_{args.student}",
        "student": args.student,
        "kd_T": args.T,
        "kd_alpha": args.alpha,
        "folds": [{k: v for k, v in r.items() if k != "model"} for r in fold_results],
        "mean_val_acc": float(np.mean(accs)) if accs else None,
        "std_val_acc": float(np.std(accs)) if accs else None,
        "n_params": fold_results[0]["n_params"] if fold_results else None,
    }
    if accs:
        oof_acc = float((oof[nh].argmax(1) == y[nh]).mean())
        metrics["oof_acc_nonholdout"] = oof_acc
        print(f"CV {metrics['mean_val_acc']:.4f}+/-{metrics['std_val_acc']:.4f} OOF_nh={oof_acc:.4f}", flush=True)
        np.savez_compressed(args.oof_out, kd=oof, y=y.astype(np.int64), users=users.astype(np.int64))
        print("Wrote", args.oof_out, flush=True)

    if not args.skip_holdout_train:
        tr, va = indices_by_users(users, DEFAULT_HOLD_OUT_USERS)
        print(f"=== holdout n_train={len(tr)} n_val={len(va)} ===", flush=True)
        hold_r = train_one(make_ds(tr, True), make_ds(va, False), args, device, "holdout")
        h_logits = predict_logits(hold_r["model"], make_ds(va, False), device, args.batch_size)
        del hold_r["model"]
        metrics["holdout"] = {k: v for k, v in hold_r.items() if k != "model"}
        np.savez_compressed(args.holdout_logits_out, kd=h_logits, y=y[va].astype(np.int64), users=users[va].astype(np.int64))
        print(f"holdout_acc={hold_r['best_val_acc']:.4f}", flush=True)

    metrics["finished_at_unix"] = time.time()
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Wrote", args.metrics_out, flush=True)


if __name__ == "__main__":
    main()
