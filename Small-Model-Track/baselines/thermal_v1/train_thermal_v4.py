"""Thermal v4: stronger recipe on existing thermal_yolo YOLO crop cache.
Improvements vs v2/v3: longer epochs, mixup, earlier unfreeze, longer patience.
Train 1-2 new pool seeds; dump hold+test logits for ir_v17 MidFuse.
No YOLO rebuild. No Kaggle submit.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained: bool = True) -> nn.Module:
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {
        "acc": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "logits": np.concatenate(logits_all),
        "y": yt,
    }


def train_one(
    X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience,
    ckpt_path, mixup_alpha=0.3, unfreeze_ep=3, label_smoothing=0.05,
):
    set_seed(seed)
    tag = f"pool_seed{seed}"
    model = build(True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ("stem", "layer1")):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    best_acc, best_ep, best_state, history = -1.0, 0, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all @ ep{ep}", flush=True)
        model.train()
        loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            if mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb_m = lam * xb + (1.0 - lam) * xb[idx]
                y1, y2 = yb, yb[idx]
            else:
                lam, xb_m, y1, y2 = 1.0, xb, yb, yb
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb_m)
                if lam < 1.0:
                    loss = lam * crit(logits, y1) + (1.0 - lam) * crit(logits, y2)
                else:
                    loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": ep,
            "tr_loss": loss_sum / max(n, 1),
            "tr_acc": correct / max(n, 1),
            "val_acc": metrics["acc"],
            "val_f1": metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
            f"val={row['val_acc']:.4f} f1={row['val_f1']:.4f}",
            flush=True,
        )
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save(
                {
                    "model": best_state,
                    "model_fp16": sd16,
                    "val_acc": best_acc,
                    "epoch": ep,
                    "seed": seed,
                    "tag": tag,
                    "history": history,
                    "hold_logits": metrics["logits"],
                    "hold_y": metrics["y"],
                    "recipe": {
                        "epochs": epochs,
                        "mixup": mixup_alpha,
                        "unfreeze_ep": unfreeze_ep,
                        "patience": patience,
                        "lr": lr,
                        "label_smoothing": label_smoothing,
                    },
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}@ep{best_ep}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    hold_m = evaluate(model, val_loader, device)
    del model
    torch.cuda.empty_cache()
    return best_acc, best_state, history, hold_m["logits"], hold_m["y"]


@torch.no_grad()
def infer_cache(model, Xt, device, bs=8):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i : i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i : i + len(x)] = model(x).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "thermal_yolo"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v4"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--seeds", type=int, nargs="+", default=[888, 3141])
    ap.add_argument("--mixup", type=float, default=0.3)
    ap.add_argument("--unfreeze-ep", type=int, default=3)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-seeds", type=int, default=0, help="If >0, train at most this many missing seeds")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy")
    users = np.load(cache / "train_users.npy")
    X = np.memmap(
        cache / f"train_x_t{args.t}_s{args.size}.npy",
        dtype=np.uint8,
        mode="r",
        shape=(len(y), args.t, args.size, args.size, 3),
    )
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(
        f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device} "
        f"cache={cache} mixup={args.mixup} ep={args.epochs} unfreeze={args.unfreeze_ep} pat={args.patience}",
        flush=True,
    )

    members = []
    trained = 0
    for seed in args.seeds:
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            acc = float(blob.get("val_acc", -1))
            print(f"reuse {ck.name} val={acc}", flush=True)
            hold_logits = blob.get("hold_logits")
            if hold_logits is None:
                model = build(False)
                model.load_state_dict(blob["model"])
                model.to(device)
                val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
                hold_m = evaluate(model, val_loader, device)
                hold_logits = hold_m["logits"]
                del model
                torch.cuda.empty_cache()
            members.append({"seed": seed, "acc": acc, "hold_logits": np.asarray(hold_logits), "ckpt": ck})
            continue
        if args.skip_train:
            print(f"missing {ck}", flush=True)
            continue
        if args.max_seeds > 0 and trained >= args.max_seeds:
            print(f"max-seeds={args.max_seeds} reached; skip seed{seed}", flush=True)
            continue
        acc, state, hist, hlog, hy = train_one(
            X, y, users, pool_idx, hold_idx, device, seed, args.epochs, args.batch_size,
            args.lr, args.patience, ck, mixup_alpha=args.mixup, unfreeze_ep=args.unfreeze_ep,
        )
        members.append({"seed": seed, "acc": acc, "hold_logits": hlog, "ckpt": ck})
        trained += 1

    if not members:
        raise SystemExit("no members")

    hold_stack = np.stack([m["hold_logits"] for m in members], 0)
    hold_ens = hold_stack.mean(0)
    yt = y[hold_idx]
    print(f"thermal v4 members: {[(m['seed'], round(m['acc'], 4)) for m in members]}", flush=True)
    print(f"thermal v4 ens hold={(hold_ens.argmax(1) == yt).mean():.4f}", flush=True)

    meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    keys = np.array(
        [(int(meta[i]["user_id"]), int(meta[i]["label"]), str(meta[i].get("trial", ""))) for i in hold_idx],
        dtype=object,
    )
    np.savez(
        ckpt_dir / "holdout_ensemble_logits.npz",
        logits=hold_ens,
        stack=hold_stack,
        y=yt,
        users=users[hold_idx],
        hold_idx=hold_idx,
        keys=keys,
        tags=np.array([f"pool_seed{m['seed']}" for m in members]),
        scores=np.array([m["acc"] for m in members]),
    )

    Xt = np.memmap(
        cache / f"test_x_t{args.t}_s{args.size}.npy",
        dtype=np.uint8,
        mode="r",
        shape=(405, args.t, args.size, args.size, 3),
    )
    test_logs = []
    for m in members:
        blob = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
        model = build(False)
        model.load_state_dict(blob["model"])
        model.to(device)
        base = infer_cache(model, Xt, device, bs=8)
        np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", base)
        test_logs.append(base)
        print(f"inferred test seed{m['seed']}", flush=True)
        del model
        torch.cuda.empty_cache()
    test_ens = np.mean(np.stack(test_logs, 0), 0)
    np.save(ckpt_dir / "test_logits.npy", test_ens)

    metrics = {
        "tag": "thermal_v4_strong",
        "cache": str(cache),
        "member_scores": {f"seed{m['seed']}": m["acc"] for m in members},
        "holdout_acc_ensemble": float((hold_ens.argmax(1) == yt).mean()),
        "holdout_acc_best_member": float(max(m["acc"] for m in members)),
        "n_hold": int(len(yt)),
        "prior_v2_trio_ens": 0.6031746031746031,
        "prior_v3_best_pool": 0.5813492063492064,
        "mixup": args.mixup,
        "unfreeze_ep": args.unfreeze_ep,
        "epochs": args.epochs,
        "patience": args.patience,
        "delta_ens_vs_v2_trio": float((hold_ens.argmax(1) == yt).mean() - 0.6031746031746031),
    }
    (ckpt_dir / "metrics_thermal_v4.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "metrics_thermal_v4.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
