"""ir_v20 IR diversity: Kinetics S3D on IR YOLO-v4 crops (non-R2+1D, ~16MB fp16).
AdaptiveAvgPool3d so 112 crops work (native S3D avgpool expects ~224).
Abort guidance: if solo << MC3 (~0.614) and no complementarity after 1 seed + quick fuse, stop.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import s3d, S3D_Weights
from sklearn.metrics import f1_score
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained=True):
    m = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
    # Native AvgPool3d(2,7,7) needs ~224; use adaptive for 112 crops.
    m.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    m.classifier[1] = nn.Conv3d(1024, NUM_CLASSES, kernel_size=1)
    return m


def freeze_early(model, freeze=True):
    """Freeze early separable stem + first pools/inception (features 0..4)."""
    for i, child in enumerate(model.features):
        req = (not freeze) or (i > 4)
        for p in child.parameters():
            p.requires_grad = req


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {"acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "logits": np.concatenate(logits_all), "y": yt}


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience,
              ckpt_path, mixup_alpha=0.2, unfreeze_ep=3):
    set_seed(seed)
    tag = f"seed{seed}"
    model = build(True).to(device)
    freeze_early(model, True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=WeightedRandomSampler(w, num_samples=len(pool_idx), replacement=True),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            freeze_early(model, False)
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all", flush=True)
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            if mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb_m = lam * xb + (1 - lam) * xb[idx]
                y1, y2 = yb, yb[idx]
            else:
                lam, xb_m, y1, y2 = 1.0, xb, yb, yb
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb_m)
                if lam < 1.0:
                    loss = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2)
                else:
                    loss = crit(logits, yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, "tr_loss": loss_sum / max(n, 1), "tr_acc": correct / max(n, 1),
               "val_acc": metrics["acc"], "val_f1": metrics["macro_f1"]}
        history.append(row)
        print(f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
              f"val={row['val_acc']:.4f} f1={row['val_f1']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc,
                        "epoch": ep, "seed": seed, "history": history, "hold_logits": metrics["logits"],
                        "hold_y": metrics["y"], "arch": "s3d"}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state); model.to(device)
    hold_m = evaluate(model, val_loader, device)
    del model; torch.cuda.empty_cache()
    return best_acc, best_state, history, hold_m["logits"], hold_m["y"]


@torch.no_grad()
def infer_cache(model, Xt, device, bs=8):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i + len(x)] = model(x).float().cpu().numpy()
    return out


@torch.no_grad()
def infer_tta(model, Xt, device, bs=8):
    base = infer_cache(model, Xt, device, bs)
    Xt_f = Xt[:, :, :, ::-1, :].copy()
    flip = infer_cache(model, Xt_f, device, bs)
    return 0.5 * (base + flip), base, flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_s3d_v20"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--unfreeze-ep", type=int, default=3)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for S3D train; aborting (CPU-only prep mode?)")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), args.t, args.size, args.size, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device} cache={cache}", flush=True)
    n_params = sum(p.numel() for p in build(False).parameters())
    print(f"S3D params={n_params} fp16_mb~{n_params*2/1e6:.1f}", flush=True)

    members = []
    for seed in args.seeds:
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            acc = float(blob.get("val_acc", -1))
            print(f"reuse {ck.name} val={acc}", flush=True)
            hold_logits = blob.get("hold_logits")
            if hold_logits is None:
                model = build(False); model.load_state_dict(blob["model"]); model.to(device)
                val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                hold_m = evaluate(model, val_loader, device)
                hold_logits = hold_m["logits"]; del model; torch.cuda.empty_cache()
            members.append({"seed": seed, "acc": acc, "hold_logits": np.asarray(hold_logits), "ckpt": ck})
            continue
        if args.skip_train:
            print(f"missing {ck}", flush=True); continue
        acc, state, hist, hlog, hy = train_one(
            X, y, users, pool_idx, hold_idx, device, seed, args.epochs, args.batch_size,
            args.lr, args.patience, ck, mixup_alpha=args.mixup, unfreeze_ep=args.unfreeze_ep)
        members.append({"seed": seed, "acc": acc, "hold_logits": hlog, "ckpt": ck})

    if not members:
        raise SystemExit("no S3D members trained/reused")

    hold_stack = np.stack([m["hold_logits"] for m in members], 0)
    hold_ens = hold_stack.mean(0)
    yt = y[hold_idx]
    print(f"s3d members: {[(m['seed'], round(m['acc'],4)) for m in members]}", flush=True)
    print(f"s3d ens hold={(hold_ens.argmax(1)==yt).mean():.4f}", flush=True)
    meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    keys = np.array([(int(meta[i]["user_id"]), int(meta[i]["label"]), str(meta[i].get("trial","")))
                     for i in hold_idx], dtype=object)
    np.savez(ckpt_dir / "hold_logits_strong.npz",
             logits=hold_stack, ens=hold_ens, y=yt, users=users[hold_idx],
             hold_idx=hold_idx, keys=keys,
             seeds=np.array([m["seed"] for m in members]),
             scores=np.array([m["acc"] for m in members]))

    if not args.skip_test:
        Xt = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                       shape=(405, args.t, args.size, args.size, 3))
        test_logs = []
        for m in members:
            blob = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
            model = build(False); model.load_state_dict(blob["model"]); model.to(device)
            tta, base, _ = infer_tta(model, Xt, device, bs=8)
            np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", base)
            np.save(ckpt_dir / f"test_logits_seed{m['seed']}_tta.npy", tta)
            test_logs.append(tta)
            print(f"inferred test seed{m['seed']}", flush=True)
            del model; torch.cuda.empty_cache()
        test_ens = np.mean(test_logs, 0)
        np.save(ckpt_dir / "test_logits_ens.npy", test_ens)

    metrics = {
        "tag": "ir_s3d_v20",
        "arch": "s3d",
        "cache": str(cache),
        "member_scores": {f"seed{m['seed']}": m["acc"] for m in members},
        "holdout_acc_ensemble": float((hold_ens.argmax(1) == yt).mean()),
        "n_hold": int(len(yt)),
        "mixup": args.mixup,
        "unfreeze_ep": args.unfreeze_ep,
        "fp16_mb_approx": round(n_params * 2 / 1e6, 2),
        "yolo_mb": 6.5,
    }
    (ckpt_dir / "metrics_ir_s3d_v20.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
