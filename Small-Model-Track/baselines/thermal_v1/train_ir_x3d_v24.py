"""ir_v24: X3D-M (Kinetics) + optional R2+1D-R50 on IR YOLO-v4 crops.
pytorchvideo has no r2plus1d_34; X3D-M (~7MB fp16) and R2+1D-R50 (~54MB) are size-legal substitutes.
Adaptive pool so T=16 S=112 works. Dump hold/test logits for nested fuse vs ir_v7.
"""
from __future__ import annotations
import argparse, json, random, time, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from pytorchvideo.models.hub import x3d_m, r2plus1d_r50
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
# Kinetics-style (same as classic R2P1D-18 pipeline)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(arch: str, pretrained=True):
    if arch == "x3d_m":
        m = x3d_m(pretrained=pretrained)
        m.blocks[5].pool.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        in_f = m.blocks[5].proj.in_features
        m.blocks[5].proj = nn.Linear(in_f, NUM_CLASSES)
        m.blocks[5].activation = nn.Identity()
        return m
    if arch == "r2plus1d_r50":
        m = r2plus1d_r50(pretrained=pretrained)
        head = m.blocks[-1]
        head.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        head.proj = nn.Linear(head.proj.in_features, NUM_CLASSES)
        head.activation = nn.Identity()
        return m
    raise ValueError(arch)


def normalize(x):
    # x: B,T,C,H,W
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def maybe_resize(x, size: int):
    # x B,T,C,H,W -> resize spatial if needed
    if x.shape[-1] == size:
        return x
    b, t, c, h, w = x.shape
    x = x.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.reshape(b, t, c, size, size)


@torch.no_grad()
def evaluate(model, loader, device, size):
    model.eval(); ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device))
        x = maybe_resize(x, size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {"acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "logits": np.concatenate(logits_all), "y": yt}


def freeze_early(model, arch):
    if arch == "x3d_m":
        # freeze stem + first ResStage
        for i in range(2):
            for p in model.blocks[i].parameters():
                p.requires_grad = False
    else:
        for i in range(2):
            for p in model.blocks[i].parameters():
                p.requires_grad = False


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience,
              ckpt_path, arch, size, mixup_alpha=0.2, unfreeze_ep=4, focal_gamma=0.0):
    set_seed(seed)
    tag = f"{arch}_seed{seed}"
    model = build(arch, True).to(device)
    freeze_early(model, arch)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if focal_gamma > 0:
        # class-balanced CE via weights + focal
        counts = np.bincount(y[pool_idx], minlength=40).astype(np.float64)
        cw = (counts.sum() / np.maximum(counts, 1.0))
        cw = cw / cw.mean()
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
        def crit(logits, target):
            logp = F.log_softmax(logits, dim=1)
            p = logp.exp()
            pt = p.gather(1, target.view(-1, 1)).squeeze(1)
            w = cw_t.gather(0, target)
            loss = -w * ((1 - pt) ** focal_gamma) * logp.gather(1, target.view(-1, 1)).squeeze(1)
            return loss.mean()
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=WeightedRandomSampler(w, num_samples=len(pool_idx), replacement=True),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 4), shuffle=False, num_workers=0, pin_memory=True)

    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all", flush=True)
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device))
            xb = maybe_resize(xb, size)
            xb = xb.permute(0, 2, 1, 3, 4).contiguous()
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
        metrics = evaluate(model, val_loader, device, size)
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
                        "epoch": ep, "seed": seed, "arch": arch, "size": size, "history": history}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    del model; torch.cuda.empty_cache()
    return best_acc, best_state, history


@torch.no_grad()
def infer_cache(model, Xt, device, size, bs=4):
    model.eval()
    out = np.zeros((len(Xt), 40), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x)
        x = maybe_resize(x, size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        out[i:i + len(x)] = model(x).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_x3d_m_v24"))
    ap.add_argument("--arch", default="x3d_m", choices=["x3d_m", "r2plus1d_r50"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112, help="model spatial size (bilinear up from 112 cache if >112)")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s112.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), args.t, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device} arch={args.arch} size={args.size}", flush=True)

    members = []
    if not args.skip_train:
        for seed in args.seeds:
            ck = ckpt_dir / f"pool_seed{seed}.pt"
            if ck.exists():
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                print(f"reuse {ck.name} val={blob.get('val_acc')}", flush=True)
                members.append({"seed": seed, "acc": float(blob["val_acc"]), "state": blob["model"], "path": ck})
                continue
            acc, state, hist = train_one(
                X, y, users, pool_idx, hold_idx, device, seed,
                args.epochs, args.batch_size, args.lr, args.patience, ck,
                args.arch, args.size, mixup_alpha=args.mixup, focal_gamma=args.focal_gamma,
            )
            members.append({"seed": seed, "acc": acc, "state": state, "path": ck})
    else:
        for seed in args.seeds:
            ck = ckpt_dir / f"pool_seed{seed}.pt"
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            members.append({"seed": seed, "acc": float(blob["val_acc"]), "state": blob["model"], "path": ck})

    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=8, shuffle=False)
    hold_logits = []
    for m in members:
        model = build(args.arch, False).to(device); model.load_state_dict(m["state"])
        met = evaluate(model, val_loader, device, args.size)
        print(f"eval seed{m['seed']} hold={met['acc']:.4f} f1={met['macro_f1']:.4f}", flush=True)
        m["hold_logits"] = met["logits"]; m["acc"] = met["acc"]
        hold_logits.append(met["logits"])
        np.save(ckpt_dir / f"hold_logits_seed{m['seed']}.npy", met["logits"])
        del model; torch.cuda.empty_cache()

    ens = np.mean(hold_logits, 0)
    yt = y[hold_idx]
    ens_acc = float((ens.argmax(1) == yt).mean())
    print(f"ENSEMBLE hold={ens_acc:.4f}", flush=True)
    np.savez_compressed(ckpt_dir / "hold_logits_strong.npz", ens=ens, **{f"s{m['seed']}": m["hold_logits"] for m in members}, y=yt, users=users[hold_idx])

    # test infer
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s112.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), args.t, 112, 112, 3))
    test_list = []
    for m in members:
        model = build(args.arch, False).to(device); model.load_state_dict(m["state"])
        tl = infer_cache(model, Xt, device, args.size, bs=4)
        np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", tl)
        test_list.append(tl)
        del model; torch.cuda.empty_cache()
    test_ens = np.mean(test_list, 0)
    np.save(ckpt_dir / "test_logits_ens.npy", test_ens)

    # fp16 pack size
    best = max(members, key=lambda d: d["acc"])
    fp16 = ckpt_dir / "model_fp16.pt"
    torch.save({"model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in best["state"].items()},
                "val_acc": best["acc"], "seed": best["seed"], "arch": args.arch}, fp16)
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    report = {
        "tag": "ir_v24",
        "arch": args.arch,
        "size": args.size,
        "member_scores": {f"seed{m['seed']}": m["acc"] for m in members},
        "holdout_acc_ensemble": ens_acc,
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) < 100,
        "ckpt_dir": str(ckpt_dir),
    }
    (ckpt_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
