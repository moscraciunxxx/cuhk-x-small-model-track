"""Train classic Kinetics R2P1D-18 on higher-T IR YOLO cache (T=24/32). Angle 3 of ir_v24."""
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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained=True):
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


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


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience, ckpt_path, mixup_alpha=0.2):
    set_seed(seed)
    tag = f"seed{seed}"
    model = build(True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ["stem", "layer1"]):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_loader = DataLoader(CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed),
                              batch_size=batch_size,
                              sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True),
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False, seed=0),
                            batch_size=max(batch_size, 4), shuffle=False, num_workers=0)
    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == 4:
            for p in model.parameters():
                p.requires_grad = True
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
                loss_fn = lambda logits: lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[idx])
            else:
                xb_m = xb
                loss_fn = lambda logits: crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb_m); loss = loss_fn(logits)
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
                        "epoch": ep, "seed": seed, "history": history}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    del model; torch.cuda.empty_cache()
    return best_acc, best_state, history


@torch.no_grad()
def infer_cache(model, Xt, device, bs=6):
    model.eval()
    out = np.zeros((len(Xt), 40), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i + len(x)] = model(x).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=24)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    cache = ROOT / "cache" / f"ir_yolo_v4_t{args.t}"
    ckpt_dir = ROOT / "checkpoints" / f"ir_yolo_r2p1d18_t{args.t}_v24"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), args.t, args.size, args.size, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"T={args.t} n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)}", flush=True)

    members = []
    for seed in args.seeds:
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if args.skip_train or ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            members.append({"seed": seed, "acc": float(blob["val_acc"]), "state": blob["model"], "path": ck})
            print(f"reuse {ck.name} val={blob.get('val_acc')}", flush=True)
            continue
        acc, state, hist = train_one(X, y, users, pool_idx, hold_idx, device, seed,
                                     args.epochs, args.batch_size, args.lr, args.patience, ck)
        members.append({"seed": seed, "acc": acc, "state": state, "path": ck})

    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=8, shuffle=False)
    hold_logits = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"])
        met = evaluate(model, val_loader, device)
        print(f"eval seed{m['seed']} hold={met['acc']:.4f}", flush=True)
        m["hold_logits"] = met["logits"]; m["acc"] = met["acc"]
        hold_logits.append(met["logits"])
        np.save(ckpt_dir / f"hold_logits_seed{m['seed']}.npy", met["logits"])
        del model; torch.cuda.empty_cache()
    ens = np.mean(hold_logits, 0)
    yt = y[hold_idx]
    ens_acc = float((ens.argmax(1) == yt).mean())
    print(f"ENSEMBLE hold={ens_acc:.4f}", flush=True)
    np.savez_compressed(ckpt_dir / "hold_logits_strong.npz", ens=ens,
                        **{f"s{m['seed']}": m["hold_logits"] for m in members}, y=yt, users=users[hold_idx])

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), args.t, args.size, args.size, 3))
    test_list = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"])
        tl = infer_cache(model, Xt, device)
        np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", tl)
        test_list.append(tl)
        del model; torch.cuda.empty_cache()
    np.save(ckpt_dir / "test_logits_ens.npy", np.mean(test_list, 0))
    report = {"t": args.t, "member_scores": {f"seed{m['seed']}": m["acc"] for m in members},
              "holdout_acc_ensemble": ens_acc, "ckpt_dir": str(ckpt_dir)}
    (ckpt_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
