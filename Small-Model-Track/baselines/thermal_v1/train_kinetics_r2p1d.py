"""Fine-tune torchvision r2plus1d_18 (Kinetics400) on YOLO-cropped Thermal; pack fp16 <100MB."""
from __future__ import annotations
import argparse, csv, json, random, time
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
# Kinetics mean/std
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(num_classes=40, pretrained=True):
    if pretrained:
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    else:
        m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def normalize(x):
    # x: B,T,C,H,W in [0,1]
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device))
        # B,T,C,H,W -> B,C,T,H,W
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pred": y_pred, "y": y_true,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "thermal_yolo"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache_dir)
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r", shape=(len(y), args.t, args.size, args.size, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device}")

    model = build(pretrained=not args.no_pretrained).to(device)
    # freeze early layers initially
    for name, p in model.named_parameters():
        if any(k in name for k in ["stem", "layer1"]):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")

    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True)
    train_loader = DataLoader(CachedClipDataset(X, y, users, pool_idx, train=True, seed=args.seed), batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False, seed=args.seed), batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    ckpt_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_state, best_ep, history = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if ep == 6:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
            print("unfroze all layers", flush=True)
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb); loss = crit(logits, yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, "tr_loss": loss_sum/max(n,1), "tr_acc": correct/max(n,1), "val_acc": metrics["acc"], "val_f1": metrics["macro_f1"]}
        history.append(row)
        print(f"[kinetics] ep{ep:03d} loss={row['tr_loss']:.4f} tr_acc={row['tr_acc']:.3f} val_acc={row['val_acc']:.4f} val_f1={row['val_f1']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # fp16 pack for size
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc, "epoch": ep, "args": vars(args)}, ckpt_dir / "holdout_train.pt")
            sz = (ckpt_dir / "holdout_train.pt").stat().st_size / (1024*1024)
            print(f"  saved best acc={best_acc:.4f} ckpt_mb={sz:.1f}", flush=True)
        if args.patience > 0 and ep - best_ep >= args.patience:
            print(f"early stop ep{ep} best={best_acc:.4f}", flush=True); break
    print(f"BEST holdout_acc={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    (ckpt_dir / "metrics.json").write_text(json.dumps({"best_acc": best_acc, "history": history}, indent=2), encoding="utf-8")

    # Infer
    if best_state is None:
        return
    model.load_state_dict(best_state); model.eval()
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r", shape=(len(meta), args.t, args.size, args.size, 3))
    logits = np.zeros((len(meta), 40), np.float32)
    with torch.no_grad():
        for i in range(0, len(meta), 16):
            arr = Xt[i:i+16].astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3)))
            x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            logits[i:i+len(x)] = model(x).float().cpu().numpy()
    preds = logits.argmax(1)
    fb = {}
    mid = Path(r"D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2\submission_skeleton_imu_v2_ensemble.csv")
    with mid.open() as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    out = ROOT / "submission_thermal_v1.csv"
    nfb = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            path = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(path, int(preds[i])); nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([path, pred])
    np.save(ckpt_dir / "test_logits.npy", logits)
    print("wrote", out, "fallback", nfb, "best_acc", best_acc)
    # promote if clearly better than MidFuse honest ~0.52
    if best_acc >= 0.55:
        dest = Path(r"D:\CUHK-X\Small-Model-Track\submission.csv")
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print("PROMOTED to", dest)
    else:
        print("NOT promoted (holdout_acc < 0.55); still submit-ready at", out)


if __name__ == "__main__":
    main()
