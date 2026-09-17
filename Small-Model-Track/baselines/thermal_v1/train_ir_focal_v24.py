"""Classic R2P1D-18 + focal/class-balanced CE on ir_yolo_v4 (Angle 4). Launch only when GPU free."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)
# from logs/ir_v7_class_weak.json
WEAK = {25, 38, 26, 37, 18, 24, 13, 15, 11, 0, 16, 19}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained=True):
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


class FocalCBCE(nn.Module):
    def __init__(self, class_weights, gamma=1.5, label_smoothing=0.02):
        super().__init__()
        self.register_buffer("cw", class_weights)
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, target):
        n = logits.size(0); c = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        # label smoothing CE
        with torch.no_grad():
            true = torch.zeros_like(logits).fill_(self.ls / (c - 1))
            true.scatter_(1, target.view(-1, 1), 1.0 - self.ls)
        ce = -(true * logp).sum(1)
        p = logp.exp().gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6, 1)
        w = self.cw.gather(0, target)
        # upweight weak classes further
        weak_boost = torch.ones_like(w)
        for k in WEAK:
            weak_boost = torch.where(target == k, weak_boost * 1.5, weak_boost)
        loss = w * weak_boost * ((1 - p) ** self.gamma) * ce
        return loss.mean()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2024])
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--cache-dir", type=str, default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", type=str, default=str(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_v24"))
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    counts = np.bincount(y[pool_idx], minlength=40).astype(np.float64)
    cw = counts.sum() / np.maximum(counts, 1.0); cw = cw / cw.mean()
    cw_t = torch.tensor(cw, dtype=torch.float32)

    members = []
    for seed in args.seeds:
        set_seed(seed)
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            members.append({"seed": seed, "acc": float(blob["val_acc"]), "state": blob["model"]})
            print("reuse", ck.name, blob["val_acc"], flush=True); continue
        model = build(True).to(device)
        for name, p in model.named_parameters():
            if any(k in name for k in ["stem", "layer1"]):
                p.requires_grad = False
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = FocalCBCE(cw_t.to(device), gamma=args.gamma)
        scaler = torch.amp.GradScaler("cuda")
        w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
        # extra sample weight for weak
        for i, yi in enumerate(y[pool_idx]):
            if int(yi) in WEAK:
                w[i] *= 1.75
        tl = DataLoader(CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed),
                        batch_size=args.batch_size,
                        sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True),
                        num_workers=0, drop_last=True)
        vl = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=8, shuffle=False)
        best, best_ep, best_state = -1.0, -1, None
        for ep in range(1, args.epochs + 1):
            if ep == 4:
                for p in model.parameters():
                    p.requires_grad = True
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - ep + 1, 1))
                print(f"[seed{seed}] unfroze", flush=True)
            model.train(); loss_sum = n = correct = 0
            for xb, yb, _u, _i in tl:
                xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous(); yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits = model(xb); loss = crit(logits, yb)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                loss_sum += float(loss.item()) * len(yb)
                correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)
            sched.step()
            met = evaluate(model, vl, device)
            print(f"[seed{seed}] ep{ep:03d} loss={loss_sum/max(n,1):.4f} tr={correct/max(n,1):.3f} val={met['acc']:.4f} f1={met['macro_f1']:.4f}", flush=True)
            if met["acc"] > best:
                best, best_ep = met["acc"], ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model": best_state, "val_acc": best, "epoch": ep, "seed": seed}, ck)
                print("  saved", best, flush=True)
            if args.patience > 0 and ep - best_ep >= args.patience:
                print("early stop", best, flush=True); break
        members.append({"seed": seed, "acc": best, "state": best_state})
        del model; torch.cuda.empty_cache()

    vl = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=8, shuffle=False)
    holds = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"])
        met = evaluate(model, vl, device)
        holds.append(met["logits"]); np.save(ckpt_dir / f"hold_logits_seed{m['seed']}.npy", met["logits"])
        print("eval", m["seed"], met["acc"], flush=True)
        del model; torch.cuda.empty_cache()
    ens = np.mean(holds, 0)
    np.savez_compressed(ckpt_dir / "hold_logits_strong.npz", ens=ens, y=y[hold_idx], users=users[hold_idx])
    print("ENSEMBLE", float((ens.argmax(1) == y[hold_idx]).mean()), flush=True)
    (ckpt_dir / "train_report.json").write_text(json.dumps({"members": {f"s{m['seed']}": m["acc"] for m in members}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
