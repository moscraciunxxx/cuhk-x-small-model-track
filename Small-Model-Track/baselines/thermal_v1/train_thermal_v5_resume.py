"""Resume thermal v5 rethink from existing pool_seed2026.pt best state."""
from __future__ import annotations
import json, random, time
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
    return {"acc": float((yt == yp).mean()), "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "logits": np.concatenate(logits_all), "y": yt}

@torch.no_grad()
def infer_cache(model, Xt, device, bs=8):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i+bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i+len(x)] = model(x).float().cpu().numpy()
    return out

def main():
    cache = ROOT / "cache" / "thermal_yolo"
    ckpt_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink"
    ck = ckpt_dir / "pool_seed2026.pt"
    seed, epochs, patience, mixup_alpha, lr, batch_size = 2026, 42, 18, 0.1, 1e-4, 8
    # already unfroze at ep7; resume fully unfrozen
    start_ep = 11
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    best_acc = float(blob["val_acc"]); best_ep = int(blob.get("epoch", 7))
    history = list(blob.get("history") or [])
    print(f"resume from {ck.name} best={best_acc:.4f}@ep{best_ep} start_ep={start_ep} device={device}", flush=True)

    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    set_seed(seed + start_ep)
    model = build(False).to(device)
    model.load_state_dict(blob["model"])
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - start_ep + 1, 1))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True),
        num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tag = f"pool_seed{seed}"
    t0 = time.time()
    for ep in range(start_ep, epochs + 1):
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous(); yb = yb.to(device)
            if mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb_m = lam * xb + (1.0 - lam) * xb[idx]; y1, y2 = yb, yb[idx]
            else:
                lam, xb_m, y1, y2 = 1.0, xb, yb, yb
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb_m)
                loss = lam * crit(logits, y1) + (1.0 - lam) * crit(logits, y2) if lam < 1 else crit(logits, yb)
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
        print(f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} val={row['val_acc']:.4f} f1={row['val_f1']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc, "epoch": ep, "seed": seed,
                        "tag": tag, "history": history, "hold_logits": metrics["logits"], "hold_y": metrics["y"],
                        "recipe": {"epochs": epochs, "mixup": mixup_alpha, "unfreeze_ep": 7, "patience": patience, "lr": lr, "label_smoothing": 0.05, "resumed_from": start_ep}}, ck)
            print(f"  saved {ck.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}@ep{best_ep}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f}@ep{best_ep} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    hold_m = evaluate(model, val_loader, device)
    # refresh ckpt hold logits
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    blob["hold_logits"] = hold_m["logits"]; blob["hold_y"] = hold_m["y"]; blob["val_acc"] = best_acc
    blob["history"] = history; blob["epoch"] = best_ep
    torch.save(blob, ck)

    hold_ens = hold_m["logits"]; yt = y[hold_idx]
    meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    keys = np.array([(int(meta[i]["user_id"]), int(meta[i]["label"]), str(meta[i].get("trial", ""))) for i in hold_idx], dtype=object)
    np.savez(ckpt_dir / "holdout_ensemble_logits.npz", logits=hold_ens, stack=hold_ens[None], y=yt,
             users=users[hold_idx], hold_idx=hold_idx, keys=keys, tags=np.array([tag]), scores=np.array([best_acc]))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(405, 16, 112, 112, 3))
    test = infer_cache(model, Xt, device, bs=8)
    np.save(ckpt_dir / "test_logits_seed2026.npy", test)
    np.save(ckpt_dir / "test_logits.npy", test)
    metrics = {
        "tag": "thermal_v5_rethink", "cache": str(cache),
        "member_scores": {"seed2026": best_acc},
        "holdout_acc_ensemble": float((hold_ens.argmax(1) == yt).mean()),
        "holdout_acc_best_member": float(best_acc), "n_hold": int(len(yt)),
        "prior_v2_trio_ens": 0.6031746031746031, "mixup": mixup_alpha, "unfreeze_ep": 7,
        "epochs": epochs, "patience": patience, "best_ep": best_ep,
        "delta_ens_vs_v2_trio": float((hold_ens.argmax(1) == yt).mean() - 0.6031746031746031),
        "resumed": True, "start_ep": start_ep,
    }
    (ckpt_dir / "metrics_thermal_v5.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "metrics_thermal_v5.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
