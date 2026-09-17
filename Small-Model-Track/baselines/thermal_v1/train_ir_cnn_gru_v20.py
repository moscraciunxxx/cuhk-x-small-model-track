"""ir_v20 fallback: compact 2D-CNN+GRU on IR YOLO-v4 crops (from depth_color_v1 TemporalCNNHAR).
Use if S3D unavailable/OOM. ~few MB; no Kinetics pretrain.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
sys.path.insert(0, str(TRACK / "baselines" / "depth_color_v1"))
from model import TemporalCNNHAR, build_model  # noqa: E402
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES  # noqa: E402

# Prefer thermal_v1 dataset (same cache API) over depth_color's
sys.path.insert(0, str(ROOT))
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES  # noqa: E402,F811

HOLD = set(DEFAULT_HOLD_OUT_USERS)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        # CachedClipDataset -> (B,T,C,H,W); TemporalCNNHAR expects same
        x = x.to(device)
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {"acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "logits": np.concatenate(logits_all), "y": yt}


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience, ckpt_path):
    set_seed(seed)
    tag = f"seed{seed}"
    model = build_model(num_classes=NUM_CLASSES, in_ch=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=WeightedRandomSampler(w, num_samples=len(pool_idx), replacement=True),
                              num_workers=0, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_amp)

    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb); loss = crit(logits, yb)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                logits = model(xb); loss = crit(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
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
                        "hold_y": metrics["y"], "arch": "cnn_gru"}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state); model.to(device)
    hold_m = evaluate(model, val_loader, device)
    if use_amp:
        del model; torch.cuda.empty_cache()
    return best_acc, hold_m["logits"], hold_m["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_cnn_gru_v20"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="Allow CPU train (slow); default requires CUDA")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA required (pass --allow-cpu to override)")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), args.t, args.size, args.size, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device}", flush=True)
    m0 = build_model(NUM_CLASSES, 3)
    n_params = sum(p.numel() for p in m0.parameters())
    print(f"CNN+GRU params={n_params} fp16_mb~{n_params*2/1e6:.2f}", flush=True)

    members = []
    for seed in args.seeds:
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            acc = float(blob.get("val_acc", -1))
            print(f"reuse {ck.name} val={acc}", flush=True)
            members.append({"seed": seed, "acc": acc, "hold_logits": np.asarray(blob["hold_logits"]), "ckpt": ck})
            continue
        if args.skip_train:
            print(f"missing {ck}", flush=True); continue
        acc, hlog, hy = train_one(X, y, users, pool_idx, hold_idx, device, seed, args.epochs,
                                  args.batch_size, args.lr, args.patience, ck)
        members.append({"seed": seed, "acc": acc, "hold_logits": hlog, "ckpt": ck})

    if not members:
        raise SystemExit("no CNN+GRU members")
    hold_stack = np.stack([m["hold_logits"] for m in members], 0)
    hold_ens = hold_stack.mean(0)
    yt = y[hold_idx]
    print(f"cnn_gru members: {[(m['seed'], round(m['acc'],4)) for m in members]}", flush=True)
    print(f"cnn_gru ens hold={(hold_ens.argmax(1)==yt).mean():.4f}", flush=True)
    meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    keys = np.array([(int(meta[i]["user_id"]), int(meta[i]["label"]), str(meta[i].get("trial","")))
                     for i in hold_idx], dtype=object)
    np.savez(ckpt_dir / "hold_logits_strong.npz",
             logits=hold_stack, ens=hold_ens, y=yt, users=users[hold_idx],
             hold_idx=hold_idx, keys=keys,
             seeds=np.array([m["seed"] for m in members]),
             scores=np.array([m["acc"] for m in members]))
    metrics = {
        "tag": "ir_cnn_gru_v20",
        "arch": "cnn_gru",
        "member_scores": {f"seed{m['seed']}": m["acc"] for m in members},
        "holdout_acc_ensemble": float((hold_ens.argmax(1) == yt).mean()),
        "n_hold": int(len(yt)),
        "fp16_mb_approx": round(n_params * 2 / 1e6, 2),
    }
    (ckpt_dir / "metrics_ir_cnn_gru_v20.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
