"""Fine-tune 4-channel Depth+IR R(2+1)D-34; dump hold/test logits; pack int8 <=100MB."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES
from model_r2p1d34 import ARCH_NAME, IN_CH, assert_r2p1d34_4ch, build_r2p1d34_4ch, pack_int8

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645, 0.401092]).view(1, 1, 4, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989, 0.222156]).view(1, 1, 4, 1, 1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def freeze_early(model, freeze: bool):
    for name, p in model.named_parameters():
        if any(k in name for k in ("stem", "layer1")):
            p.requires_grad = not freeze


@torch.no_grad()
def infer_logits(model, X, indices, device, bs):
    model.eval()
    out = np.zeros((len(indices), NUM_CLASSES), np.float32)
    dummy_y = np.zeros(X.shape[0], np.int64)
    dummy_u = np.zeros(X.shape[0], np.int64)
    loader = DataLoader(
        CachedClipDataset(X, dummy_y, dummy_u, indices, train=False, seed=0),
        batch_size=bs, shuffle=False, num_workers=0,
    )
    ptr = 0
    for x, _y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x).float()
        n = logits.size(0)
        out[ptr:ptr + n] = logits.cpu().numpy()
        ptr += n
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return float((y_true == y_pred).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "depth_ir_4ch_v31"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--unfreeze-ep", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--ckpt-dir", default="", help="default checkpoints/depth_ir_r2p1d34_v31")
    ap.add_argument("--init-ckpt", default="", help="optional best.pt to fine-tune from (same 4ch R34)")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache_dir)
    y = np.load(cache / "train_y.npy")
    users = np.load(cache / "train_users.npy")
    n = len(y)
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                  shape=(n, args.t, args.size, args.size, IN_CH))
    if X.shape[-1] != 4:
        raise SystemExit(f"cache last dim {X.shape[-1]} != 4")
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={n} hold={len(hold_idx)} pool={len(pool_idx)} device={device} shape={tuple(X.shape)}", flush=True)

    print("building R(2+1)D-34 4ch (download IG-65M/Kinetics if needed)", flush=True)
    model = build_r2p1d34_4ch(pretrained=not args.no_pretrained, progress=True)
    ident = assert_r2p1d34_4ch(model)
    print("arch_ident", json.dumps(ident), flush=True)
    if args.init_ckpt:
        blob = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f"loaded init_ckpt={args.init_ckpt} missing={missing} unexpected={unexpected}", flush=True)
    model = model.to(device)
    freeze_early(model, False if args.init_ckpt else True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True)
    train_loader = DataLoader(
        CachedClipDataset(X, y, users, pool_idx, train=True, seed=args.seed),
        batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        CachedClipDataset(X, y, users, hold_idx, train=False, seed=args.seed),
        batch_size=max(args.batch_size, 2), shuffle=False, num_workers=0,
    )

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (ROOT / "checkpoints" / "depth_ir_r2p1d34_v31")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"ckpt_dir={ckpt_dir} seed={args.seed}", flush=True)
    best_acc, best_ep, stale = -1.0, 0, 0
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if ep == args.unfreeze_ep:
            print("unfreeze stem+layer1", flush=True)
            freeze_early(model, False)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
        model.train()
        opt.zero_grad(set_to_none=True)
        total, nstep = 0.0, 0
        for step, (x, yy, _u, _i) in enumerate(train_loader, 1):
            x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yy = yy.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), yy) / args.accum
            scaler.scale(loss).backward()
            if step % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item()) * args.accum
            nstep += 1
        sched.step()
        acc = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "loss": total / max(nstep, 1), "val_acc": acc})
        print(f"ep {ep:02d} loss={total / max(nstep, 1):.4f} hold_acc={acc:.4f} min={(time.time() - t0) / 60:.1f}", flush=True)
        if acc > best_acc + 1e-6:
            best_acc, best_ep, stale = acc, ep, 0
            torch.save({
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "val_acc": acc, "epoch": ep, "arch": ARCH_NAME, "in_ch": IN_CH,
                "ident": ident, "history": history,
            }, ckpt_dir / "best.pt")
            print(f"saved best.pt val_acc={acc:.4f}", flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print("early stop", flush=True)
                break

    blob = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model = model.to(device)
    print("infer hold+test logits", flush=True)
    hold_logits = infer_logits(model, X, hold_idx, device, bs=max(args.batch_size, 2))
    np.save(ckpt_dir / "hold_logits.npy", hold_logits)
    np.save(ckpt_dir / "hold_idx.npy", hold_idx)
    np.save(ckpt_dir / "hold_y.npy", y[hold_idx])
    np.save(ckpt_dir / "hold_users.npy", users[hold_idx])
    meta_te = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta_te), args.t, args.size, args.size, IN_CH))
    test_idx = np.arange(len(meta_te))
    test_logits = infer_logits(model, Xt, test_idx, device, bs=max(args.batch_size, 2))
    np.save(ckpt_dir / "test_logits.npy", test_logits)

    packed = pack_int8(blob["model"])
    pack_path = ckpt_dir / "model_int8.pt"
    torch.save({
        "schema": "r2p1d34-4ch-int8/v1",
        "arch": ARCH_NAME,
        "in_ch": IN_CH,
        "layers": [3, 4, 6, 3],
        "val_acc": blob["val_acc"],
        "packed": packed,
    }, pack_path)
    yolo = ROOT / "yolov8n.pt"
    pack_mb = (pack_path.stat().st_size + (yolo.stat().st_size if yolo.exists() else 0)) / (1024 * 1024)
    report = {
        "arch": ARCH_NAME, "in_ch": IN_CH, "ident": ident,
        "best_hold_acc": float(blob["val_acc"]), "best_epoch": int(blob["epoch"]),
        "hold_logits": str(ckpt_dir / "hold_logits.npy"),
        "test_logits": str(ckpt_dir / "test_logits.npy"),
        "hold_n": int(hold_logits.shape[0]), "test_n": int(test_logits.shape[0]),
        "pack": str(pack_path), "pack_plus_yolo_mb": pack_mb,
        "history": history,
    }
    (ckpt_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("arch", "in_ch", "best_hold_acc", "hold_n", "test_n", "pack_plus_yolo_mb")}), flush=True)
    print("WROTE_LOGITS hold+test", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
