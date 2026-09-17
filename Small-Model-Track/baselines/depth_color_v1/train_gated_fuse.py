"""Dump MidFuse test logits + train gated Depth/Thermal + MidFuse fuse (optimize accuracy)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V2 = TRACK / "baselines" / "skeleton_imu_v2"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(ROOT))

from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES  # noqa: E402

HOLD = set(DEFAULT_HOLD_OUT_USERS)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


class GatedVideoFuse(nn.Module):
    def __init__(self, video: nn.Module, init_alpha: float = 0.0):
        super().__init__()
        self.video = video
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, frames, mid_logits, has_video=None):
        v = self.video(frames)
        a = torch.sigmoid(self.alpha)
        if has_video is None:
            return mid_logits + a * v
        flag = has_video.view(-1, 1).to(v.dtype)
        return mid_logits + a * flag * v


class FuseDS(Dataset):
    def __init__(self, X, y, users, mid, indices, train=False, seed=0):
        self.inner = CachedClipDataset(X, y, users, indices, train=train, seed=seed)
        self.mid = mid
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        x, y, uid, idx = self.inner[i]
        # idx is global cache index
        return x, torch.from_numpy(self.mid[idx].astype(np.float32)), y, uid, idx


def dump_midfuse_test_logits(out_path: Path, device):
    import importlib
    import importlib.util

    # Ensure skeleton_imu_v2 dataset/model resolve, not local video dataset
    sys.path = [str(V2)] + [p for p in sys.path if Path(p).resolve() != ROOT.resolve()]
    for mod in ("dataset", "model"):
        if mod in sys.modules:
            del sys.modules[mod]
    import model as v2m  # noqa

    cache = V2 / "cache"
    skel = np.load(cache / "skel_test.npz", allow_pickle=True)
    imu = np.load(cache / "imu_test.npz", allow_pickle=True)
    Xs = skel["X"]
    Xi = imu["X"]
    flag = imu["has_imu"]
    print("Xs", Xs.shape, "Xi", Xi.shape, "has_imu", flag.shape)

    ckpt = V2 / "checkpoints_midfuse_s123" / "best_all_train.pt"
    if not ckpt.exists():
        ckpt = V2 / "checkpoints" / "best_all_train.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = v2m.build_model(blob.get("model_name", "midfuse"), num_classes=int(blob.get("num_classes", 40)))
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()

    n = len(Xs)
    logits = np.zeros((n, 40), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for i in range(0, n, bs):
            xs = torch.from_numpy(np.asarray(Xs[i : i + bs], dtype=np.float32)).to(device)
            xi = torch.from_numpy(np.asarray(Xi[i : i + bs], dtype=np.float32)).to(device)
            f = torch.from_numpy(np.asarray(flag[i : i + bs], dtype=np.float32)).to(device)
            out = model(xs, xi, f)
            logits[i : i + bs] = out.cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, logits)
    print("saved midfuse test logits", out_path, logits.shape)

    # restore path for local video modules
    sys.path = [str(ROOT), str(V2)] + [p for p in sys.path if p not in (str(ROOT), str(V2))]
    for mod in ("dataset", "model"):
        if mod in sys.modules:
            del sys.modules[mod]
    return logits


def build_video(arch: str):
    if arch == "r2p1d":
        from model_r2p1d import build_model_r2p1d, model_size_mb, count_parameters

        return build_model_r2p1d(40, in_ch=3, base=48), model_size_mb, count_parameters
    from model import build_model, model_size_mb, count_parameters

    return build_model(40, in_ch=3), model_size_mb, count_parameters


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = n = 0
    ys, preds = [], []
    for frames, mid, y, _u, _i in loader:
        frames = frames.to(device)
        mid = mid.to(device)
        y = y.to(device)
        logits = model(frames, mid)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        n += len(y)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    from sklearn.metrics import f1_score

    return {
        "acc": correct / max(n, 1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color")
    ap.add_argument("--arch", default="r2p1d", choices=["r2p1d", "cnn_gru"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--dump-midfuse-only", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--alpha-init", type=float, default=0.0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mid_test_path = ROOT / "cache" / "midfuse_test_logits.npy"
    if args.dump_midfuse_only or not mid_test_path.exists():
        dump_midfuse_test_logits(mid_test_path, device)
        if args.dump_midfuse_only:
            return

    mid_train = np.load(TRACK / "baselines" / "v4" / "checkpoints_thermal" / "midfuse_train_logits.npy")
    cache_dir = ROOT / "cache" / args.modality.lower()
    y = np.load(cache_dir / "train_y.npy")
    users = np.load(cache_dir / "train_users.npy")
    X = np.memmap(
        cache_dir / f"train_x_t{args.t}_s{args.size}.npy",
        dtype=np.uint8,
        mode="r",
        shape=(len(y), args.t, args.size, args.size, 3),
    )
    assert len(mid_train) == len(y)

    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]

    # MidFuse-only holdout baseline accuracy
    mid_hold_pred = mid_train[hold_idx].argmax(1)
    mid_acc = float((mid_hold_pred == y[hold_idx]).mean())
    print(f"MidFuse holdout acc={mid_acc:.4f}")

    video, size_fn, count_fn = build_video(args.arch)
    model = GatedVideoFuse(video, init_alpha=args.alpha_init).to(device)
    print(f"params~{count_fn(video)} video_mb~{size_fn(video):.2f}")

    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True)
    train_loader = DataLoader(
        FuseDS(X, y, users, mid_train, pool_idx, train=True, seed=args.seed),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        FuseDS(X, y, users, mid_train, hold_idx, train=False, seed=args.seed),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    ckpt_dir = ROOT / "checkpoints" / f"gated_{args.modality.lower()}_{args.arch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_state, best_ep, history = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = n = 0
        for frames, mid, yy, _u, _i in train_loader:
            frames, mid, yy = frames.to(device), mid.to(device), yy.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(frames, mid)
                loss = crit(logits, yy)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * len(yy)
            correct += int((logits.argmax(1) == yy).sum().item())
            n += len(yy)
        sched.step()
        metrics = eval_acc(model, val_loader, device)
        row = {
            "epoch": ep,
            "tr_loss": loss_sum / max(n, 1),
            "tr_acc": correct / max(n, 1),
            "val_acc": metrics["acc"],
            "val_f1": metrics["macro_f1"],
            "alpha": float(torch.sigmoid(model.alpha).item()),
        }
        history.append(row)
        print(
            f"[gated] ep{ep:03d} loss={row['tr_loss']:.4f} tr_acc={row['tr_acc']:.3f} "
            f"val_acc={row['val_acc']:.4f} val_f1={row['val_f1']:.4f} a={row['alpha']:.3f} "
            f"(mid={mid_acc:.4f})",
            flush=True,
        )
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "val_acc": best_acc,
                    "mid_acc": mid_acc,
                    "epoch": ep,
                    "args": vars(args),
                    "alpha": row["alpha"],
                },
                ckpt_dir / "holdout_train.pt",
            )
        if args.patience > 0 and ep - best_ep >= args.patience:
            print(f"early stop ep{ep} best_acc={best_acc:.4f}", flush=True)
            break

    print(f"BEST gated holdout acc={best_acc:.4f} mid={mid_acc:.4f} delta={best_acc-mid_acc:+.4f} took={time.time()-t0:.1f}s", flush=True)
    (ckpt_dir / "metrics.json").write_text(
        json.dumps({"best_acc": best_acc, "mid_acc": mid_acc, "history": history, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )

    # Infer if requested or always produce CSV when best beats mid or anyway
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    mid_test = np.load(mid_test_path)
    test_meta = json.loads((cache_dir / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache_dir / "test_empty.json").read_text(encoding="utf-8")))
    Xt = np.memmap(
        cache_dir / f"test_x_t{args.t}_s{args.size}.npy",
        dtype=np.uint8,
        mode="r",
        shape=(len(test_meta), args.t, args.size, args.size, 3),
    )
    assert len(mid_test) == len(test_meta)
    preds = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_meta), bs):
            arr = Xt[i : i + bs].astype(np.float32) / 255.0
            frames = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
            mid = torch.from_numpy(mid_test[i : i + bs]).to(device)
            has = torch.tensor(
                [0.0 if (test_meta[j].get("empty") or test_meta[j]["sample_id"] in empty) else 1.0 for j in range(i, min(i + bs, len(test_meta)))],
                device=device,
            )
            logits = model(frames, mid, has)
            preds.extend(logits.argmax(1).cpu().tolist())

    out = ROOT / f"submission_gated_{args.modality.lower()}_{args.arch}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for meta, pred in zip(test_meta, preds):
            path = meta["path"] if meta["path"].endswith("/") else meta["path"] + "/"
            w.writerow([path, int(pred)])
    print("wrote", out, "best_acc", best_acc)

    # Promote only if clearly beats MidFuse holdout (~0.54) OR at least beats prior video-only
    promote = best_acc >= mid_acc + 0.005
    if promote:
        dest = TRACK / "submission.csv"
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print("PROMOTED to", dest)
    else:
        print("NOT promoted (gated did not clearly beat MidFuse holdout)")


if __name__ == "__main__":
    main()
