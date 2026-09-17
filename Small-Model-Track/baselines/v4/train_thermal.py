"""Gated Thermal probe: tiny CNN on sampled frames; fuse with MidFuse when present.

Holdout-first. Fallback to MidFuse logits when Thermal empty (~10 test clips).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "skeleton_imu_v2"))
sys.path.insert(0, str(ROOT))

from dataset import DEFAULT_HOLD_OUT_USERS, load_skel_train_cache  # noqa
from model import TinyThermalCNN, GatedThermalFuse, count_parameters  # noqa

TRAIN_ROOT = Path(r"D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\Thermal")
TEST_ROOT = Path(r"D:\CUHK-X\Small-Model-Track\Testing\data\small_model_track_test")
N_FRAMES = 8
IMG_SIZE = 64


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def sample_frames(folder: Path, n: int = N_FRAMES, size: int = IMG_SIZE) -> np.ndarray:
    """Return (n, H, W) float32 in [0,1]. Zeros if empty/missing."""
    if folder is None or not Path(folder).is_dir():
        return np.zeros((n, size, size), dtype=np.float32)
    files = sorted(
        list(Path(folder).glob("*.jpg"))
        + list(Path(folder).glob("*.png"))
        + list(Path(folder).glob("*.jpeg"))
    )
    if not files:
        return np.zeros((n, size, size), dtype=np.float32)
    idxs = np.linspace(0, len(files) - 1, num=n).astype(int)
    out = np.zeros((n, size, size), dtype=np.float32)
    for i, j in enumerate(idxs):
        try:
            im = Image.open(files[j]).convert("L").resize((size, size), Image.BILINEAR)
            out[i] = np.asarray(im, dtype=np.float32) / 255.0
        except Exception:
            pass
    return out


def thermal_dir_from_meta(meta_i: dict) -> Path:
    # meta pred_dir like ...\Skeleton\action\user\trial\predictions
    # Thermal parallel: ...\Thermal\action\user\trial
    pred = Path(meta_i["pred_dir"])
    # .../Skeleton/action/user/trial/predictions
    trial = pred.parent
    user = trial.parent
    action = user.parent
    return TRAIN_ROOT / action.name / user.name / trial.name


class ThermalDS(Dataset):
    def __init__(self, metas, y, indices, midfuse_logits, augment=False, seed=42):
        self.metas = metas
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.logits = midfuse_logits
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        # cache frames lazily in dict
        self._cache = {}

    def __len__(self):
        return len(self.indices)

    def _get_frames(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        td = thermal_dir_from_meta(self.metas[idx])
        fr = sample_frames(td)
        has = 1.0 if fr.sum() > 0 else 0.0
        self._cache[idx] = (fr, has)
        return fr, has

    def __getitem__(self, i):
        idx = int(self.indices[i])
        fr, has = self._get_frames(idx)
        fr = fr.copy()
        if self.augment and has > 0 and self.rng.rand() < 0.5:
            fr += self.rng.randn(*fr.shape).astype(np.float32) * 0.02
            fr = np.clip(fr, 0, 1)
        return (
            torch.from_numpy(self.logits[idx].astype(np.float32)),
            torch.from_numpy(fr),
            torch.tensor(has, dtype=torch.float32),
            int(self.y[idx]),
        )


@torch.no_grad()
def midfuse_train_logits(ckpt, X_skel, X_imu, has_imu, device, bs=64):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v2m", ROOT.parent / "skeleton_imu_v2" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = mod.build_model("midfuse", num_classes=40)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    n = len(X_skel)
    out = np.zeros((n, 40), dtype=np.float32)
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        xs = torch.from_numpy(X_skel[i0:i1]).to(device)
        xi = torch.from_numpy(X_imu[i0:i1]).to(device)
        fl = torch.from_numpy(has_imu[i0:i1].astype(np.float32)).to(device)
        out[i0:i1] = model(xs, xi, fl).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def run_epoch(model, loader, crit, opt, device, train):
    model.train(train)
    loss_sum = cor = n = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for logits, frames, has, y in loader:
            logits, frames, has, y = logits.to(device), frames.to(device), has.to(device), y.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            out = model(logits, frames, has)
            loss = crit(out, y)
            if train:
                loss.backward()
                opt.step()
            bs = y.size(0)
            loss_sum += loss.item() * bs
            cor += (out.argmax(1) == y).sum().item()
            n += bs
    return loss_sum / max(n, 1), cor / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midfuse-ckpt", default=str(ROOT.parent / "skeleton_imu_v2" / "checkpoints" / "best.pt"))
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints_thermal"))
    p.add_argument("--metrics-out", default=str(ROOT / "metrics_thermal.json"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    cache = Path(args.cache_dir)
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz")
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)

    print("computing MidFuse train logits...", flush=True)
    logits_path = Path(args.ckpt_dir) / "midfuse_train_logits.npy"
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    if logits_path.exists():
        mid_logits = np.load(logits_path)
    else:
        mid_logits = midfuse_train_logits(args.midfuse_ckpt, X_skel, X_imu, has_imu, device)
        np.save(logits_path, mid_logits)

    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr = np.where(~np.isin(users, list(hold)))[0]
    va = np.where(np.isin(users, list(hold)))[0]
    print(f"holdout n_train={len(tr)} n_val={len(va)}", flush=True)

    # quick check how many thermal present in holdout
    n_has = 0
    for i in va[:50]:
        td = thermal_dir_from_meta(meta[int(i)])
        if td.is_dir() and list(td.glob("*")):
            n_has += 1
    print(f"thermal present in first 50 holdout: {n_has}/50 path eg {thermal_dir_from_meta(meta[int(va[0])])}", flush=True)

    train_ds = ThermalDS(meta, y, tr, mid_logits, augment=True, seed=args.seed)
    val_ds = ThermalDS(meta, y, va, mid_logits, augment=False, seed=args.seed)
    # prefetch a bit by touching
    print("prefetching val thermal frames...", flush=True)
    for i in range(len(val_ds)):
        val_ds[i]
    print(f"val cache size={len(val_ds._cache)}", flush=True)

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GatedThermalFuse(num_classes=40).to(device)
    print(f"params={count_parameters(model)}", flush=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # baseline: midfuse-only on holdout
    with torch.no_grad():
        base_pred = mid_logits[va].argmax(1)
        base_acc = float((base_pred == y[va]).mean())
    print(f"midfuse-only holdout (from all-train logits leaked!) careful: using all-train midfuse on holdout is optimistic", flush=True)
    print(f"NOTE: for fair probe we should use holdout-trained midfuse. Checking...", flush=True)

    # Prefer holdout midfuse ckpt if exists
    hold_ckpt = ROOT.parent / "skeleton_imu_v2" / "checkpoints" / "best_holdout.pt"
    if hold_ckpt.exists():
        print(f"recomputing logits with {hold_ckpt}", flush=True)
        mid_logits = midfuse_train_logits(str(hold_ckpt), X_skel, X_imu, has_imu, device)
        train_ds = ThermalDS(meta, y, tr, mid_logits, augment=True, seed=args.seed)
        val_ds = ThermalDS(meta, y, va, mid_logits, augment=False, seed=args.seed)
        for i in range(len(val_ds)):
            val_ds[i]
        tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        base_acc = float((mid_logits[va].argmax(1) == y[va]).mean())
    print(f"midfuse holdout baseline acc={base_acc:.4f}", flush=True)

    best = -1.0
    best_path = Path(args.ckpt_dir) / "best_gated_thermal.pt"
    patience = args.patience
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_l, tr_a = run_epoch(model, tl, crit, opt, device, True)
        va_l, va_a = run_epoch(model, vl, crit, opt, device, False)
        hist.append({"epoch": ep, "train_acc": tr_a, "val_acc": va_a, "train_loss": tr_l, "val_loss": va_l})
        print(f"[thermal] ep {ep}/{args.epochs} train={tr_a:.4f} val={va_a:.4f} base={base_acc:.4f}", flush=True)
        if va_a >= best:
            best = va_a
            patience = args.patience
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": best,
                "base_acc": base_acc,
                "epoch": ep,
                "n_params": count_parameters(model),
            }, best_path)
        else:
            patience -= 1
            if patience <= 0:
                print("early stop", flush=True)
                break

    metrics = {
        "holdout_gated_thermal": float(best),
        "holdout_midfuse_baseline": float(base_acc),
        "beats_midfuse": bool(best > base_acc + 1e-6),
        "beats_v2_0537": bool(best > 0.5366336633663367),
        "history": hist,
        "elapsed_sec": time.time() - t0,
        "ckpt": str(best_path),
        "n_params": count_parameters(model),
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in metrics if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
