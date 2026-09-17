"""Compute hold+test hflip-TTA logits for IR seeds missing TTA (1,333,777,55)."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

JOBS = [
    (ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "pool_seed1.pt",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "hold_logits_pool_seed1_tta.npy",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "test_logits_seed1_tta.npy"),
    (ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "pool_seed333.pt",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "hold_logits_pool_seed333_tta.npy",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "test_logits_seed333_tta.npy"),
    (ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "pool_seed777.pt",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "hold_logits_pool_seed777_tta.npy",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "test_logits_seed777_tta.npy"),
    (ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "pool_seed55.pt",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "hold_logits_seed55_tta.npy",
     ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_seed55_tta.npy"),
]


def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


@torch.no_grad()
def eval_logits(model, loader, device, tta=False):
    model.eval()
    outs, ys = [], []

    def _fwd(x):
        # x: B,T,C,H,W float 0-1
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        return model(x).float()

    for x, y, _u, _i in loader:
        logits = _fwd(x)
        if tta:
            logits = 0.5 * (logits + _fwd(torch.flip(x, dims=[-1])))  # flip W
        outs.append(logits.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


@torch.no_grad()
def infer_cache(model, Xt, device, bs=6, tta=False):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)  # B,T,C,H,W
        x_n = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x_n).float()
        if tta:
            x_f = torch.flip(x, dims=[-1])
            x_fn = normalize(x_f).permute(0, 2, 1, 3, 4).contiguous()
            logits = 0.5 * (logits + model(x_fn).float())
        out[i:i + len(x)] = logits.cpu().numpy()
    return out


def main():
    free, total = torch.cuda.mem_get_info()
    used_mb = (total - free) / (1024 * 1024)
    print(f"GPU used~{used_mb:.0f}MB", flush=True)
    if used_mb > 2000:
        print("GPU busy; yield", flush=True)
        return
    device = torch.device("cuda")
    cache = ROOT / "cache" / "ir_yolo_v4"
    y = np.load(cache / "train_y.npy")
    users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=8, shuffle=False)
    y_ref = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_logits_v6.npz")["y"]
    meta = __import__("json").loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
    print(f"hold n={len(hold_idx)} test n={len(meta)}", flush=True)

    for ckpt, hold_p, test_p in JOBS:
        if hold_p.exists() and test_p.exists():
            lg = np.load(hold_p)
            print(f"skip {ckpt.name} tta_acc={(lg.argmax(1)==y_ref).mean():.4f}", flush=True)
            continue
        print(f"TTA {ckpt.name} ...", flush=True)
        t0 = time.time()
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build().to(device)
        model.load_state_dict(blob["model"])
        lb, _ = eval_logits(model, loader, device, tta=False)
        lg, _ = eval_logits(model, loader, device, tta=True)
        ab = float((lb.argmax(1) == y_ref).mean())
        at = float((lg.argmax(1) == y_ref).mean())
        print(f"  hold base={ab:.4f} tta={at:.4f} d={at-ab:+.4f}", flush=True)
        np.save(hold_p, lg.astype(np.float32))
        tl = infer_cache(model, Xt, device, bs=6, tta=True)
        np.save(test_p, tl.astype(np.float32))
        del model
        torch.cuda.empty_cache()
        print(f"  saved in {time.time()-t0:.1f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
