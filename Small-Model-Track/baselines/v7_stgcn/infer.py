"""Inference + optional submission for v7 ST-GCN."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import load_imu_caches, load_skel_test_cache
from model import build_model

ROOT = Path(__file__).resolve().parent
TRACK = ROOT.parent.parent


class TestDual(Dataset):
    def __init__(self, X_skel, X_imu, has_imu=None):
        self.X_skel = X_skel
        self.X_imu = X_imu
        self.has_imu = (
            has_imu if has_imu is not None else np.ones(len(X_skel), dtype=np.bool_)
        )

    def __len__(self):
        return len(self.X_skel)

    def __getitem__(self, i):
        xs = torch.from_numpy(np.asarray(self.X_skel[i], dtype=np.float32))
        xi = torch.from_numpy(np.asarray(self.X_imu[i], dtype=np.float32))
        flag = float(self.has_imu[i])
        return xs, xi, flag


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt.get("model_name", "stgcn_fuse")
    model = build_model(name, num_classes=ckpt.get("num_classes", 40))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_logits(model, loader, device, dual: bool):
    outs = []
    for batch in loader:
        if dual:
            xs, xi, flag = batch
            xs = xs.to(device)
            xi = xi.to(device)
            flag = flag.to(device)
            logits = model(xs, xi, flag)
        else:
            x = batch[0].to(device)
            logits = model(x)
        outs.append(logits.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    p.add_argument("--out", type=str, default=str(ROOT / "submission_v7_stgcn.csv"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, ckpt = load_model(Path(args.ckpt), device)
    dual = bool(ckpt.get("dual", True))
    cache = Path(args.cache_dir)
    X_te, paths = load_skel_test_cache(cache)
    if dual:
        _, _, X_imu_te, has_imu_te = load_imu_caches(cache)
        ds = TestDual(X_te, X_imu_te, has_imu_te)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    else:
        class SkelOnly(Dataset):
            def __init__(self, X):
                self.X = X

            def __len__(self):
                return len(self.X)

            def __getitem__(self, i):
                return (torch.from_numpy(np.asarray(self.X[i], dtype=np.float32)),)

        loader = DataLoader(SkelOnly(X_te), batch_size=args.batch_size, shuffle=False)

    logits = predict_logits(model, loader, device, dual)
    pred = logits.argmax(1)
    df = pd.DataFrame({"path": list(paths), "prediction": [int(p) for p in pred]})
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"Wrote {out} n={len(df)}", flush=True)
    np.savez_compressed(out.with_suffix(".npz"), logits=logits, paths=np.array(paths, dtype=object))
    print(f"Wrote logits -> {out.with_suffix('.npz')}", flush=True)


if __name__ == "__main__":
    main()
