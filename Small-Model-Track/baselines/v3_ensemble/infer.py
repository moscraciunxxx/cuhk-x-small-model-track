"""Inference for skeleton / skeleton+IMU v2 -> submission CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import DEFAULT_T, IMU_DIM, SKEL_DIM, load_skel_test_cache
from model import build_model

ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLE = Path(
    r"D:\CUHK-X\Small-Model-Track\Testing\test_file\sample_submission.csv"
)
DEFAULT_CKPT = ROOT / "checkpoints" / "best.pt"
DEFAULT_OUT = ROOT / "submission_skeleton_imu_v2.csv"
DEFAULT_CACHE = ROOT / "cache"


class DualTestDS(Dataset):
    def __init__(self, Xs, Xi, has, paths):
        self.Xs = Xs
        self.Xi = Xi
        self.has = has
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return (
            torch.from_numpy(np.asarray(self.Xs[i], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.Xi[i], dtype=np.float32)),
            float(self.has[i]),
            self.paths[i],
        )


class SkelTestDS(Dataset):
    def __init__(self, Xs, paths):
        self.Xs = Xs
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.Xs[i], dtype=np.float32)), self.paths[i]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-csv", type=str, default=str(DEFAULT_SAMPLE))
    p.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "midfuse")
    num_classes = ckpt.get("num_classes", 40)
    dual = bool(ckpt.get("dual", model_name == "midfuse"))
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(
        f"Loaded {args.ckpt} model={model_name} dual={dual} "
        f"val_acc={ckpt.get('val_acc')} params={ckpt.get('n_params')}"
    )

    cache = Path(args.cache_dir)
    Xs, paths = load_skel_test_cache(cache)
    pred_map = {}

    if dual and (cache / "imu_test.npz").exists():
        imu = np.load(cache / "imu_test.npz", allow_pickle=True)
        Xi = imu["X"]
        has = imu["has_imu"].astype(np.float32)
        ds = DualTestDS(Xs, Xi, has, paths)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for xs, xi, flag, batch_paths in tqdm(loader, desc="infer"):
                logits = model(
                    xs.to(device), xi.to(device), flag.to(device)
                )
                preds = logits.argmax(1).cpu().tolist()
                for path, pred in zip(batch_paths, preds):
                    pred_map[path] = int(pred)
    else:
        if dual:
            print("No imu_test.npz — skeleton-only forward with zero IMU")
            Xi = np.zeros((len(paths), Xs.shape[1], IMU_DIM), dtype=np.float32)
            has = np.zeros((len(paths),), dtype=np.float32)
            ds = DualTestDS(Xs, Xi, has, paths)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            with torch.no_grad():
                for xs, xi, flag, batch_paths in tqdm(loader, desc="infer"):
                    logits = model(xs.to(device), xi.to(device), flag.to(device))
                    preds = logits.argmax(1).cpu().tolist()
                    for path, pred in zip(batch_paths, preds):
                        pred_map[path] = int(pred)
        else:
            ds = SkelTestDS(Xs, paths)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            with torch.no_grad():
                for x, batch_paths in tqdm(loader, desc="infer"):
                    logits = model(x.to(device))
                    preds = logits.argmax(1).cpu().tolist()
                    for path, pred in zip(batch_paths, preds):
                        pred_map[path] = int(pred)

    sample = pd.read_csv(args.sample_csv)
    rows = []
    missing = 0
    for path in sample["path"].tolist():
        if path not in pred_map:
            alt = path if path.endswith("/") else path + "/"
            if alt in pred_map:
                pred_map[path] = pred_map[alt]
            else:
                missing += 1
                pred_map[path] = 0
        rows.append({"path": path, "prediction": pred_map[path]})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} rows={len(out_df)} missing_filled={missing}")


if __name__ == "__main__":
    main()
