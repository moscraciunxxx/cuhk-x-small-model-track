"""Inference: write submission CSV matching sample_submission paths."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from dataset import (
    DEFAULT_T,
    FEAT_DIM,
    discover_test_clips,
    load_sequence,
    load_test_cache,
    normalize_sequence,
    resample_sequence,
)
from model import build_model

ROOT = Path(__file__).resolve().parent
DEFAULT_TEST = Path(
    r"D:\CUHK-X\Small-Model-Track\Testing\data\small_model_track_test"
)
DEFAULT_SAMPLE = Path(
    r"D:\CUHK-X\Small-Model-Track\Testing\test_file\sample_submission.csv"
)
DEFAULT_CKPT = ROOT / "checkpoints" / "best.pt"
DEFAULT_OUT = ROOT / "submission_skeleton_v1.csv"
DEFAULT_CACHE = ROOT / "cache"


class TestSkeletonDataset(Dataset):
    def __init__(self, clips, T: int = DEFAULT_T, normalize: bool = True):
        self.clips = clips
        self.T = T
        self.normalize = normalize

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        if c["pred_dir"] is None:
            feat = torch.zeros(self.T, FEAT_DIM, dtype=torch.float32)
        else:
            seq = load_sequence(Path(c["pred_dir"]))
            seq = resample_sequence(seq, self.T)
            if self.normalize:
                seq = normalize_sequence(seq)
            feat = torch.from_numpy(seq.reshape(self.T, FEAT_DIM))
        return feat, c["path"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=str, default=str(DEFAULT_TEST))
    p.add_argument("--sample-csv", type=str, default=str(DEFAULT_SAMPLE))
    p.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "conv1d")
    num_classes = ckpt.get("num_classes", 40)
    T = ckpt.get("T", DEFAULT_T)
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded {args.ckpt} model={model_name} val_acc={ckpt.get('val_acc')} T={T}")

    cache_path = Path(args.cache_dir) / "test.npz"
    pred_map = {}
    if (not args.no_cache) and cache_path.exists():
        X, paths = load_test_cache(Path(args.cache_dir))
        print(f"Using test cache {cache_path} n={len(paths)}")
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds_all = []
        with torch.no_grad():
            for (xb,) in tqdm(loader, desc="infer"):
                logits = model(xb.to(device))
                preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        for path, pred in zip(paths, preds_all):
            pred_map[path] = int(pred)
    else:
        clips = discover_test_clips(Path(args.test_root))
        print(f"Found {len(clips)} test clips; missing skeleton: "
              f"{sum(1 for c in clips if c['pred_dir'] is None)}")
        ds = TestSkeletonDataset(clips, T=T, normalize=True)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        with torch.no_grad():
            for x, paths in tqdm(loader, desc="infer"):
                logits = model(x.to(device))
                preds = logits.argmax(dim=1).cpu().tolist()
                for path, pred in zip(paths, preds):
                    pred_map[path] = int(pred)

    sample = pd.read_csv(args.sample_csv)
    out_rows = []
    missing = 0
    for path in sample["path"].tolist():
        if path not in pred_map:
            alt = path if path.endswith("/") else path + "/"
            alt2 = path.rstrip("/") + "/"
            if alt in pred_map:
                pred_map[path] = pred_map[alt]
            elif alt2 in pred_map:
                pred_map[path] = pred_map[alt2]
            else:
                missing += 1
                pred_map[path] = 0
        out_rows.append({"path": path, "prediction": pred_map[path]})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} rows={len(out_df)} missing_filled={missing}")
    print(out_df.head())


if __name__ == "__main__":
    main()
