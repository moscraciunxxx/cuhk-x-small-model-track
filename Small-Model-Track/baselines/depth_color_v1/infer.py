"""Infer test set + MidFuse fallback for empty modality clips."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import NUM_CLASSES, discover_test_clips
from model import build_model, model_size_mb

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
MIDFUSE_SUB = TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv"
V8_SUB = TRACK / "baselines" / "v8" / "submission_v8.csv"


class TestCacheDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        arr = self.X[i].astype(np.float32) / 255.0  # T,H,W,C
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
        return x, i


def load_fallback_map(paths: list[Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                path = row["path"].replace("\\", "/")
                if not path.endswith("/"):
                    path += "/"
                out[path] = int(row["prediction"])
        print(f"fallback loaded {p.name}: {len(out)} rows")
        break  # first available
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color")
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    cache_dir = ROOT / "cache" / args.modality.lower()
    ckpt_dir = ROOT / "checkpoints" / args.modality.lower()
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / "holdout_train.pt"
    assert ckpt_path.exists(), ckpt_path

    test_meta = json.loads((cache_dir / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache_dir / "test_empty.json").read_text(encoding="utf-8")))
    tx_path = cache_dir / f"test_x_t{args.t}_s{args.size}.npy"
    X = np.memmap(tx_path, dtype=np.uint8, mode="r", shape=(len(test_meta), args.t, args.size, args.size, args.in_ch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(NUM_CLASSES, in_ch=args.in_ch).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    print(f"loaded {ckpt_path} val_f1={blob.get('val_f1')} size_mb={model_size_mb(model):.2f}")

    loader = DataLoader(TestCacheDataset(X), batch_size=args.batch_size, shuffle=False, num_workers=0)
    logits_all = np.zeros((len(test_meta), NUM_CLASSES), dtype=np.float32)
    for x, idxs in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()
        for row, i in zip(logits, idxs.numpy()):
            logits_all[int(i)] = row
    preds = logits_all.argmax(1)

    fb = load_fallback_map([MIDFUSE_SUB, V8_SUB, TRACK / "submission.csv"])
    n_fb = 0
    rows = []
    for i, meta in enumerate(test_meta):
        path = meta["path"]
        if not path.endswith("/"):
            path += "/"
        if meta.get("empty") or meta["sample_id"] in empty or meta.get("n_frames", 1) == 0:
            if path in fb:
                pred = fb[path]
                n_fb += 1
            else:
                pred = int(preds[i])  # last resort
        else:
            pred = int(preds[i])
        rows.append((path, pred))

    out = Path(args.out) if args.out else ROOT / f"submission_{args.modality.lower()}_v1.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for path, pred in rows:
            w.writerow([path, pred])
    np.save(ckpt_dir / "test_logits.npy", logits_all)
    print(f"wrote {out} n={len(rows)} fallback_used={n_fb} empty={len(empty)}")

    # also alias thermal_v1 naming if modality Depth? keep separate
    if args.promote:
        dest = TRACK / "submission.csv"
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print("promoted to", dest)

    # summary of class distribution
    from collections import Counter
    print("pred dist top10", Counter(p for _, p in rows).most_common(10))


if __name__ == "__main__":
    main()
