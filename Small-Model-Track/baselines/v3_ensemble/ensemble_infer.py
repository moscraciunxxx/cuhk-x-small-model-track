"""Average softmax from fold checkpoints for a stronger submission."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import load_skel_test_cache, IMU_DIM
from model import build_model
from infer import DualTestDS, SkelTestDS

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    p.add_argument("--sample-csv", type=str, default=r"D:\CUHK-X\Small-Model-Track\Testing\test_file\sample_submission.csv")
    p.add_argument("--out", type=str, default=str(ROOT / "submission_skeleton_imu_v2_ensemble.csv"))
    p.add_argument("--pattern", type=str, default="best_fold*.pt")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(Path(args.ckpt_dir).glob(args.pattern))
    if not ckpts:
        raise SystemExit(f"No ckpts matching {args.pattern} in {args.ckpt_dir}")
    print("Ensemble:", [c.name for c in ckpts])

    cache = Path(args.cache_dir)
    Xs, paths = load_skel_test_cache(cache)
    imu_path = cache / "imu_test.npz"
    dual = False
    if imu_path.exists():
        imu = np.load(imu_path, allow_pickle=True)
        Xi, has = imu["X"], imu["has_imu"].astype(np.float32)
        dual = True
        ds = DualTestDS(Xs, Xi, has, paths)
    else:
        ds = SkelTestDS(Xs, paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # accumulate probs
    n = len(paths)
    probs = None
    for ck in ckpts:
        blob = torch.load(ck, map_location=device, weights_only=False)
        model = build_model(blob.get("model_name", "midfuse"), num_classes=blob.get("num_classes", 40))
        model.load_state_dict(blob["model_state"])
        model.to(device).eval()
        use_dual = bool(blob.get("dual", blob.get("model_name") == "midfuse"))
        preds = []
        with torch.no_grad():
            if use_dual and dual:
                for xs, xi, flag, _bp in tqdm(loader, desc=ck.stem):
                    logits = model(xs.to(device), xi.to(device), flag.to(device))
                    preds.append(torch.softmax(logits, dim=1).cpu())
            else:
                # skeleton path
                if dual:
                    # DualTestDS but model wants skel only
                    for xs, xi, flag, _bp in tqdm(loader, desc=ck.stem):
                        logits = model(xs.to(device))
                        preds.append(torch.softmax(logits, dim=1).cpu())
                else:
                    for x, _bp in tqdm(loader, desc=ck.stem):
                        logits = model(x.to(device))
                        preds.append(torch.softmax(logits, dim=1).cpu())
        P = torch.cat(preds, dim=0).numpy()
        probs = P if probs is None else probs + P
        print(ck.name, "val_acc", blob.get("val_acc"))
    probs /= len(ckpts)
    pred_ids = probs.argmax(1)

    pred_map = {paths[i]: int(pred_ids[i]) for i in range(n)}
    sample = pd.read_csv(args.sample_csv)
    rows = []
    for path in sample["path"]:
        rows.append({"path": path, "prediction": pred_map.get(path, pred_map.get(path if path.endswith("/") else path+"/", 0))})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
