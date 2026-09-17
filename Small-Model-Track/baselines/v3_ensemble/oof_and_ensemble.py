"""OOF accuracy from fold ckpts + test softmax ensemble submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CachedDualDataset, load_skel_train_cache, load_skel_test_cache, IMU_DIM
from model import build_model
from infer import DualTestDS, SkelTestDS

ROOT = Path(__file__).resolve().parent


def eval_loader(model, loader, device, dual: bool):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if dual:
                xs, xi, y, _u, flag = batch
                logits = model(xs.to(device), xi.to(device), flag.to(device))
            else:
                x, y, _u = batch
                logits = model(x.to(device))
            preds.append(torch.softmax(logits, dim=1).cpu())
            labels.append(y)
    P = torch.cat(preds, dim=0).numpy()
    Y = torch.cat(labels, dim=0).numpy()
    acc = float((P.argmax(1) == Y).mean())
    return P, Y, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default=str(ROOT / ".." / "skeleton_imu_v2" / "checkpoints"))
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    p.add_argument("--sample-csv", type=str, default=r"D:\CUHK-X\Small-Model-Track\Testing\test_file\sample_submission.csv")
    p.add_argument("--out", type=str, default=str(ROOT / "submission_v3_ensemble.csv"))
    p.add_argument("--metrics-out", type=str, default=str(ROOT / "metrics_oof_ensemble.json"))
    p.add_argument("--pattern", type=str, default="best_fold*.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}", flush=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob(args.pattern))
    if len(ckpts) != args.cv_splits:
        print(f"WARN: expected {args.cv_splits} ckpts, found {len(ckpts)}: {[c.name for c in ckpts]}", flush=True)
    if not ckpts:
        raise SystemExit(f"No ckpts in {ckpt_dir}")

    cache = Path(args.cache_dir)
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)

    # --- OOF ---
    gkf = GroupKFold(n_splits=args.cv_splits)
    oof_probs = np.zeros((len(y), 40), dtype=np.float64)
    oof_filled = np.zeros(len(y), dtype=bool)
    fold_rows = []

    all_idx = np.arange(len(y))
    for fi, (tr, va) in enumerate(gkf.split(all_idx, y, users)):
        ck = ckpt_dir / f"best_fold{fi}.pt"
        if not ck.exists():
            # fallback to sorted list
            ck = ckpts[fi] if fi < len(ckpts) else None
        if ck is None or not Path(ck).exists():
            print(f"missing fold{fi} ckpt", flush=True)
            continue
        blob = torch.load(ck, map_location=device, weights_only=False)
        model = build_model(blob.get("model_name", "midfuse"), num_classes=blob.get("num_classes", 40))
        model.load_state_dict(blob["model_state"])
        model.to(device)
        ds = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        P, Y, acc = eval_loader(model, loader, device, dual=True)
        oof_probs[va] = P
        oof_filled[va] = True
        ckpt_val = blob.get("val_acc")
        fold_rows.append({
            "fold": fi,
            "ckpt": str(ck),
            "oof_acc": acc,
            "ckpt_val_acc": ckpt_val,
            "n_val": int(len(va)),
            "val_users": sorted(set(int(users[i]) for i in va)),
        })
        print(f"fold{fi} oof_acc={acc:.4f} ckpt_val={ckpt_val} n={len(va)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert oof_filled.all(), f"OOF incomplete: {oof_filled.sum()}/{len(y)}"
    oof_acc = float((oof_probs.argmax(1) == y).mean())
    fold_accs = [r["oof_acc"] for r in fold_rows]
    print(f"OOF accuracy={oof_acc:.4f}  mean_fold={np.mean(fold_accs):.4f}±{np.std(fold_accs):.4f}", flush=True)

    # --- Test ensemble ---
    Xs, paths = load_skel_test_cache(cache)
    imu_te = np.load(cache / "imu_test.npz", allow_pickle=True)
    Xi, has_te = imu_te["X"], imu_te["has_imu"].astype(np.float32)
    ds_te = DualTestDS(Xs, Xi, has_te, paths)
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    probs = None
    for ck in ckpts:
        blob = torch.load(ck, map_location=device, weights_only=False)
        model = build_model(blob.get("model_name", "midfuse"), num_classes=blob.get("num_classes", 40))
        model.load_state_dict(blob["model_state"])
        model.to(device).eval()
        preds = []
        with torch.no_grad():
            for xs, xi, flag, _bp in tqdm(loader_te, desc=ck.stem):
                logits = model(xs.to(device), xi.to(device), flag.to(device))
                preds.append(torch.softmax(logits, dim=1).cpu())
        P = torch.cat(preds, dim=0).numpy()
        probs = P if probs is None else probs + P
        print(ck.name, "val_acc", blob.get("val_acc"), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probs /= len(ckpts)
    pred_ids = probs.argmax(1)
    pred_map = {paths[i]: int(pred_ids[i]) for i in range(len(paths))}

    sample = pd.read_csv(args.sample_csv)
    rows = []
    for path in sample["path"]:
        rows.append({"path": path, "prediction": pred_map.get(path, pred_map.get(path + "/", 0))})
    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out, flush=True)

    # compare to single-model submission if present
    single_path = ROOT / ".." / "skeleton_imu_v2" / "submission_skeleton_imu_v2.csv"
    agree = None
    if single_path.exists():
        s = pd.read_csv(single_path)
        e = pd.DataFrame(rows)
        merged = s.merge(e, on="path", suffixes=("_single", "_ens"))
        agree = float((merged["prediction_single"] == merged["prediction_ens"]).mean())
        print(f"agreement with single MidFuse submission: {agree:.4f}", flush=True)

    metrics = {
        "oof_accuracy": oof_acc,
        "mean_fold_oof": float(np.mean(fold_accs)),
        "std_fold_oof": float(np.std(fold_accs)),
        "folds": fold_rows,
        "n_ensemble_models": len(ckpts),
        "ckpt_dir": str(ckpt_dir),
        "submission": str(out),
        "agreement_vs_single_v2": agree,
        "v2_cv_mean": 0.5050853925069481,
        "v2_cv_std": 0.05826910368539711,
        "v2_holdout": 0.5366336633663367,
        "note": "OOF = softmax preds from each fold model on its GroupKFold val users; test = mean of 5 fold softmaxes",
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", args.metrics_out, flush=True)


if __name__ == "__main__":
    main()
