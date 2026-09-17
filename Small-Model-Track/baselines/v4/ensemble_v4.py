"""Inference + ensemble for v4 (bone / seeds / v2 folds)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Prefer local dataset/bones; v2 model separately
sys.path.insert(0, str(ROOT))
from dataset import load_skel_test_cache, load_skel_train_cache, DEFAULT_HOLD_OUT_USERS  # noqa
from bones import precompute_bone_cache  # noqa

v4model = _load_module("v4_model_mod", ROOT / "model.py")
v2model = _load_module("v2_model_mod", ROOT.parent / "skeleton_imu_v2" / "model.py")


class BoneTestDS(Dataset):
    def __init__(self, Xb, Xi, has, paths, Xs=None):
        self.Xb, self.Xi, self.has, self.paths, self.Xs = Xb, Xi, has, paths, Xs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.Xs is not None:
            return (
                torch.from_numpy(np.asarray(self.Xs[i], np.float32)),
                torch.from_numpy(np.asarray(self.Xb[i], np.float32)),
                torch.from_numpy(np.asarray(self.Xi[i], np.float32)),
                float(self.has[i]),
                self.paths[i],
            )
        return (
            torch.from_numpy(np.asarray(self.Xb[i], np.float32)),
            torch.from_numpy(np.asarray(self.Xi[i], np.float32)),
            float(self.has[i]),
            self.paths[i],
        )


def load_v2_midfuse(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = v2model.build_model(ckpt.get("model_name", "midfuse"), num_classes=ckpt.get("num_classes", 40))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def load_v4_bone(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = v4model.build_model(ckpt.get("model_name", "bonemidfuse"), num_classes=ckpt.get("num_classes", 40))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_midfuse(model, Xs, Xi, has, device, bs=64):
    n = len(Xs)
    logits = np.zeros((n, 40), dtype=np.float32)
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        xs = torch.from_numpy(Xs[i0:i1]).to(device)
        xi = torch.from_numpy(Xi[i0:i1]).to(device)
        fl = torch.from_numpy(has[i0:i1].astype(np.float32)).to(device)
        logits[i0:i1] = model(xs, xi, fl).cpu().numpy()
    return logits


@torch.no_grad()
def predict_bone(model, Xb, Xi, has, device, Xs=None, bs=64):
    n = len(Xb)
    logits = np.zeros((n, 40), dtype=np.float32)
    triple = Xs is not None
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        xb = torch.from_numpy(Xb[i0:i1]).to(device)
        xi = torch.from_numpy(Xi[i0:i1]).to(device)
        fl = torch.from_numpy(has[i0:i1].astype(np.float32)).to(device)
        if triple:
            xs = torch.from_numpy(Xs[i0:i1]).to(device)
            logits[i0:i1] = model(xs, xb, xi, fl).cpu().numpy()
        else:
            logits[i0:i1] = model(xb, xi, fl).cpu().numpy()
    return logits


def write_submission(paths, preds, sample_csv, out_csv):
    sample = pd.read_csv(sample_csv)
    pred_map = {p: int(pr) for p, pr in zip(paths, preds)}
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
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} missing={missing}", flush=True)


def oof_bone_and_holdout(args, device):
    from sklearn.model_selection import GroupKFold
    cache = Path(args.cache_dir)
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz")
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    bone_path = cache / "bone_train.npz"
    X_bone = np.load(bone_path)["X"] if bone_path.exists() else precompute_bone_cache(X_skel)

    ckpt_dir = Path(args.bone_ckpt_dir)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    gkf = GroupKFold(n_splits=5)
    fold_accs = []
    for fi, (tr, va) in enumerate(gkf.split(np.arange(len(y)), y, users)):
        ck = ckpt_dir / f"best_fold{fi}.pt"
        if not ck.exists():
            print(f"missing {ck}", flush=True)
            continue
        model, _ = load_v4_bone(ck, device)
        logits = predict_bone(model, X_bone[va], X_imu[va], has_imu[va], device)
        preds = logits.argmax(1)
        oof_pred[va] = preds
        acc = float((preds == y[va]).mean())
        fold_accs.append(acc)
        print(f"bone fold{fi} oof_acc={acc:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    valid = oof_pred >= 0
    oof_acc = float((oof_pred[valid] == y[valid]).mean()) if valid.any() else None

    hold_acc = None
    hk = ckpt_dir / "best_holdout.pt"
    if hk.exists():
        hold = set(DEFAULT_HOLD_OUT_USERS)
        va = np.where(np.isin(users, list(hold)))[0]
        model, ck = load_v4_bone(hk, device)
        logits = predict_bone(model, X_bone[va], X_imu[va], has_imu[va], device)
        hold_acc = float((logits.argmax(1) == y[va]).mean())
        print(f"bone holdout_acc={hold_acc:.4f} (ckpt val={ck.get('val_acc')})", flush=True)
        del model
        torch.cuda.empty_cache()

    return {
        "oof_accuracy": oof_acc,
        "fold_accs": fold_accs,
        "mean_fold": float(np.mean(fold_accs)) if fold_accs else None,
        "holdout_acc": hold_acc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--sample-csv", default=r"D:\CUHK-X\Small-Model-Track\Testing\test_file\sample_submission.csv")
    p.add_argument("--bone-ckpt-dir", default=str(ROOT / "checkpoints_bone"))
    p.add_argument("--seed-ckpt-dir", default=str(ROOT / "checkpoints_seeds"))
    p.add_argument("--v2-ckpt-dir", default=str(ROOT.parent / "skeleton_imu_v2" / "checkpoints"))
    p.add_argument("--out-dir", default=str(ROOT))
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")

    cache = Path(args.cache_dir)
    Xs, paths = load_skel_test_cache(cache)
    imu = np.load(cache / "imu_test.npz", allow_pickle=True)
    Xi, has = imu["X"], imu["has_imu"].astype(bool)

    bone_test = cache / "bone_test.npz"
    Xb = np.load(bone_test)["X"] if bone_test.exists() else precompute_bone_cache(Xs)

    bone_metrics = oof_bone_and_holdout(args, device)
    bags = {}

    v2_folds = sorted(Path(args.v2_ckpt_dir).glob("best_fold*.pt"))
    if v2_folds:
        logits_sum = None
        for ck in v2_folds:
            model, _ = load_v2_midfuse(ck, device)
            lg = predict_midfuse(model, Xs, Xi, has, device)
            logits_sum = lg if logits_sum is None else logits_sum + lg
            del model
            torch.cuda.empty_cache()
        bags["v2_folds"] = logits_sum / len(v2_folds)
        print(f"v2 folds n={len(v2_folds)}", flush=True)

    bone_folds = sorted(Path(args.bone_ckpt_dir).glob("best_fold*.pt"))
    if bone_folds:
        logits_sum = None
        for ck in bone_folds:
            model, ckpt = load_v4_bone(ck, device)
            triple = bool(ckpt.get("triple")) or ckpt.get("model_name") == "triple"
            lg = predict_bone(model, Xb, Xi, has, device, Xs=Xs if triple else None)
            logits_sum = lg if logits_sum is None else logits_sum + lg
            del model
            torch.cuda.empty_cache()
        bags["bone_folds"] = logits_sum / len(bone_folds)
        print(f"bone folds n={len(bone_folds)}", flush=True)

    bone_all = Path(args.bone_ckpt_dir) / "best_all_train.pt"
    if not bone_all.exists():
        bone_all = Path(args.bone_ckpt_dir) / "best.pt"
    if bone_all.exists():
        model, ckpt = load_v4_bone(bone_all, device)
        triple = bool(ckpt.get("triple")) or ckpt.get("model_name") == "triple"
        bags["bone_all"] = predict_bone(model, Xb, Xi, has, device, Xs=Xs if triple else None)
        del model
        torch.cuda.empty_cache()

    seed_cks = sorted(Path(args.seed_ckpt_dir).glob("best_*_seed*.pt"))
    if seed_cks:
        logits_sum = None
        for ck in seed_cks:
            model, _ = load_v2_midfuse(ck, device)
            lg = predict_midfuse(model, Xs, Xi, has, device)
            logits_sum = lg if logits_sum is None else logits_sum + lg
            del model
            torch.cuda.empty_cache()
        bags["seed_bag"] = logits_sum / len(seed_cks)
        print(f"seed bag n={len(seed_cks)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, lg in bags.items():
        write_submission(paths, lg.argmax(1), args.sample_csv, out_dir / f"submission_{name}.csv")

    combos = {}
    if "v2_folds" in bags and "seed_bag" in bags:
        combos["v2folds_seeds"] = 0.5 * bags["v2_folds"] + 0.5 * bags["seed_bag"]
    if "bone_folds" in bags and "v2_folds" in bags:
        combos["v2folds_bonefolds"] = 0.5 * bags["v2_folds"] + 0.5 * bags["bone_folds"]
    if "bone_folds" in bags and "seed_bag" in bags and "v2_folds" in bags:
        combos["v4_triple_bag"] = (bags["v2_folds"] + bags["bone_folds"] + bags["seed_bag"]) / 3.0
    if "bone_all" in bags and "v2_folds" in bags:
        combos["v2folds_boneall"] = 0.55 * bags["v2_folds"] + 0.45 * bags["bone_all"]
    if "bone_all" in bags and "seed_bag" in bags and "v2_folds" in bags:
        combos["v4_full_mix"] = 0.4 * bags["v2_folds"] + 0.3 * bags["seed_bag"] + 0.3 * bags["bone_all"]

    for name, lg in combos.items():
        write_submission(paths, lg.argmax(1), args.sample_csv, out_dir / f"submission_{name}.csv")

    primary = None
    for cand in ("v4_full_mix", "v4_triple_bag", "v2folds_seeds", "v2folds_bonefolds", "bone_folds", "v2_folds"):
        if cand in combos or cand in bags:
            primary = cand
            src = combos.get(cand, bags.get(cand))
            write_submission(paths, src.argmax(1), args.sample_csv, out_dir / "submission_v4_best.csv")
            break

    v3 = ROOT.parent / "v3_ensemble" / "submission_v3_ensemble.csv"
    agree = None
    if primary and v3.exists():
        a = pd.read_csv(out_dir / "submission_v4_best.csv")
        b = pd.read_csv(v3)
        m = a.merge(b, on="path", suffixes=("_v4", "_v3"))
        agree = float((m["prediction_v4"] == m["prediction_v3"]).mean())

    # also agreement vs single v2
    v2sub = ROOT.parent / "skeleton_imu_v2" / "submission_skeleton_imu_v2.csv"
    agree_v2 = None
    if primary and v2sub.exists():
        a = pd.read_csv(out_dir / "submission_v4_best.csv")
        b = pd.read_csv(v2sub)
        m = a.merge(b, on="path", suffixes=("_v4", "_v2"))
        agree_v2 = float((m["prediction_v4"] == m["prediction_v2"]).mean())

    report = {
        "bone_metrics": bone_metrics,
        "bags": list(bags.keys()),
        "combos": list(combos.keys()),
        "primary": primary,
        "agreement_vs_v3": agree,
        "agreement_vs_v2_single": agree_v2,
        "baseline_holdout_v2": 0.5366336633663367,
        "baseline_oof_v3": 0.5015,
    }
    with open(out_dir / "metrics_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
