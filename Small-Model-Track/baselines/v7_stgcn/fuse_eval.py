"""Late-fuse / agreement of ST-GCN logits vs MidFuse v2 holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_imu_caches,
    load_skel_train_cache,
)
from model import build_model

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"


def load_v7(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = build_model(ck.get("model_name", "stgcn_fuse"), num_classes=ck.get("num_classes", 40))
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m, ck


def load_midfuse(ckpt, device):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v2_model", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = mod.build_model(ck.get("model_name", "midfuse"), num_classes=ck.get("num_classes", 40))
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m, ck


@torch.no_grad()
def logits_on(model, ds, device, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs, ys = [], []
    for xs, xi, y, _u, flag in loader:
        xs = xs.to(device)
        xi = xi.to(device)
        flag = flag.to(device)
        outs.append(model(xs, xi, flag).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7-ckpt", type=str, default=str(ROOT / "checkpoints" / "best_holdout.pt"))
    p.add_argument(
        "--midfuse-ckpt",
        type=str,
        default=str(V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"),
    )
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    p.add_argument("--out", type=str, default=str(ROOT / "agreement.json"))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_skel, y, users, _ = load_skel_train_cache(Path(args.cache_dir))
    X_imu, has_imu, _, _ = load_imu_caches(Path(args.cache_dir))
    hold = set(DEFAULT_HOLD_OUT_USERS)
    va = np.where(np.isin(users, list(hold)))[0]
    ds = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False)

    m7, _ = load_v7(Path(args.v7_ckpt), device)
    mf, _ = load_midfuse(Path(args.midfuse_ckpt), device)
    L7, yt = logits_on(m7, ds, device)
    Lf, _ = logits_on(mf, ds, device)

    p7 = L7.argmax(1)
    pf = Lf.argmax(1)
    acc7 = float((p7 == yt).mean())
    accf = float((pf == yt).mean())
    agree = float((p7 == pf).mean())

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    S7, Sf = softmax(L7), softmax(Lf)
    fuse_rows = []
    best = {"alpha": None, "acc": -1.0}
    for a in args.alphas:
        mix = a * S7 + (1.0 - a) * Sf
        pred = mix.argmax(1)
        acc = float((pred == yt).mean())
        fuse_rows.append({"alpha_stgcn": a, "acc": acc})
        if acc > best["acc"]:
            best = {"alpha": a, "acc": acc}

    out = {
        "n_holdout": int(len(yt)),
        "stgcn_holdout_acc": acc7,
        "midfuse_holdout_acc": accf,
        "pred_agreement": agree,
        "late_fuse": fuse_rows,
        "best_late_fuse": best,
        "delta_vs_midfuse": acc7 - accf,
        "complementary_hint": agree < 0.85 and best["acc"] > max(acc7, accf) + 0.005,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
