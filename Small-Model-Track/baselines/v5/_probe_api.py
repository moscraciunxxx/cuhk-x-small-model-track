"""Class-wise / confusion analysis for MidFuse holdout + OOF folds."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_V2 = Path(r"D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2")
OUT = Path(r"D:\CUHK-X\Small-Model-Track\baselines\v5")
sys.path.insert(0, str(ROOT_V2))

from dataset import (
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_skel_train_cache,
)
from model import build_model

CLASS_MAP = Path(r"D:\CUHK-X\Small-Model-Track\class_mapping.csv")


def load_class_names():
    names = {}
    for line in CLASS_MAP.read_text(encoding="utf-8").strip().splitlines()[1:]:
        aid, aname = line.split(",", 1)
        names[int(aid)] = aname
    return names


def predict_loader(model, loader, device):
    model.eval()
    ys, preds, users, confs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            xs, xi, y, u, flag = batch
            xs = xs.to(device)
            xi = xi.to(device)
            flag = flag.to(device)
            logits = model(xs, xi, flag)
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            ys.append(y.numpy())
            preds.append(pred.cpu().numpy())
            users.append(u.numpy() if hasattr(u, "numpy") else np.asarray(u))
            confs.append(conf.cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(users),
        np.concatenate(confs),
    )


def confusion(y_true, y_pred, n_classes=40):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def analyze(y_true, y_pred, names, tag, confs=None):
    n_classes = 40
    cm = confusion(y_true, y_pred, n_classes)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    recall = np.divide(correct, np.maximum(support, 1))
    pred_count = cm.sum(axis=0)
    precision = np.divide(correct, np.maximum(pred_count, 1))

    # top confused pairs (off-diagonal)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append({
                    "true": int(i),
                    "pred": int(j),
                    "true_name": names.get(i, str(i)),
                    "pred_name": names.get(j, str(j)),
                    "count": int(cm[i, j]),
                    "true_support": int(support[i]),
                    "frac_of_true": float(cm[i, j] / max(support[i], 1)),
                })
    pairs.sort(key=lambda x: (-x["count"], -x["frac_of_true"]))

    # worst classes by recall
    class_rows = []
    for i in range(n_classes):
        class_rows.append({
            "class": int(i),
            "name": names.get(i, str(i)),
            "support": int(support[i]),
            "correct": int(correct[i]),
            "recall": float(recall[i]),
            "precision": float(precision[i]),
            "pred_count": int(pred_count[i]),
        })
    worst = sorted(class_rows, key=lambda r: (r["recall"], -r["support"]))
    best = sorted(class_rows, key=lambda r: (-r["recall"], -r["support"]))

    # imbalance vs error: correlation support vs recall
    mask = support > 0
    if mask.sum() > 2:
        corr = float(np.corrcoef(support[mask].astype(float), recall[mask])[0, 1])
    else:
        corr = None

    # majority baseline
    maj = int(np.bincount(y_true, minlength=n_classes).argmax()) if len(y_true) else 0
    maj_acc = float((y_true == maj).mean()) if len(y_true) else 0.0

    # errors concentrated?
    n_err = int((y_true != y_pred).sum())
    top10_pair_share = float(sum(p["count"] for p in pairs[:10]) / max(n_err, 1))

    out = {
        "tag": tag,
        "n": int(len(y_true)),
        "accuracy": acc,
        "n_errors": n_err,
        "majority_class": maj,
        "majority_acc": maj_acc,
        "support_recall_corr": corr,
        "top10_confused_pair_share_of_errors": top10_pair_share,
        "top_confused_pairs": pairs[:25],
        "worst_classes": worst[:15],
        "best_classes": best[:10],
        "per_class": class_rows,
        "cm": cm.tolist(),
    }
    if confs is not None:
        out["mean_confidence"] = float(confs.mean())
        out["mean_confidence_correct"] = float(confs[y_true == y_pred].mean()) if (y_true == y_pred).any() else None
        out["mean_confidence_wrong"] = float(confs[y_true != y_pred].mean()) if (y_true != y_pred).any() else None
    return out


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # ckpt may be dict with model_state / state_dict / raw
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
        model_name = ckpt.get("model_name") or ckpt.get("args", {}).get("model") if isinstance(ckpt.get("args"), dict) else None
        model_name = model_name or "midfuse"
        num_classes = ckpt.get("num_classes", 40)
    else:
        state, model_name, num_classes = ckpt, "midfuse", 40
    model = build_model(model_name, num_classes=num_classes)
    # strip module. prefix if any
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    # if state still wrapped
    if "skel_branch.0.weight" not in state and not any("skel" in k for k in list(state.keys())[:5]):
        # try nested
        for k in ("model", "net"):
            if k in state:
                state = state[k]
                break
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = load_class_names()
    device = torch.device("cpu")  # avoid fighting DAM4SAM on GPU
    print("device", device)

    skel, imu, flags, labels, users, meta = load_skel_train_cache(ROOT_V2 / "cache")
    # load_skel_train_cache may return differently - inspect
    print("cache types", type(skel), getattr(skel, "shape", None))

if __name__ == "__main__":
    # probe cache loader signature
    import inspect
    print(inspect.signature(load_skel_train_cache))
    print(inspect.getsource(load_skel_train_cache)[:1500])
