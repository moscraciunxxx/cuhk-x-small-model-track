"""Class-wise / confusion analysis for MidFuse holdout + OOF from fold models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

ROOT_V2 = Path(r"D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2")
OUT = Path(r"D:\CUHK-X\Small-Model-Track\baselines\v5")
CLASS_MAP = Path(r"D:\CUHK-X\Small-Model-Track\class_mapping.csv")
sys.path.insert(0, str(ROOT_V2))

from dataset import DEFAULT_HOLD_OUT_USERS, CachedDualDataset, load_skel_train_cache
from model import build_model

N_CLASSES = 40


def load_class_names():
    names = {}
    for line in CLASS_MAP.read_text(encoding="utf-8").strip().splitlines()[1:]:
        aid, aname = line.split(",", 1)
        names[int(aid)] = aname
    return names


def predict(model, loader, device):
    model.eval()
    ys, preds, users, confs = [], [], [], []
    with torch.no_grad():
        for xs, xi, y, u, flag in loader:
            xs = xs.to(device)
            xi = xi.to(device)
            flag = flag.to(device)
            logits = model(xs, xi, flag)
            prob = torch.softmax(logits.float(), dim=1)
            conf, pred = prob.max(dim=1)
            ys.append(y.numpy())
            preds.append(pred.cpu().numpy())
            users.append(np.asarray(u))
            confs.append(conf.cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(users),
        np.concatenate(confs),
    )


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "midfuse")
    num_classes = ckpt.get("num_classes", N_CLASSES)
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def analyze(y_true, y_pred, names, tag, confs=None):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    recall = np.divide(correct, np.maximum(support, 1))
    pred_count = cm.sum(axis=0)
    precision = np.divide(correct, np.maximum(pred_count, 1))

    pairs = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i == j or cm[i, j] == 0:
                continue
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

    class_rows = []
    for i in range(N_CLASSES):
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

    mask = support > 0
    corr = float(np.corrcoef(support[mask].astype(float), recall[mask])[0, 1]) if mask.sum() > 2 else None
    maj = int(np.bincount(y_true, minlength=N_CLASSES).argmax()) if len(y_true) else 0
    maj_acc = float((y_true == maj).mean()) if len(y_true) else 0.0
    n_err = int((y_true != y_pred).sum())
    top10_share = float(sum(p["count"] for p in pairs[:10]) / max(n_err, 1))

    # imbalance explanation: compare recall of bottom-quartile support vs top
    if mask.sum() >= 8:
        q1, q3 = np.percentile(support[mask], [25, 75])
        low = recall[(support > 0) & (support <= q1)]
        high = recall[(support > 0) & (support >= q3)]
        imbalance_gap = float(high.mean() - low.mean()) if len(low) and len(high) else None
    else:
        imbalance_gap = None

    out = {
        "tag": tag,
        "n": int(len(y_true)),
        "accuracy": acc,
        "n_errors": n_err,
        "majority_class": maj,
        "majority_class_name": names.get(maj, str(maj)),
        "majority_acc": maj_acc,
        "support_recall_corr": corr,
        "imbalance_recall_gap_high_minus_low_support": imbalance_gap,
        "top10_confused_pair_share_of_errors": top10_share,
        "top_confused_pairs": pairs[:25],
        "worst_classes": worst[:15],
        "best_classes": best[:10],
        "per_class": class_rows,
        "cm": cm.tolist(),
    }
    if confs is not None and len(confs):
        out["mean_confidence"] = float(confs.mean())
        ok = y_true == y_pred
        out["mean_confidence_correct"] = float(confs[ok].mean()) if ok.any() else None
        out["mean_confidence_wrong"] = float(confs[~ok].mean()) if (~ok).any() else None
    return out


def md_section(a, names):
    lines = []
    lines.append(f"## {a['tag']}")
    lines.append("")
    lines.append(f"- n={a['n']}  accuracy=**{a['accuracy']:.4f}**  errors={a['n_errors']}")
    lines.append(f"- majority baseline: class {a['majority_class']} ({a['majority_class_name']}) acc={a['majority_acc']:.4f}")
    lines.append(f"- corr(support, recall)={a['support_recall_corr']}")
    lines.append(f"- recall gap (high-support quartile − low-support quartile)={a['imbalance_recall_gap_high_minus_low_support']}")
    lines.append(f"- top-10 confused pairs cover {a['top10_confused_pair_share_of_errors']:.1%} of errors")
    if "mean_confidence" in a:
        lines.append(
            f"- mean conf: all={a['mean_confidence']:.3f} correct={a['mean_confidence_correct']} wrong={a['mean_confidence_wrong']}"
        )
    lines.append("")
    lines.append("### Top confused pairs")
    lines.append("")
    lines.append("| true | pred | count | frac_of_true |")
    lines.append("|------|------|------:|-------------:|")
    for p in a["top_confused_pairs"][:15]:
        lines.append(
            f"| {p['true_name']} → | {p['pred_name']} | {p['count']} | {p['frac_of_true']:.2f} |"
        )
    lines.append("")
    lines.append("### Worst classes (by recall)")
    lines.append("")
    lines.append("| class | support | recall | precision |")
    lines.append("|-------|--------:|-------:|----------:|")
    for r in a["worst_classes"][:12]:
        lines.append(f"| {r['name']} | {r['support']} | {r['recall']:.3f} | {r['precision']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = load_class_names()
    device = torch.device("cpu")
    print("device", device, flush=True)

    cache = ROOT_V2 / "cache"
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y, dtype=np.int64)
    users = np.asarray(users, dtype=np.int64)
    print(f"n={len(y)} skel={X_skel.shape} imu={X_imu.shape} has_imu={has_imu.sum()}", flush=True)

    hold_users = set(DEFAULT_HOLD_OUT_USERS)
    hold_idx = np.where(np.isin(users, list(hold_users)))[0]
    print(f"holdout users={sorted(hold_users)} n={len(hold_idx)}", flush=True)

    # Prefer v2b holdout ckpt (0.537), fall back to checkpoints/
    hold_ckpt = ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"
    if not hold_ckpt.exists():
        hold_ckpt = ROOT_V2 / "checkpoints" / "best_holdout.pt"
    print("holdout ckpt", hold_ckpt, flush=True)
    model, ckpt_meta = load_model(hold_ckpt, device)
    print("ckpt val_acc", ckpt_meta.get("val_acc"), "model", ckpt_meta.get("model_name"), flush=True)

    ds = CachedDualDataset(X_skel, X_imu, y, users, hold_idx, has_imu, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    yt, yp, yu, conf = predict(model, loader, device)
    hold_an = analyze(yt, yp, names, "holdout_midfuse_v2b_users_8_9_24", conf)

    # OOF from 5 fold ckpts (GroupKFold same as train)
    fold_dir = ROOT_V2 / "checkpoints_midfuse_v2b"
    if not (fold_dir / "best_fold0.pt").exists():
        fold_dir = ROOT_V2 / "checkpoints"
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    oof_conf = np.zeros(len(y), dtype=np.float32)
    fold_accs = []
    for fold_i, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, groups=users)):
        ck = fold_dir / f"best_fold{fold_i}.pt"
        print(f"OOF fold{fold_i} ckpt={ck.exists()} n_val={len(va)}", flush=True)
        if not ck.exists():
            continue
        m, _ = load_model(ck, device)
        ds_f = CachedDualDataset(X_skel, X_imu, y, users, va, has_imu, augment=False)
        ld = DataLoader(ds_f, batch_size=64, shuffle=False, num_workers=0)
        yt_f, yp_f, _, cf = predict(m, ld, device)
        # map back — Dataset returns in va order
        oof_pred[va] = yp_f
        oof_conf[va] = cf
        fold_accs.append(float((yt_f == yp_f).mean()))
        print(f"  fold{fold_i} acc={fold_accs[-1]:.4f}", flush=True)
        del m

    valid = oof_pred >= 0
    oof_an = analyze(y[valid], oof_pred[valid], names, "oof_5fold_midfuse_v2b", oof_conf[valid])
    oof_an["fold_accs"] = fold_accs
    oof_an["mean_fold_acc"] = float(np.mean(fold_accs)) if fold_accs else None

    # Class support overall (imbalance reference)
    overall_support = np.bincount(y, minlength=N_CLASSES).tolist()
    hold_support = np.bincount(y[hold_idx], minlength=N_CLASSES).tolist()

    # Imbalance verdict
    def verdict(a):
        corr = a.get("support_recall_corr")
        gap = a.get("imbalance_recall_gap_high_minus_low_support")
        notes = []
        if corr is not None and abs(corr) < 0.25:
            notes.append(
                f"Support–recall correlation is weak ({corr:.3f}); class imbalance alone does NOT strongly explain errors."
            )
        elif corr is not None and corr > 0.25:
            notes.append(
                f"Positive support–recall corr ({corr:.3f}); rarer classes tend to have lower recall — imbalance contributes."
            )
        elif corr is not None and corr < -0.25:
            notes.append(
                f"Negative support–recall corr ({corr:.3f}); rare classes are not systematically worse — imbalance not the main driver."
            )
        if gap is not None:
            notes.append(f"High-vs-low support recall gap={gap:.3f}.")
        # semantic confusions
        top = a["top_confused_pairs"][:5]
        if top:
            notes.append(
                "Top confusions look largely semantic/similar-motion (not pure frequency): "
                + "; ".join(f"{p['true_name']}→{p['pred_name']} ({p['count']})" for p in top)
            )
        return notes

    summary = {
        "holdout": {k: v for k, v in hold_an.items() if k != "cm"},
        "oof": {k: v for k, v in oof_an.items() if k != "cm"},
        "holdout_cm": hold_an["cm"],
        "oof_cm": oof_an["cm"],
        "overall_class_support": overall_support,
        "holdout_class_support": hold_support,
        "holdout_verdict": verdict(hold_an),
        "oof_verdict": verdict(oof_an),
        "holdout_ckpt": str(hold_ckpt),
        "fold_ckpt_dir": str(fold_dir),
        "baseline_holdout_reported": 0.5366336633663367,
    }

    # keep cm in separate lighter analysis json without huge duplication in md
    json_path = OUT / "error_analysis.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# MidFuse v2 error analysis (v5)")
    md.append("")
    md.append("Quiet MSI run. Holdout users **{8,9,24}**; OOF from 5 GroupKFold MidFuse v2b fold ckpts.")
    md.append("")
    md.append(f"- Holdout ckpt: `{hold_ckpt}`")
    md.append(f"- Fold ckpt dir: `{fold_dir}`")
    md.append(f"- Reported holdout baseline: **0.537** (measured here {hold_an['accuracy']:.4f})")
    md.append(f"- OOF accuracy: **{oof_an['accuracy']:.4f}** (fold mean {oof_an.get('mean_fold_acc')})")
    md.append("")
    md.append("## Does imbalance explain errors?")
    md.append("")
    for v in summary["holdout_verdict"]:
        md.append(f"- (holdout) {v}")
    for v in summary["oof_verdict"]:
        md.append(f"- (oof) {v}")
    md.append("")
    md.append(md_section(hold_an, names))
    md.append(md_section(oof_an, names))
    md.append("## Notes for v5 experiments")
    md.append("")
    md.append("- Skip Radar/Thermal (known bad).")
    md.append("- Depth_Color + IR have full train/test coverage (see modality_coverage.json).")
    md.append("- High-ROI MidFuse tweak candidate: focal / class-focused loss on worst-recall classes from above.")
    md.append("")
    (OUT / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote", json_path, flush=True)
    print("Wrote", OUT / "error_analysis.md", flush=True)
    print("HOLD", hold_an["accuracy"], "OOF", oof_an["accuracy"], flush=True)


if __name__ == "__main__":
    main()
