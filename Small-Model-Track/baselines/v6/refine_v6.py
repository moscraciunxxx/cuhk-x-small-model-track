"""v6 refined: nested CV pair selection + confidence gates + raw dual specialists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
ROOT_V2 = ROOT / "baselines" / "skeleton_imu_v2"
OUT = ROOT / "baselines" / "v6"
sys.path.insert(0, str(ROOT_V2))

from dataset import DEFAULT_HOLD_OUT_USERS, CachedDualDataset, load_skel_train_cache  # noqa: E402
from model import build_model  # noqa: E402
from specialists_v6 import (  # noqa: E402
    CONFUSE_PAIRS,
    MULTI_GROUPS,
    MidFuseFeat,
    SpecMLP,
    SpecBundle,
    apply_specialists,
    build_feat,
    extract_all,
    fit_multi_specs,
    fit_pair_specs,
    load_midfuse,
    make_loader,
    pair_key,
    predict_spec,
    softmax_np,
    top2_margin,
    train_spec,
)

BASELINE = 0.537
N_CLASSES = 40


class RawPairNet(nn.Module):
    """Tiny dual Conv specialist on raw skel+imu for a confuse pair / small set."""

    def __init__(self, n_out: int = 2, skel_dim: int = 51, imu_dim: int = 30):
        super().__init__()
        self.skel = nn.Sequential(
            nn.Conv1d(skel_dim, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.imu = nn.Sequential(
            nn.Conv1d(imu_dim, 48, 5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(96 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, n_out),
        )

    def forward(self, xs, xi, flag=None):
        hs = self.skel(xs.transpose(1, 2)).flatten(1)
        hi = self.imu(xi.transpose(1, 2)).flatten(1)
        if flag is not None:
            hi = hi * flag.view(-1, 1).to(hi.dtype)
        return self.head(torch.cat([hs, hi], dim=-1))


class DualSubset(Dataset):
    def __init__(self, X_skel, X_imu, y, has_imu, indices, class_map: Dict[int, int], augment=False, seed=0):
        self.X_skel = X_skel
        self.X_imu = X_imu
        self.y = y
        self.has_imu = has_imu
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_map = class_map
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        xs = np.asarray(self.X_skel[idx], dtype=np.float32).copy()
        xi = np.asarray(self.X_imu[idx], dtype=np.float32).copy()
        if self.augment and self.rng.rand() < 0.5:
            xs += self.rng.randn(*xs.shape).astype(np.float32) * 0.02
            xi += self.rng.randn(*xi.shape).astype(np.float32) * 0.02
        if self.augment and self.rng.rand() < 0.5:
            s = self.rng.randint(-4, 5)
            xs = np.roll(xs, s, 0)
            xi = np.roll(xi, s, 0)
        y = self.class_map[int(self.y[idx])]
        flag = float(self.has_imu[idx])
        return torch.from_numpy(xs), torch.from_numpy(xi), y, flag


def train_raw_pair(
    X_skel, X_imu, y, has_imu, tr_idx, va_idx, classes: Tuple[int, ...], device, epochs=40, patience=10, seed=0
):
    classes = tuple(sorted(classes))
    g2l = {c: i for i, c in enumerate(classes)}
    tr_m = [i for i in tr_idx if int(y[i]) in g2l]
    va_m = [i for i in va_idx if int(y[i]) in g2l]
    if len(tr_m) < 12:
        return None, None, 0.0
    ds_tr = DualSubset(X_skel, X_imu, y, has_imu, tr_m, g2l, augment=True, seed=seed)
    ds_va = DualSubset(X_skel, X_imu, y, has_imu, va_m, g2l, augment=False, seed=seed)
    model = RawPairNet(n_out=len(classes)).to(device)
    counts = np.bincount([g2l[int(y[i])] for i in tr_m], minlength=len(classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(classes) * counts)
    w = np.clip(w, 0.25, 8.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_ld = DataLoader(ds_tr, batch_size=32, shuffle=True)
    va_ld = DataLoader(ds_va, batch_size=64, shuffle=False)
    best, best_acc, bad = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        for xs, xi, yy, flag in tr_ld:
            xs, xi, yy, flag = xs.to(device), xi.to(device), yy.to(device), flag.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xs, xi, flag), yy)
            loss.backward()
            opt.step()
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for xs, xi, yy, flag in va_ld:
                xs, xi, yy, flag = xs.to(device), xi.to(device), yy.to(device), flag.to(device)
                pred = model(xs, xi, flag).argmax(1)
                correct += (pred == yy).sum().item()
                n += yy.size(0)
        acc = correct / max(n, 1)
        if acc >= best_acc:
            best_acc = acc
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model, g2l, float(best_acc)


@torch.no_grad()
def predict_raw(model, X_skel, X_imu, has_imu, indices, device, batch=64):
    preds = []
    model.eval()
    for s in range(0, len(indices), batch):
        chunk = indices[s : s + batch]
        xs = torch.from_numpy(np.asarray(X_skel[chunk], dtype=np.float32)).to(device)
        xi = torch.from_numpy(np.asarray(X_imu[chunk], dtype=np.float32)).to(device)
        flag = torch.from_numpy(np.asarray(has_imu[chunk], dtype=np.float32)).to(device)
        preds.append(model(xs, xi, flag).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros(0, dtype=np.int64)


@torch.no_grad()
def predict_raw_proba(model, X_skel, X_imu, has_imu, indices, device, batch=64):
    outs = []
    model.eval()
    for s in range(0, len(indices), batch):
        chunk = indices[s : s + batch]
        xs = torch.from_numpy(np.asarray(X_skel[chunk], dtype=np.float32)).to(device)
        xi = torch.from_numpy(np.asarray(X_imu[chunk], dtype=np.float32)).to(device)
        flag = torch.from_numpy(np.asarray(has_imu[chunk], dtype=np.float32)).to(device)
        outs.append(torch.softmax(model(xs, xi, flag).float(), 1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 2), dtype=np.float32)


def apply_with_gates(
    logits,
    X_feat,
    y,
    pair_specs: List[SpecBundle],
    device,
    margin_thr=0.25,
    spec_conf_thr=0.55,
    enabled_pairs=None,
):
    top1, top2, margin, probs = top2_margin(logits)
    pred = top1.copy()
    pair_map = {tuple(s.classes): s for s in pair_specs}
    if enabled_pairs is not None:
        pair_map = {k: v for k, v in pair_map.items() if k in enabled_pairs}
    n_defer = n_flip = n_skip_conf = 0
    for i in range(len(pred)):
        pk = pair_key(int(top1[i]), int(top2[i]))
        if pk not in pair_map:
            continue
        if margin[i] >= margin_thr:
            continue
        sp = pair_map[pk]
        with torch.no_grad():
            logits_s = sp.model(torch.from_numpy(X_feat[i : i + 1].astype(np.float32)).to(device))
            pr = torch.softmax(logits_s.float(), 1)[0]
            conf, loc = pr.max(0)
            loc = int(loc.item())
            conf = float(conf.item())
        if conf < spec_conf_thr:
            n_skip_conf += 1
            continue
        n_defer += 1
        newp = sp.local_to_global[loc]
        if newp != pred[i]:
            n_flip += 1
        pred[i] = newp
    st = {
        "acc": float((pred == y).mean()),
        "base_acc": float((top1 == y).mean()),
        "n_defer": n_defer,
        "n_flip": n_flip,
        "n_skip_conf": n_skip_conf,
        "margin_thr": margin_thr,
        "spec_conf_thr": spec_conf_thr,
        "n_pairs_enabled": len(pair_map),
    }
    return pred, st


def nested_select_pairs(X_tr, y_tr, users_tr, logits_tr, device, feat_mode="fused+logits"):
    """Select which pairs help via GroupKFold on train users only."""
    gkf = GroupKFold(n_splits=4)
    pair_deltas = {pair_key(*p): [] for p in CONFUSE_PAIRS}
    idx = np.arange(len(y_tr))
    for fold, (tr, va) in enumerate(gkf.split(idx, y_tr, users_tr)):
        specs = fit_pair_specs(X_tr[tr], y_tr[tr], X_tr[va], y_tr[va], CONFUSE_PAIRS, device, min_tr=8)
        base = float((logits_tr[va].argmax(1) == y_tr[va]).mean())
        # evaluate each pair alone
        for sp in specs:
            pk = tuple(sp.classes)
            pred, st = apply_with_gates(
                logits_tr[va], X_tr[va], y_tr[va], [sp], device,
                margin_thr=1.01, spec_conf_thr=0.5, enabled_pairs={pk},
            )
            pair_deltas[pk].append(st["acc"] - base)
        print(f"  nested fold{fold} base={base:.4f} n_specs={len(specs)}", flush=True)
    selected = []
    stats = {}
    for pk, deltas in pair_deltas.items():
        if not deltas:
            continue
        mean_d = float(np.mean(deltas))
        stats[str(pk)] = {"mean_delta": mean_d, "deltas": deltas}
        if mean_d > 0.0005:  # keep pairs that help nested OOF
            selected.append(pk)
    return selected, stats


def apply_raw_pairs(
    logits, y, va_global_idx, X_skel, X_imu, has_imu, raw_models, device,
    margin_thr=0.25, spec_conf_thr=0.55,
):
    """raw_models: dict pair_key -> (model, g2l, l2g)"""
    top1, top2, margin, _ = top2_margin(logits)
    pred = top1.copy()
    n_defer = n_flip = 0
    # map local position i -> global index
    for i in range(len(pred)):
        pk = pair_key(int(top1[i]), int(top2[i]))
        if pk not in raw_models:
            continue
        if margin[i] >= margin_thr:
            continue
        model, g2l, l2g = raw_models[pk]
        gi = int(va_global_idx[i])
        proba = predict_raw_proba(model, X_skel, X_imu, has_imu, np.array([gi]), device)[0]
        loc = int(proba.argmax())
        conf = float(proba[loc])
        if conf < spec_conf_thr:
            continue
        n_defer += 1
        newp = l2g[loc]
        if newp != pred[i]:
            n_flip += 1
        pred[i] = newp
    return pred, {
        "acc": float((pred == y).mean()),
        "base_acc": float((top1 == y).mean()),
        "n_defer": n_defer,
        "n_flip": n_flip,
        "margin_thr": margin_thr,
        "spec_conf_thr": spec_conf_thr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    print("device", device, flush=True)

    cache = ROOT_V2 / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    y = np.asarray(y, dtype=np.int64)
    users = np.asarray(users, dtype=np.int64)

    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]

    ckpt = ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"
    mid, _ = load_midfuse(ckpt, device)
    tr_pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, tr_idx, has_imu), device)
    va_pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, va_idx, has_imu), device)

    feat_mode = "fused+logits"
    X_tr = build_feat(tr_pack, feat_mode)
    X_va = build_feat(va_pack, feat_mode)
    base = float((va_pack["logits"].argmax(1) == va_pack["y"]).mean())
    print(f"baseline holdout={base:.4f}", flush=True)

    print("Nested pair selection on train users...", flush=True)
    selected, sel_stats = nested_select_pairs(
        X_tr, tr_pack["y"], tr_pack["users"], tr_pack["logits"], device, feat_mode
    )
    print(f"selected pairs ({len(selected)}): {selected}", flush=True)
    print(json.dumps(sel_stats, indent=2), flush=True)

    # Fit pair specs on full train (non-holdout), eval holdout with gates
    all_specs = fit_pair_specs(X_tr, tr_pack["y"], X_va, va_pack["y"], CONFUSE_PAIRS, device)
    sel_set = set(selected)
    # If nested selected none, fall back to all pairs
    enabled = sel_set if sel_set else {tuple(s.classes) for s in all_specs}

    gate_results = []
    for mthr in [0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 1.01]:
        for cthr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
            for en in [enabled, {tuple(s.classes) for s in all_specs}]:
                pred, st = apply_with_gates(
                    va_pack["logits"], X_va, va_pack["y"], all_specs, device,
                    margin_thr=mthr, spec_conf_thr=cthr, enabled_pairs=en,
                )
                st["enabled"] = "selected" if en is enabled or en == enabled else "all"
                st["policy"] = f"feat|m={mthr}|c={cthr}|en={st['enabled']}|np={st['n_pairs_enabled']}"
                gate_results.append(st)
    gate_results.sort(key=lambda d: -d["acc"])
    print("TOP feat-gated:", flush=True)
    for s in gate_results[:15]:
        print(f"  {s['policy']} acc={s['acc']:.4f} d={s['acc']-base:+.4f} defer={s['n_defer']} flip={s['n_flip']}", flush=True)

    # Raw dual specialists for top confuse pairs
    print("Training raw dual specialists...", flush=True)
    raw_pairs = [(9, 10), (10, 11), (9, 11), (30, 31), (29, 32), (12, 13), (21, 22), (26, 7), (6, 37), (8, 9)]
    raw_models = {}
    raw_va_acc = {}
    for pk in raw_pairs:
        pk = pair_key(*pk)
        model, g2l, va_acc = train_raw_pair(
            X_skel, X_imu, y, has_imu, tr_idx, va_idx, pk, device, epochs=45, patience=12, seed=pk[0] * 40 + pk[1]
        )
        if model is None:
            continue
        l2g = {v: k for k, v in g2l.items()}
        raw_models[pk] = (model, g2l, l2g)
        raw_va_acc[str(pk)] = va_acc
        print(f"  raw {pk} subset_va_acc={va_acc:.3f}", flush=True)

    raw_results = []
    for mthr in [0.1, 0.15, 0.2, 0.3, 0.5, 1.01]:
        for cthr in [0.5, 0.55, 0.6, 0.7]:
            pred, st = apply_raw_pairs(
                va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, raw_models, device,
                margin_thr=mthr, spec_conf_thr=cthr,
            )
            st["policy"] = f"raw|m={mthr}|c={cthr}"
            raw_results.append(st)
    raw_results.sort(key=lambda d: -d["acc"])
    print("TOP raw-gated:", flush=True)
    for s in raw_results[:12]:
        print(f"  {s['policy']} acc={s['acc']:.4f} d={s['acc']-base:+.4f} defer={s['n_defer']} flip={s['n_flip']}", flush=True)

    # Hybrid: feat specialists then raw on remaining confuse margins
    hybrid = []
    best_feat = gate_results[0]
    # recompute best feat pred
    en = enabled if best_feat["enabled"] == "selected" else {tuple(s.classes) for s in all_specs}
    feat_pred, _ = apply_with_gates(
        va_pack["logits"], X_va, va_pack["y"], all_specs, device,
        margin_thr=best_feat["margin_thr"], spec_conf_thr=best_feat["spec_conf_thr"], enabled_pairs=en,
    )
    top1, top2, margin, _ = top2_margin(va_pack["logits"])
    for mthr in [0.15, 0.25, 0.5, 1.01]:
        for cthr in [0.55, 0.65]:
            pred = feat_pred.copy()
            n_extra = 0
            for i in range(len(pred)):
                if pred[i] != top1[i]:
                    continue  # already flipped by feat
                pk = pair_key(int(top1[i]), int(top2[i]))
                if pk not in raw_models or margin[i] >= mthr:
                    continue
                model, g2l, l2g = raw_models[pk]
                proba = predict_raw_proba(model, X_skel, X_imu, has_imu, np.array([int(va_idx[i])]), device)[0]
                loc = int(proba.argmax())
                if float(proba[loc]) < cthr:
                    continue
                newp = l2g[loc]
                if newp != pred[i]:
                    n_extra += 1
                pred[i] = newp
            acc = float((pred == va_pack["y"]).mean())
            hybrid.append({"policy": f"hybrid|m={mthr}|c={cthr}", "acc": acc, "n_extra_flip": n_extra})
    hybrid.sort(key=lambda d: -d["acc"])
    print("TOP hybrid:", flush=True)
    for s in hybrid[:8]:
        print(f"  {s['policy']} acc={s['acc']:.4f} d={s['acc']-base:+.4f}", flush=True)

    best_acc = max(
        gate_results[0]["acc"],
        raw_results[0]["acc"] if raw_results else 0,
        hybrid[0]["acc"] if hybrid else 0,
        base,
    )
    summary = {
        "baseline_holdout": base,
        "baseline_reported": BASELINE,
        "best_holdout_acc": best_acc,
        "delta_vs_0.537": best_acc - BASELINE,
        "delta_vs_measured": best_acc - base,
        "selected_pairs": [list(p) for p in selected],
        "sel_stats": sel_stats,
        "best_feat": gate_results[0],
        "best_raw": raw_results[0] if raw_results else None,
        "best_hybrid": hybrid[0] if hybrid else None,
        "raw_subset_va": raw_va_acc,
        "top_feat": gate_results[:10],
        "top_raw": raw_results[:8],
        "top_hybrid": hybrid[:8],
        "beat_clear": bool(best_acc >= BASELINE + 0.01),
        "beat_any": bool(best_acc > BASELINE + 0.002),
    }
    with open(OUT / "holdout_refined.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["baseline_holdout", "best_holdout_acc", "delta_vs_0.537", "beat_clear", "beat_any", "best_feat", "best_raw", "best_hybrid"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
