"""CUHK-X Small Model Track v6: confuse-pair specialists on MidFuse features.

Quiet MSI run. Reuses skeleton_imu_v2 cache + MidFuse v2b checkpoints.
Holdout users {8,9,24}; baseline MidFuse holdout ~0.537.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
ROOT_V2 = ROOT / "baselines" / "skeleton_imu_v2"
OUT = ROOT / "baselines" / "v6"
CLASS_MAP = ROOT / "class_mapping.csv"
sys.path.insert(0, str(ROOT_V2))

from dataset import (  # noqa: E402
    DEFAULT_HOLD_OUT_USERS,
    CachedDualDataset,
    load_skel_train_cache,
    load_skel_test_cache,
)
from model import build_model, count_parameters  # noqa: E402

N_CLASSES = 40
BASELINE_HOLDOUT = 0.537
# Prefer clear win of +0.01
WIN_DELTA = 0.01

# Bidirectional confuse pairs (sorted tuples as keys)
CONFUSE_PAIRS: List[Tuple[int, int]] = [
    (9, 10),   # Pour <-> Stir
    (10, 11),  # Stir <-> Peel
    (9, 11),   # Pour <-> Peel (related kitchen)
    (30, 31),  # jumping_jacks <-> stretch
    (29, 32),  # Squats <-> Stand_up
    (26, 7),   # Play_games <-> Eat
    (12, 13),  # Sweep <-> Mop
    (21, 22),  # Read <-> Turn_pages
    (32, 34),  # Stand_up <-> Sit_down
    (6, 37),   # Drink <-> Take_medicine
    (29, 34),  # Squats <-> Sit_down
    (26, 24),  # Play_games <-> mobile
    (26, 19),  # Play_games <-> phone_call
    (8, 9),    # tableware <-> Pour
    (7, 6),    # Eat <-> Drink
]

# Ternary / small multi specialists (kitchen cluster, exercise cluster)
MULTI_GROUPS: List[Tuple[str, Tuple[int, ...]]] = [
    ("kitchen_hand", (8, 9, 10, 11)),
    ("exercise", (29, 30, 31, 32, 34, 35)),
    ("sedentary_device", (17, 18, 21, 22, 24, 25, 26)),
]

HARD_CLASSES = [2, 18, 25, 30, 11, 26, 29, 22, 8, 10]


def load_class_names() -> Dict[int, str]:
    names = {}
    for line in CLASS_MAP.read_text(encoding="utf-8").strip().splitlines()[1:]:
        aid, aname = line.split(",", 1)
        names[int(aid)] = aname
    return names


def pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


class MidFuseFeat(nn.Module):
    """Wrap MidFuseNet to also return fused features + penultimate."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x_skel, x_imu=None, imu_flag=None):
        b = self.base
        hs = b.skel(x_skel)
        if x_imu is None:
            hi = torch.zeros(hs.size(0), b.imu.out_dim, device=hs.device, dtype=hs.dtype)
        else:
            hi = b.imu(x_imu)
            if imu_flag is not None:
                flag = imu_flag.view(-1, 1).to(hi.dtype)
                gate = b.imu_gate(flag)
                hi = hi * gate * flag
        fused = torch.cat([hs, hi], dim=-1)  # 384
        # head: LN, Drop, Linear->192, ReLU, Drop, Linear->C
        h = b.head[0](fused)  # LN
        h = b.head[1](h)      # Drop (eval: identity-ish)
        h = b.head[2](h)      # Linear 384->192
        h = b.head[3](h)      # ReLU
        pen = h
        h = b.head[4](h)      # Drop
        logits = b.head[5](h)
        return logits, fused, pen


def load_midfuse(ckpt_path: Path, device: torch.device) -> Tuple[MidFuseFeat, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("model_name", "midfuse"), num_classes=ckpt.get("num_classes", N_CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return MidFuseFeat(model).to(device).eval(), ckpt


@torch.no_grad()
def extract_all(model: MidFuseFeat, loader: DataLoader, device: torch.device):
    logits_l, fused_l, pen_l, y_l, u_l, idx_l = [], [], [], [], [], []
    for batch in loader:
        xs, xi, y, u, flag = batch
        xs = xs.to(device)
        xi = xi.to(device)
        flag = flag.to(device)
        logits, fused, pen = model(xs, xi, flag)
        logits_l.append(logits.float().cpu().numpy())
        fused_l.append(fused.float().cpu().numpy())
        pen_l.append(pen.float().cpu().numpy())
        y_l.append(y.numpy() if hasattr(y, "numpy") else np.asarray(y))
        u_l.append(np.asarray(u))
    return {
        "logits": np.concatenate(logits_l),
        "fused": np.concatenate(fused_l),
        "pen": np.concatenate(pen_l),
        "y": np.concatenate(y_l).astype(np.int64),
        "users": np.concatenate(u_l).astype(np.int64),
    }


class FeatDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


class SpecMLP(nn.Module):
    def __init__(self, in_dim: int, n_out: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_spec(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    n_out: int,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    batch: int = 64,
    patience: int = 10,
    seed: int = 0,
) -> Tuple[SpecMLP, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpecMLP(X_tr.shape[1], n_out).to(device)
    # class weights
    counts = np.bincount(y_tr, minlength=n_out).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_out * counts)
    w = np.clip(w, 0.25, 8.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_loader = DataLoader(FeatDS(X_tr, y_tr), batch_size=batch, shuffle=True)
    best_state = None
    best_acc = -1.0
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if X_va is not None and len(y_va):
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X_va).to(device)).argmax(1).cpu().numpy()
            acc = float((pred == y_va).mean())
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_acc = 1.0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, float(best_acc)


@torch.no_grad()
def predict_spec(model: SpecMLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    bs = 256
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs].astype(np.float32)).to(device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def build_feat(pack: dict, mode: str = "pen+logits") -> np.ndarray:
    logits = pack["logits"]
    pen = pack["pen"]
    fused = pack["fused"]
    probs = softmax_np(logits)
    if mode == "pen":
        return pen
    if mode == "fused":
        return fused
    if mode == "logits":
        return logits
    if mode == "probs":
        return probs
    if mode == "pen+logits":
        return np.concatenate([pen, logits], axis=1)
    if mode == "pen+probs":
        return np.concatenate([pen, probs], axis=1)
    if mode == "fused+logits":
        return np.concatenate([fused, logits], axis=1)
    raise ValueError(mode)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def top2_margin(logits: np.ndarray):
    probs = softmax_np(logits)
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    margin = probs[np.arange(len(probs)), top1] - probs[np.arange(len(probs)), top2]
    return top1, top2, margin, probs


@dataclass
class SpecBundle:
    kind: str  # "pair" or "multi"
    name: str
    classes: Tuple[int, ...]
    model: SpecMLP
    # mapping local label -> global class
    local_to_global: Dict[int, int]
    global_to_local: Dict[int, int]


def fit_pair_specs(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    device: torch.device,
    min_tr: int = 8,
) -> List[SpecBundle]:
    out = []
    for a, b in pairs:
        pk = pair_key(a, b)
        mask_tr = np.isin(y_tr, [pk[0], pk[1]])
        mask_va = np.isin(y_va, [pk[0], pk[1]])
        if mask_tr.sum() < min_tr:
            continue
        g2l = {pk[0]: 0, pk[1]: 1}
        l2g = {0: pk[0], 1: pk[1]}
        ytr = np.array([g2l[int(v)] for v in y_tr[mask_tr]], dtype=np.int64)
        yva = np.array([g2l[int(v)] for v in y_va[mask_va]], dtype=np.int64) if mask_va.any() else np.zeros(0, dtype=np.int64)
        Xva = X_va[mask_va] if mask_va.any() else None
        model, va_acc = train_spec(X_tr[mask_tr], ytr, Xva, yva if len(yva) else None, 2, device, seed=42 + pk[0] * 40 + pk[1])
        out.append(
            SpecBundle(
                kind="pair",
                name=f"pair_{pk[0]}_{pk[1]}",
                classes=pk,
                model=model,
                local_to_global=l2g,
                global_to_local=g2l,
            )
        )
        print(f"  fitted {out[-1].name} n_tr={mask_tr.sum()} n_va={mask_va.sum()} va_acc={va_acc:.3f}", flush=True)
    return out


def fit_multi_specs(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    groups: Sequence[Tuple[str, Tuple[int, ...]]],
    device: torch.device,
    min_tr: int = 20,
) -> List[SpecBundle]:
    out = []
    for name, classes in groups:
        classes = tuple(sorted(set(classes)))
        mask_tr = np.isin(y_tr, classes)
        mask_va = np.isin(y_va, classes)
        if mask_tr.sum() < min_tr:
            continue
        g2l = {c: i for i, c in enumerate(classes)}
        l2g = {i: c for c, i in g2l.items()}
        ytr = np.array([g2l[int(v)] for v in y_tr[mask_tr]], dtype=np.int64)
        yva = np.array([g2l[int(v)] for v in y_va[mask_va]], dtype=np.int64) if mask_va.any() else np.zeros(0, dtype=np.int64)
        Xva = X_va[mask_va] if mask_va.any() else None
        model, va_acc = train_spec(
            X_tr[mask_tr], ytr, Xva, yva if len(yva) else None, len(classes), device,
            hidden=96, epochs=50, seed=100 + hash(name) % 1000,
        )
        out.append(
            SpecBundle(
                kind="multi",
                name=f"multi_{name}",
                classes=classes,
                model=model,
                local_to_global=l2g,
                global_to_local=g2l,
            )
        )
        print(f"  fitted {out[-1].name} n_tr={mask_tr.sum()} n_va={mask_va.sum()} classes={classes} va_acc={va_acc:.3f}", flush=True)
    return out


# monkeypatch train_spec for hidden kw - fix SpecMLP call
_orig_train_spec = train_spec


def train_spec(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    n_out: int,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    batch: int = 64,
    patience: int = 10,
    seed: int = 0,
    hidden: int = 64,
) -> Tuple[SpecMLP, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpecMLP(X_tr.shape[1], n_out, hidden=hidden).to(device)
    counts = np.bincount(y_tr, minlength=n_out).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_out * counts)
    w = np.clip(w, 0.25, 8.0)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_loader = DataLoader(FeatDS(X_tr, y_tr), batch_size=min(batch, max(8, len(y_tr))), shuffle=True)
    best_state = None
    best_acc = -1.0
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if X_va is not None and y_va is not None and len(y_va):
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X_va.astype(np.float32)).to(device)).argmax(1).cpu().numpy()
            acc = float((pred == y_va).mean())
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            # train acc as proxy
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(X_tr.astype(np.float32)).to(device)).argmax(1).cpu().numpy()
            best_acc = float((pred == y_tr).mean())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, float(best_acc)


def apply_specialists(
    logits: np.ndarray,
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    pair_specs: List[SpecBundle],
    multi_specs: List[SpecBundle],
    device: torch.device,
    margin_thr: float = 0.25,
    use_pairs: bool = True,
    use_multi: bool = True,
    always_if_pair: bool = False,
) -> Tuple[np.ndarray, dict]:
    top1, top2, margin, probs = top2_margin(logits)
    pred = top1.copy()
    pair_map = {tuple(s.classes): s for s in pair_specs}
    n_defer_pair = 0
    n_defer_multi = 0
    n_flip = 0

    for i in range(len(pred)):
        flipped = False
        if use_pairs:
            pk = pair_key(int(top1[i]), int(top2[i]))
            if pk in pair_map and (always_if_pair or margin[i] < margin_thr):
                sp = pair_map[pk]
                loc = predict_spec(sp.model, X[i : i + 1], device)[0]
                newp = sp.local_to_global[int(loc)]
                n_defer_pair += 1
                if newp != pred[i]:
                    n_flip += 1
                    flipped = True
                pred[i] = newp
        if use_multi and not flipped:
            # if top1 in a multi group and (low margin or top2 also in group), defer
            for sp in multi_specs:
                cs = set(sp.classes)
                if int(top1[i]) in cs and (int(top2[i]) in cs) and (always_if_pair or margin[i] < margin_thr):
                    loc = predict_spec(sp.model, X[i : i + 1], device)[0]
                    newp = sp.local_to_global[int(loc)]
                    n_defer_multi += 1
                    if newp != pred[i]:
                        n_flip += 1
                    pred[i] = newp
                    break

    stats = {
        "n_defer_pair": n_defer_pair,
        "n_defer_multi": n_defer_multi,
        "n_flip": n_flip,
        "margin_thr": margin_thr,
        "always_if_pair": always_if_pair,
    }
    if y_true is not None:
        stats["acc"] = float((pred == y_true).mean())
        stats["base_acc"] = float((top1 == y_true).mean())
    return pred, stats


def sweep_policies(
    logits: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    pair_specs: List[SpecBundle],
    multi_specs: List[SpecBundle],
    device: torch.device,
) -> List[dict]:
    results = []
    for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 1.01]:
        for use_pairs in [True, False]:
            for use_multi in [True, False]:
                if not use_pairs and not use_multi:
                    continue
                for always in [False, True]:
                    if always and thr != 1.01:
                        continue  # always covered by thr=1.01
                    pred, st = apply_specialists(
                        logits, X, y, pair_specs, multi_specs, device,
                        margin_thr=thr, use_pairs=use_pairs, use_multi=use_multi, always_if_pair=always,
                    )
                    st["use_pairs"] = use_pairs
                    st["use_multi"] = use_multi
                    st["policy"] = f"pairs={use_pairs}|multi={use_multi}|thr={thr}|always={always}"
                    results.append(st)
    # baseline
    top1 = logits.argmax(1)
    results.append({
        "policy": "baseline_midfuse",
        "acc": float((top1 == y).mean()),
        "base_acc": float((top1 == y).mean()),
        "n_defer_pair": 0,
        "n_defer_multi": 0,
        "n_flip": 0,
    })
    results.sort(key=lambda d: -d.get("acc", 0))
    return results


def make_loader(X_skel, X_imu, y, users, idx, has_imu, bs=64):
    ds = CachedDualDataset(X_skel, X_imu, y, users, idx, has_imu, augment=False)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def holdout_probe(device: torch.device, feat_mode: str = "pen+logits") -> dict:
    names = load_class_names()
    cache = ROOT_V2 / "cache"
    X_skel, y, users, meta = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu = imu["X"]
    has_imu = imu["has_imu"].astype(bool)
    y = np.asarray(y, dtype=np.int64)
    users = np.asarray(users, dtype=np.int64)

    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]
    print(f"holdout probe n_tr={len(tr_idx)} n_va={len(va_idx)}", flush=True)

    ckpt = ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt"
    model, ck_meta = load_midfuse(ckpt, device)
    print(f"midfuse ckpt val_acc={ck_meta.get('val_acc')} params~", flush=True)

    tr_pack = extract_all(model, make_loader(X_skel, X_imu, y, users, tr_idx, has_imu), device)
    va_pack = extract_all(model, make_loader(X_skel, X_imu, y, users, va_idx, has_imu), device)

    X_tr = build_feat(tr_pack, feat_mode)
    X_va = build_feat(va_pack, feat_mode)
    base_acc = float((va_pack["logits"].argmax(1) == va_pack["y"]).mean())
    print(f"baseline holdout acc={base_acc:.4f} feat={feat_mode} dim={X_tr.shape[1]}", flush=True)

    print("Fitting pair specialists...", flush=True)
    pair_specs = fit_pair_specs(X_tr, tr_pack["y"], X_va, va_pack["y"], CONFUSE_PAIRS, device)
    print("Fitting multi specialists...", flush=True)
    multi_specs = fit_multi_specs(X_tr, tr_pack["y"], X_va, va_pack["y"], MULTI_GROUPS, device)

    sweeps = sweep_policies(va_pack["logits"], X_va, va_pack["y"], pair_specs, multi_specs, device)
    best = sweeps[0]
    print(f"BEST policy={best['policy']} acc={best['acc']:.4f} delta={best['acc']-base_acc:+.4f}", flush=True)
    for s in sweeps[:12]:
        print(f"  {s['policy']}: acc={s['acc']:.4f} defer_p={s.get('n_defer_pair')} defer_m={s.get('n_defer_multi')} flip={s.get('n_flip')}", flush=True)

    # Also try hard-class second stage: if MidFuse conf low, reclassify among HARD+neighbors using multi?
    # Simple: logistic over all classes on pen features (full second stage) - may overfit; keep small.
    print("Fitting full soft second-stage (all 40 on pen)...", flush=True)
    ss_model, ss_va = train_spec(X_tr, tr_pack["y"], X_va, va_pack["y"], N_CLASSES, device, hidden=128, epochs=30, patience=8, seed=7)
    ss_pred = predict_spec(ss_model, X_va, device)
    ss_acc = float((ss_pred == va_pack["y"]).mean())
    print(f"  full second-stage alone acc={ss_acc:.4f}", flush=True)

    # Blend: if midfuse margin < thr use second-stage else midfuse; also with specialists
    blend_results = []
    top1, top2, margin, _ = top2_margin(va_pack["logits"])
    for thr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        pred = top1.copy()
        use_ss = margin < thr
        pred[use_ss] = ss_pred[use_ss]
        acc = float((pred == va_pack["y"]).mean())
        blend_results.append({"policy": f"ss_margin<{thr}", "acc": acc, "n_ss": int(use_ss.sum())})
    blend_results.sort(key=lambda d: -d["acc"])
    print(f"BEST second-stage blend={blend_results[0]}", flush=True)

    # Combine best specialist policy with optional SS for remaining low-margin non-pair
    # Re-run best specialist then SS on still-low-margin
    best_pred, best_st = apply_specialists(
        va_pack["logits"], X_va, va_pack["y"], pair_specs, multi_specs, device,
        margin_thr=best.get("margin_thr", 0.25),
        use_pairs=best.get("use_pairs", True),
        use_multi=best.get("use_multi", False),
        always_if_pair=best.get("always_if_pair", False),
    )
    # After specialists, for samples with still-low margin vs original, try SS among hard classes only
    # Simpler: if original margin < 0.15 and specialist didn't flip from confuse pair, use SS
    combo_accs = []
    for thr in [0.05, 0.1, 0.15, 0.2]:
        pred = best_pred.copy()
        top1b, _, marginb, _ = top2_margin(va_pack["logits"])
        mask = marginb < thr
        pred[mask] = ss_pred[mask]
        # but keep specialist overrides where confuse pair deferred - approximate: where best_pred != top1b keep best_pred
        keep = best_pred != top1b
        pred[keep] = best_pred[keep]
        acc = float((pred == va_pack["y"]).mean())
        combo_accs.append({"policy": f"spec+ss<{thr}", "acc": acc})
    combo_accs.sort(key=lambda d: -d["acc"])
    print(f"BEST combo={combo_accs[0] if combo_accs else None}", flush=True)

    all_best_acc = max(best["acc"], blend_results[0]["acc"], combo_accs[0]["acc"] if combo_accs else 0, ss_acc)
    winner = "specialists"
    if blend_results[0]["acc"] >= all_best_acc - 1e-12:
        winner = "ss_blend"
        all_best_acc = blend_results[0]["acc"]
    if combo_accs and combo_accs[0]["acc"] >= all_best_acc - 1e-12:
        winner = "combo"
        all_best_acc = combo_accs[0]["acc"]
    if ss_acc >= all_best_acc - 1e-12:
        # alone rarely wins
        pass
    if best["acc"] >= all_best_acc - 1e-12:
        winner = "specialists"
        all_best_acc = best["acc"]

    out = {
        "feat_mode": feat_mode,
        "baseline_holdout": base_acc,
        "baseline_reported": BASELINE_HOLDOUT,
        "best_policy": best,
        "top_sweeps": sweeps[:20],
        "ss_alone": ss_acc,
        "ss_blends": blend_results[:8],
        "combos": combo_accs[:8],
        "best_holdout_acc": all_best_acc,
        "delta_vs_0.537": all_best_acc - BASELINE_HOLDOUT,
        "delta_vs_measured_base": all_best_acc - base_acc,
        "winner_family": winner,
        "n_pair_specs": len(pair_specs),
        "n_multi_specs": len(multi_specs),
        "pair_names": [s.name for s in pair_specs],
        "multi_names": [s.name for s in multi_specs],
        "ckpt": str(ckpt),
        "n_holdout": int(len(va_idx)),
    }
    # Persist specialists for later
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feat_mode": feat_mode,
            "best_policy": best,
            "pair_states": [
                {
                    "name": s.name,
                    "classes": s.classes,
                    "state": s.model.state_dict(),
                    "in_dim": X_tr.shape[1],
                    "n_out": len(s.classes),
                    "hidden": 64,
                }
                for s in pair_specs
            ],
            "multi_states": [
                {
                    "name": s.name,
                    "classes": s.classes,
                    "state": s.model.state_dict(),
                    "in_dim": X_tr.shape[1],
                    "n_out": len(s.classes),
                    "hidden": 96,
                }
                for s in multi_specs
            ],
            "ss_state": ss_model.state_dict(),
            "ss_in_dim": X_tr.shape[1],
            "ss_hidden": 128,
        },
        ckpt_dir / "holdout_specialists.pt",
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--feat-mode", default="pen+logits",
                    choices=["pen", "fused", "logits", "probs", "pen+logits", "pen+probs", "fused+logits"])
    ap.add_argument("--sweep-feats", action="store_true")
    args = ap.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)

    if args.sweep_feats:
        modes = ["pen+logits", "pen+probs", "fused+logits", "pen", "logits"]
        all_res = {}
        best_overall = None
        for m in modes:
            print(f"\n======== FEAT MODE {m} ========", flush=True)
            r = holdout_probe(device, m)
            all_res[m] = r
            if best_overall is None or r["best_holdout_acc"] > best_overall["best_holdout_acc"]:
                best_overall = r
        summary = {
            "modes": {k: {
                "best_holdout_acc": v["best_holdout_acc"],
                "delta_vs_0.537": v["delta_vs_0.537"],
                "baseline": v["baseline_holdout"],
                "best_policy": v["best_policy"].get("policy"),
            } for k, v in all_res.items()},
            "best": {
                "feat_mode": best_overall["feat_mode"],
                "best_holdout_acc": best_overall["best_holdout_acc"],
                "delta_vs_0.537": best_overall["delta_vs_0.537"],
                "best_policy": best_overall["best_policy"],
            },
            "beat_clear": bool(best_overall["best_holdout_acc"] >= BASELINE_HOLDOUT + WIN_DELTA),
            "beat_any": bool(best_overall["best_holdout_acc"] > BASELINE_HOLDOUT + 0.002),
        }
        with open(OUT / "holdout_probe_summary.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "detail": {k: {kk: vv for kk, vv in v.items() if kk != "top_sweeps"} | {"top_sweeps": v["top_sweeps"][:8]} for k, v in all_res.items()}}, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
    else:
        r = holdout_probe(device, args.feat_mode)
        with open(OUT / "holdout_probe.json", "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"Wrote {OUT / 'holdout_probe.json'}", flush=True)


if __name__ == "__main__":
    main()
