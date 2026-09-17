"""IR v15 upgrade on ir_yolo_v4: multi-view hold-aware TTA + 2 stronger diversity seeds + nested fuse.
Write submission_ir_v15.csv ONLY if hold AND nested_fixed >= ir_v7+0.01 (~0.763) AND disagree>=20.
Else metrics-only; keep ir_v7. Prefer sameT / nested LOUO (avoid perT overfit).
"""
from __future__ import annotations
import argparse, csv, json, random, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score

from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES
from fuse_ir_v9 import (
    load_members, softmax_np, nested_fixed, write_sub, fuse3_sameT, fuse3_geom, conf_gate_blend,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
HOLD = set(DEFAULT_HOLD_OUT_USERS)
V7_HOLD = 0.7530364372469636
GATE = V7_HOLD + 0.01
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}

K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

CKPT_JOBS = [
    # (ckpt_path, tag, out_stem_dir)
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed11.pt", "pool_seed11", "v5"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed123.pt", "pool_seed123", "v5"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed2024.pt", "pool_seed2024", "v5"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed42.pt", "pool_seed42", "v5"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed7.pt", "pool_seed7", "v5"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed99.pt", "pool_seed99", "v5"),
    ("checkpoints/ir_yolo_r2p1d18_v6/pool_seed1.pt", "pool_seed1", "v6"),
    ("checkpoints/ir_yolo_r2p1d18_v6/pool_seed333.pt", "pool_seed333", "v6"),
    ("checkpoints/ir_yolo_r2p1d18_v6/pool_seed777.pt", "pool_seed777", "v6"),
    ("checkpoints/ir_yolo_r2p1d18_v7/pool_seed55.pt", "pool_seed55", "v7"),
    ("checkpoints/ir_yolo_r2p1d18_v13/pool_seed4096.pt", "v13_seed4096", "v13"),
    ("checkpoints/ir_yolo_r2p1d18_v13/pool_seed999.pt", "v13_seed999", "v13"),
    ("checkpoints/ir_yolo_r2p1d18_v13/pool_seed2026.pt", "v13_seed2026", "v13"),
]

NEW_SEEDS = [555, 2048]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained=True):
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def load_state(ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in blob:
        return blob["model"], float(blob.get("val_acc", -1))
    return blob, -1.0


# ---------- multi-view TTA helpers ----------
def _resize_clip(arr, size=112):
    """arr: B,T,H,W,C float32 -> resize spatial to size via torch."""
    # use nearest/bilinear on NCHW frames
    b, t, h, w, c = arr.shape
    x = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.reshape(b, t, c, size, size).permute(0, 1, 3, 4, 2).contiguous().numpy()
    return x


def make_views_uint8(batch_uint8):
    """Mild hold-aware TTA views (avoid aggressive corner/reverse that hurt YOLO crops).
    Views: identity, hflip, center-crop 0.92, center-crop 0.92 + hflip.
    """
    arr = batch_uint8.astype(np.float32) / 255.0  # B,T,H,W,C
    b, t, h, w, c = arr.shape
    views = []

    def to_btchw(a):
        return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 1, 4, 2, 3)))

    views.append(to_btchw(arr))  # identity
    views.append(to_btchw(arr[:, :, :, ::-1, :].copy()))  # hflip

    ch = int(round(h * 0.92)); cw = int(round(w * 0.92))
    top = (h - ch) // 2; left = (w - cw) // 2
    crop = arr[:, :, top:top + ch, left:left + cw, :]
    crop_r = _resize_clip(crop, h)
    views.append(to_btchw(crop_r))
    views.append(to_btchw(crop_r[:, :, :, ::-1, :].copy()))
    return views


@torch.no_grad()
def infer_multiview(model, X_uint8, device, bs=4):
    model.eval()
    n = len(X_uint8)
    out = np.zeros((n, NUM_CLASSES), np.float32)
    for i in range(0, n, bs):
        batch = np.asarray(X_uint8[i:i + bs])  # B,T,H,W,C
        view_logits = []
        for v in make_views_uint8(batch):
            x = normalize(v.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            view_logits.append(model(x).float())
        logits = torch.stack(view_logits, 0).mean(0)
        out[i:i + len(batch)] = logits.cpu().numpy()
        del view_logits, logits
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {"acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "logits": np.concatenate(logits_all), "y": yt}


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience,
              ckpt_path, mixup_alpha=0.3, unfreeze_ep=3, label_smoothing=0.05):
    set_seed(seed)
    tag = f"seed{seed}"
    model = build(True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ["stem", "layer1"]):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=WeightedRandomSampler(w, num_samples=len(pool_idx), replacement=True),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all", flush=True)
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            if mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb_m = lam * xb + (1 - lam) * xb[idx]
                y1, y2 = yb, yb[idx]
            else:
                lam, xb_m, y1, y2 = 1.0, xb, yb, yb
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb_m)
                if lam < 1.0:
                    loss = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2)
                else:
                    loss = crit(logits, yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, "tr_loss": loss_sum / max(n, 1), "tr_acc": correct / max(n, 1),
               "val_acc": metrics["acc"], "val_f1": metrics["macro_f1"]}
        history.append(row)
        print(f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
              f"val={row['val_acc']:.4f} f1={row['val_f1']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc,
                        "epoch": ep, "seed": seed, "history": history, "hold_logits": metrics["logits"],
                        "hold_y": metrics["y"], "recipe": {"mixup": mixup_alpha, "unfreeze_ep": unfreeze_ep,
                        "label_smoothing": label_smoothing, "patience": patience}}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state); model.to(device)
    hold_m = evaluate(model, val_loader, device)
    del model; torch.cuda.empty_cache()
    return best_acc, hold_m["logits"]


def apply_cfg(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
         + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean())


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=21, leave_users=(8, 9, 24)):
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": acc,
                      "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def build_variants(members):
    stack = np.stack([m["logits"] for m in members], 0)
    w = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64)
    w /= w.sum()
    variants = {}
    for k in range(3, len(members) + 1):
        variants[f"top{k}"] = np.mean(stack[:k], 0)
    variants["all_mean"] = np.mean(stack, 0)
    variants["all_acc_w"] = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)
    sm = np.stack([softmax_np(m["logits"], 1.0) for m in members], 0)
    variants["sm_mean"] = np.log(np.mean(sm, 0) + 1e-8).astype(np.float32)
    return variants, stack, w


def phase_tta(device, cache, out_dir, yt_ref, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.load(cache / "train_y.npy")
    users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    assert np.array_equal(y[hold_idx], yt_ref), "hold y mismatch vs v7 members"
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), 16, 112, 112, 3))
    print(f"[TTA] hold={len(hold_idx)} test={len(meta)} views=4(mild)", flush=True)

    results = []
    X_hold = np.asarray(X[hold_idx])  # materialize hold once (~505*16*112*112*3 ~ 290MB)
    print(f"[TTA] materialized hold X {X_hold.shape} {X_hold.nbytes/1e6:.0f}MB", flush=True)

    for rel, tag, _family in CKPT_JOBS:
        ck = ROOT / rel
        hold_p = out_dir / f"hold_mv_{tag}.npy"
        test_p = out_dir / f"test_mv_{tag}.npy"
        if hold_p.exists() and test_p.exists() and not force:
            hl = np.load(hold_p)
            acc = float((hl.argmax(1) == yt_ref).mean())
            print(f"[TTA] reuse {tag} acc={acc:.4f}", flush=True)
            results.append({"tag": tag, "acc": acc, "hold": str(hold_p), "test": str(test_p)})
            continue
        if not ck.exists():
            print(f"[TTA] MISSING {ck}", flush=True)
            continue
        state, vacc = load_state(ck)
        model = build(False); model.load_state_dict(state); model.to(device); model.eval()
        t0 = time.time()
        print(f"[TTA] {tag} (ckpt_val={vacc:.4f}) ...", flush=True)
        hl = infer_multiview(model, X_hold, device, bs=3)
        tl = infer_multiview(model, Xt, device, bs=3)
        # also base (identity-only) for selective pick
        with torch.no_grad():
            base = np.zeros((len(X_hold), NUM_CLASSES), np.float32)
            flip = np.zeros_like(base)
            for i0 in range(0, len(X_hold), 4):
                arr = X_hold[i0:i0+4].astype(np.float32) / 255.0
                x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0,1,4,2,3))).to(device)
                xn = normalize(x).permute(0,2,1,3,4).contiguous()
                base[i0:i0+len(x)] = model(xn).float().cpu().numpy()
                xf = torch.flip(x, dims=[-1])
                xfn = normalize(xf).permute(0,2,1,3,4).contiguous()
                flip[i0:i0+len(x)] = model(xfn).float().cpu().numpy()
            # test base/flip
            tb = np.zeros((len(Xt), NUM_CLASSES), np.float32)
            tf = np.zeros_like(tb)
            for i0 in range(0, len(Xt), 4):
                arr = np.asarray(Xt[i0:i0+4]).astype(np.float32) / 255.0
                x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0,1,4,2,3))).to(device)
                xn = normalize(x).permute(0,2,1,3,4).contiguous()
                tb[i0:i0+len(x)] = model(xn).float().cpu().numpy()
                xf = torch.flip(x, dims=[-1])
                xfn = normalize(xf).permute(0,2,1,3,4).contiguous()
                tf[i0:i0+len(x)] = model(xfn).float().cpu().numpy()
        cand = {
            'base': (base, tb),
            'flip': (0.5*(base+flip), 0.5*(tb+tf)),
            'mild4': (hl, tl),
        }
        best_name, best_acc, best_h, best_t = None, -1.0, None, None
        for name, (hlog, tlog) in cand.items():
            a = float((hlog.argmax(1) == yt_ref).mean())
            if a > best_acc:
                best_name, best_acc, best_h, best_t = name, a, hlog, tlog
        np.save(hold_p, best_h.astype(np.float32))
        np.save(test_p, best_t.astype(np.float32))
        acc = best_acc
        print(f"[TTA] {tag} pick={best_name} acc={acc:.4f} (base={(base.argmax(1)==yt_ref).mean():.4f} flip={(0.5*(base+flip)).argmax(1).mean() if False else (cand['flip'][0].argmax(1)==yt_ref).mean():.4f} mild4={(hl.argmax(1)==yt_ref).mean():.4f}) took={time.time()-t0:.1f}s", flush=True)
        results.append({"tag": tag, "acc": acc, "pick": best_name, "hold": str(hold_p), "test": str(test_p)})
        del model; torch.cuda.empty_cache(); gc.collect()
    del X_hold; gc.collect()
    (out_dir / "tta_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def phase_train(device, cache, ckpt_dir, seeds, epochs=36, patience=12, mixup=0.3, unfreeze_ep=3):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), 16, 112, 112, 3))
    members = []
    for seed in seeds:
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            acc = float(blob.get("val_acc", -1))
            hlog = blob.get("hold_logits")
            print(f"[TRAIN] reuse seed{seed} val={acc:.4f}", flush=True)
            if hlog is None:
                model = build(False); model.load_state_dict(blob["model"]); model.to(device)
                val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False),
                                        batch_size=6, shuffle=False)
                hlog = evaluate(model, val_loader, device)["logits"]
                del model; torch.cuda.empty_cache()
        else:
            acc, hlog = train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, 6,
                                  1e-4, patience, ck, mixup_alpha=mixup, unfreeze_ep=unfreeze_ep)
        # multiview TTA for new seed
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        model = build(False); model.load_state_dict(blob["model"]); model.to(device)
        X_hold = np.asarray(X[hold_idx])
        t0 = time.time()
        hl_mv = infer_multiview(model, X_hold, device, bs=3)
        tl_mv = infer_multiview(model, Xt, device, bs=3)
        tl_base = None
        # also save base test
        with torch.no_grad():
            base = np.zeros((len(Xt), NUM_CLASSES), np.float32)
            for i in range(0, len(Xt), 6):
                arr = Xt[i:i + 6].astype(np.float32) / 255.0
                x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
                x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
                base[i:i + len(x)] = model(x).float().cpu().numpy()
        np.save(ckpt_dir / f"test_logits_seed{seed}.npy", base)
        np.save(ckpt_dir / f"hold_mv_seed{seed}.npy", hl_mv)
        np.save(ckpt_dir / f"test_mv_seed{seed}.npy", tl_mv)
        acc_mv = float((hl_mv.argmax(1) == y[hold_idx]).mean())
        acc_base = float((np.asarray(hlog).argmax(1) == y[hold_idx]).mean())
        print(f"[TRAIN] seed{seed} base={acc_base:.4f} mv={acc_mv:.4f} tta_took={time.time()-t0:.1f}s", flush=True)
        use = hl_mv if acc_mv >= acc_base else np.asarray(hlog)
        test_use = tl_mv if acc_mv >= acc_base else base
        members.append({
            "tag": f"v15_seed{seed}",
            "logits": use.astype(np.float32),
            "base": np.asarray(hlog).astype(np.float32),
            "acc": float((use.argmax(1) == y[hold_idx]).mean()),
            "acc_base": acc_base,
            "acc_mv": acc_mv,
            "test_logits": test_use.astype(np.float32),
            "source": "v15",
        })
        del model, X_hold; torch.cuda.empty_cache(); gc.collect()
    np.savez(ckpt_dir / "hold_logits_v15.npz",
             logits=np.stack([m["logits"] for m in members], 0),
             y=y[hold_idx], users=users[hold_idx],
             seeds=np.array(seeds),
             scores=np.array([m["acc"] for m in members]))
    return members


def load_mv_members(tta_dir, yt, yu):
    members = []
    for _rel, tag, _fam in CKPT_JOBS:
        hp = tta_dir / f"hold_mv_{tag}.npy"
        tp = tta_dir / f"test_mv_{tag}.npy"
        if not hp.exists():
            continue
        hl = np.load(hp).astype(np.float32)
        tl = np.load(tp).astype(np.float32)
        assert len(hl) == len(yt)
        acc = float((hl.argmax(1) == yt).mean())
        members.append({
            "tag": f"mv_{tag}", "logits": hl, "acc": acc,
            "test_logits": tl, "source": "mv_tta",
        })
    return sorted(members, key=lambda d: -d["acc"])


def phase_fuse(mv_members, new_members, yt, yu):
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx] if len(mid_full) == len(tu) else mid_full
    assert len(yt) == len(th) == len(mid)
    mask = th.any(1) & mid.any(1)
    print(f"[FUSE] mask n={int(mask.sum())}/{len(yt)}", flush=True)

    classic_mv = [m for m in mv_members if not m["tag"].startswith("mv_v13_")]
    v13_mv = [m for m in mv_members if m["tag"].startswith("mv_v13_")]
    # also load selective classic from load_members for baseline compare
    sel_members, _, _ = load_members()
    classic_sel = [m for m in sel_members if m["tag"] != "pool_seed55"]
    seed55_sel = [m for m in sel_members if m["tag"] == "pool_seed55"]

    pools = {
        "classic9_mv": sorted(classic_mv, key=lambda d: -d["acc"])[:9] if len(classic_mv) >= 9 else sorted(classic_mv, key=lambda d: -d["acc"]),
        "classic_all_mv": sorted(classic_mv, key=lambda d: -d["acc"]),
        "classic_mv_plus_v13mv": sorted(classic_mv + v13_mv, key=lambda d: -d["acc"]),
        "classic_mv_plus_v15": sorted(classic_mv + new_members, key=lambda d: -d["acc"]),
        "all_mv_v13_v15": sorted(classic_mv + v13_mv + new_members, key=lambda d: -d["acc"]),
        "sel9_plus_mv_top3": sorted(classic_sel + sorted(mv_members, key=lambda d: -d["acc"])[:3], key=lambda d: -d["acc"]),
        "sel9_plus_v15": sorted(classic_sel + new_members, key=lambda d: -d["acc"]),
        "sel10_plus_v15_v13mv": sorted(classic_sel + seed55_sel + v13_mv + new_members, key=lambda d: -d["acc"]),
    }
    # filter empty
    pools = {k: v for k, v in pools.items() if len(v) >= 3}

    Ts_fine = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
    Ts_med = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    results = []

    for pool_name, mems in pools.items():
        variants, stack, w = build_variants(mems)
        for ens_name, elogs in variants.items():
            if ens_name not in ("all_mean", "all_acc_w", "sm_mean",
                                f"top{min(9, len(mems))}", f"top{min(6, len(mems))}",
                                f"top{min(5, len(mems))}", f"top{min(12, len(mems))}"):
                if not ens_name.startswith("top"):
                    continue
                # keep only a few tops
                k = int(ens_name.replace("top", ""))
                if k not in {5, 6, 8, 9, 10, 12, len(mems)}:
                    continue
            ens_acc = float((elogs.argmax(1) == yt).mean())
            if ens_name.startswith("sm_"):
                ens_acc = float((np.exp(elogs).argmax(1) == yt).mean())
            b_acc, bcfg = fuse3_sameT(elogs, th, mid, yt, mask, Ts_fine, ngrid=41)
            nest_fixed = nested_fixed(elogs, th, mid, yt, yu, mask, bcfg)
            nest_ret = nested_retune(elogs, th, mid, yt, yu, mask, Ts_med, ngrid=21)
            # conf-calibrated blend of IR into thermal/mid (nested-validated later)
            conf_acc, conf_cfg = conf_gate_blend(elogs, th, yt, mask,
                                                 gate_probs=[0.35, 0.45, 0.55, 0.65, 0.75],
                                                 aux_w_grid=[0.1, 0.2, 0.3, 0.4], T=bcfg["T"])
            row = {
                "pool": pool_name, "ens": ens_name, "ens_acc": ens_acc,
                "n_members": len(mems),
                "triple_acc": b_acc, "triple_cfg": bcfg,
                "nested_fixed_mean": nest_fixed["mean"],
                "nested_fixed_folds": nest_fixed["folds"],
                "nested_retune_mean": nest_ret["mean"],
                "conf_gate_acc": conf_acc, "conf_gate_cfg": conf_cfg,
                "gate_hold": b_acc, "gate_nested": nest_fixed["mean"],
                "clears_metric_gate": bool(b_acc >= GATE and nest_fixed["mean"] >= GATE),
            }
            results.append(row)
            print(f"{pool_name}/{ens_name}: ens={ens_acc:.4f} trip={b_acc:.4f} "
                  f"nestF={nest_fixed['mean']:.4f} nestR={nest_ret['mean']:.4f} "
                  f"conf={conf_acc:.4f} gate={row['clears_metric_gate']}", flush=True)

    # nested-oriented refine on top pools
    refine = []
    for pool_name in list(pools.keys())[:6]:
        mems = pools[pool_name]
        variants, _, _ = build_variants(mems)
        for ens_name in ["all_mean", "all_acc_w", f"top{min(9, len(mems))}", f"top{min(6, len(mems))}"]:
            if ens_name not in variants:
                continue
            elogs = variants[ens_name]
            best_n = (-1.0, None)
            for T in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]:
                for wa in np.linspace(0.40, 0.75, 15):
                    for wb in np.linspace(0.15, 0.50, 15):
                        wc = 1.0 - wa - wb
                        if wc < 0.02 or wc > 0.35:
                            continue
                        cfg = {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T), "mode": "sameT"}
                        full = apply_cfg(elogs, th, mid, yt, mask, cfg)
                        nest = nested_fixed(elogs, th, mid, yt, yu, mask, cfg)["mean"]
                        if nest > best_n[0] or (abs(nest - best_n[0]) < 1e-12 and full > (best_n[1]["full"] if best_n[1] else -1)):
                            best_n = (nest, {"cfg": {**cfg, "acc": full, "n": int(mask.sum())},
                                             "full": full, "nested": nest})
            bn = best_n[1]
            if bn:
                refine.append({"pool": pool_name, "ens": ens_name, **bn,
                               "clears": bool(bn["full"] >= GATE and bn["nested"] >= GATE)})
                print(f"refine {pool_name}/{ens_name}: full={bn['full']:.4f} nest={bn['nested']:.4f} "
                      f"clears={bn['full'] >= GATE and bn['nested'] >= GATE}", flush=True)

    candidates = []
    for r in results:
        candidates.append({"source": "grid", "pool": r["pool"], "ens": r["ens"],
                           "full": r["gate_hold"], "nested": r["gate_nested"],
                           "cfg": r["triple_cfg"], "members": r["n_members"]})
    for r in refine:
        candidates.append({"source": "refine_nested", "pool": r["pool"], "ens": r["ens"],
                           "full": r["full"], "nested": r["nested"],
                           "cfg": r["cfg"], "members": len(pools[r["pool"]])})
    candidates = sorted(candidates, key=lambda c: (min(c["full"], c["nested"]), c["full"], c["nested"]), reverse=True)
    top = candidates[0]
    print(f"\nTOP: {top['source']} {top['pool']}/{top['ens']} "
          f"full={top['full']:.6f} nest={top['nested']:.6f}", flush=True)

    # temperature-scaled IR self-blend probe (nested)
    # pick best IR ens logits for conf blend with thermal as primary fallback
    mems = pools[top["pool"]]
    variants, _, w = build_variants(mems)
    elogs = variants.get(top["ens"], variants["all_mean"])
    # conf gate: when IR conf low, lean thermal — validate nested on fixed cfg from hold
    conf_best = (-1.0, None)
    for T in [1.5, 2.0, 2.5, 3.0]:
        for thr in [0.4, 0.5, 0.6, 0.7]:
            for w_aux in [0.15, 0.25, 0.35]:
                # build blended logits-as-probs then treat as primary for triple? simpler: blend IR+TH first then +mid
                pp = softmax_np(elogs, T); ap = softmax_np(th, T)
                mx = pp.max(1)
                blend = pp.copy()
                low = mx < thr
                blend[low] = (1 - w_aux) * pp[low] + w_aux * ap[low]
                # convert back to logits via log
                ir_cal = np.log(np.clip(blend, 1e-8, 1)).astype(np.float32)
                for wa in [0.5, 0.55, 0.6]:
                    for wb in [0.25, 0.3, 0.35]:
                        wc = 1 - wa - wb
                        if wc < 0.05:
                            continue
                        cfg = {"wa": wa, "wb": wb, "wc": wc, "T": 1.0, "mode": "conf_cal"}
                        # here a=ir_cal already softmaxed-ish; use T=1 on ir_cal/th/mid carefully
                        # reuse fuse with ir_cal as logits
                        full = apply_cfg(ir_cal, th, mid, yt, mask, {"wa": wa, "wb": wb, "wc": wc, "T": 1.0})
                        nest = nested_fixed(ir_cal, th, mid, yt, yu, mask, {"wa": wa, "wb": wb, "wc": wc, "T": 1.0})["mean"]
                        score = min(full, nest)
                        if score > conf_best[0]:
                            conf_best = (score, {"full": full, "nested": nest, "thr": thr, "w_aux": w_aux,
                                                 "T_ir": T, "cfg": {"wa": wa, "wb": wb, "wc": wc, "T": 1.0},
                                                 "ir_cal": ir_cal})
    conf_info = None
    if conf_best[1]:
        conf_info = {k: v for k, v in conf_best[1].items() if k != "ir_cal"}
        print(f"conf_cal best: full={conf_best[1]['full']:.4f} nest={conf_best[1]['nested']:.4f} {conf_info}", flush=True)
        candidates.append({"source": "conf_cal", "pool": top["pool"], "ens": top["ens"],
                           "full": conf_best[1]["full"], "nested": conf_best[1]["nested"],
                           "cfg": conf_best[1]["cfg"], "members": len(mems),
                           "extra": {k: conf_best[1][k] for k in ("thr", "w_aux", "T_ir")}})
        candidates = sorted(candidates, key=lambda c: (min(c["full"], c["nested"]), c["full"], c["nested"]), reverse=True)
        top = candidates[0]
        print(f"TOP after conf: {top['source']} full={top['full']:.6f} nest={top['nested']:.6f}", flush=True)

    # build test preds for top
    mems = pools[top["pool"]]
    variants, _, w = build_variants(mems)
    test_stack = np.stack([m["test_logits"] for m in mems], 0)
    ens = top["ens"]
    if ens.startswith("top"):
        k = int(ens.replace("top", ""))
        ir_test = np.mean(test_stack[:k], 0)
    elif ens == "all_acc_w":
        ir_test = np.tensordot(w, test_stack, axes=(0, 0)).astype(np.float32)
    elif ens == "sm_mean":
        sm = np.stack([softmax_np(t, 1.0) for t in test_stack], 0)
        ir_test = np.log(np.mean(sm, 0) + 1e-8).astype(np.float32)
    else:
        ir_test = np.mean(test_stack, 0)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_p.exists():
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_p)

    cfg = top["cfg"]
    if top["source"] == "conf_cal" and conf_best[1]:
        # apply same conf cal on test
        T_ir = conf_best[1]["T_ir"]; thr = conf_best[1]["thr"]; w_aux = conf_best[1]["w_aux"]
        pp = softmax_np(ir_test, T_ir); ap = softmax_np(th_test, T_ir)
        mx = pp.max(1); blend = pp.copy(); low = mx < thr
        blend[low] = (1 - w_aux) * pp[low] + w_aux * ap[low]
        ir_use = np.log(np.clip(blend, 1e-8, 1)).astype(np.float32)
        T = 1.0
        probs = (cfg["wa"] * softmax_np(ir_use, T) + cfg["wb"] * softmax_np(th_test, T)
                 + cfg["wc"] * softmax_np(mid_test, T))
    else:
        T = cfg["T"]
        probs = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T)
                 + cfg["wc"] * softmax_np(mid_test, T))
    preds = probs.argmax(1)

    v7_preds = []
    with open(ROOT / "submission_ir_v7.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v7_preds.append(int(row["prediction"]))
    disagree = int(sum(int(a) != int(b) for a, b in zip(preds, v7_preds)))
    print(f"disagree vs ir_v7: {disagree}", flush=True)

    clears = bool(top["full"] >= GATE and top["nested"] >= GATE and disagree >= MIN_DISAGREE)
    wrote_csv = False
    out_csv = None
    if clears:
        cache = ROOT / "cache" / "ir_yolo_v4"
        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        empty = set()
        ep = cache / "test_empty.json"
        if ep.exists():
            empty = set(json.loads(ep.read_text(encoding="utf-8")))
        fb = {}
        out_csv = ROOT / "submission_ir_v15.csv"
        nfb = write_sub(out_csv, meta, preds, empty, fb)
        wrote_csv = True
        print(f"WROTE {out_csv} empty_fb={nfb} GATE CLEAR", flush=True)
    else:
        print("GATE NOT CLEARED — metrics only, keep ir_v7", flush=True)

    status = {
        "tag": "ir_v15_upgrade",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "top_candidate": {k: v for k, v in top.items() if k != "extra"},
        "disagree_vs_v7": disagree,
        "clears_gate": clears,
        "wrote_csv": wrote_csv,
        "csv": str(out_csv) if out_csv else None,
        "mv_member_acc": {m["tag"]: m["acc"] for m in mv_members},
        "v15_member_acc": {m["tag"]: m["acc"] for m in new_members},
        "grid_top5": [{k: r[k] for k in ("pool", "ens", "triple_acc", "nested_fixed_mean", "ens_acc", "n_members")
                       if k in r} for r in sorted(results, key=lambda x: min(x["gate_hold"], x["gate_nested"]), reverse=True)[:5]],
        "refine_top3": refine[:3] if refine else [],
        "conf_cal": conf_info,
        "next_roi": [] if clears else [
            "Multi-view TTA + stronger seeds did not clear +0.01 nested gate",
            "Consider longer train (epochs 48+) on best new seed only, or temporal-stride TTA",
            "Avoid YOLO11n/Depth; stay ir_yolo_v4",
            "If IR ens plateaus ~0.71-0.72, bottleneck may be Thermal/Mid fusion not IR",
        ],
    }
    return status, results, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tta", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force-tta", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=NEW_SEEDS)
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--mixup", type=float, default=0.3)
    args = ap.parse_args()

    t0 = time.time()
    free, total = torch.cuda.mem_get_info()
    used_mb = (total - free) / (1024 * 1024)
    print(f"GPU used~{used_mb:.0f}MB / {total/1024/1024:.0f}MB", flush=True)
    if used_mb > 2500:
        print("GPU busy; abort", flush=True)
        return
    device = torch.device("cuda")
    cache = ROOT / "cache" / "ir_yolo_v4"
    tta_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v15_tta"
    train_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v15"

    # reference y/users from classic members
    z = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_logits_v6.npz", allow_pickle=True)
    yt, yu = z["y"], z["users"]

    if not args.skip_tta:
        print("==== PHASE A: multi-view TTA ====", flush=True)
        mv_summary = phase_tta(device, cache, tta_dir, yt, force=args.force_tta)
    else:
        mv_summary = json.loads((tta_dir / "tta_summary.json").read_text(encoding="utf-8"))

    if not args.skip_train:
        print("==== PHASE B: diversity seeds ====", flush=True)
        new_members = phase_train(device, cache, train_dir, args.seeds,
                                  epochs=args.epochs, patience=args.patience, mixup=args.mixup)
    else:
        new_members = []
        if (train_dir / "hold_logits_v15.npz").exists():
            zz = np.load(train_dir / "hold_logits_v15.npz", allow_pickle=True)
            for i, seed in enumerate(zz["seeds"]):
                seed = int(seed)
                hl = np.load(train_dir / f"hold_mv_seed{seed}.npy")
                tl = np.load(train_dir / f"test_mv_seed{seed}.npy")
                new_members.append({
                    "tag": f"v15_seed{seed}", "logits": hl, "acc": float((hl.argmax(1) == yt).mean()),
                    "test_logits": tl, "source": "v15",
                })

    print("==== PHASE C: fuse retune ====", flush=True)
    mv_members = load_mv_members(tta_dir, yt, yu)
    print("MV members:", [(m["tag"], round(m["acc"], 4)) for m in mv_members], flush=True)
    print("V15 members:", [(m["tag"], round(m["acc"], 4)) for m in new_members], flush=True)

    status, results, top = phase_fuse(mv_members, new_members, yt, yu)
    status["elapsed_sec"] = round(time.time() - t0, 1)
    status["best_public"] = {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD}
    (ROOT / "metrics_ir_v15_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    handoff = ROOT / "logs" / "gpu_handoff_ir_v15.txt"
    handoff.parent.mkdir(exist_ok=True)
    # ensure idle
    torch.cuda.empty_cache()
    free2, total2 = torch.cuda.mem_get_info()
    used2 = (total2 - free2) / (1024 * 1024)
    handoff.write_text(
        f"ir_v15 done outcome={status['outcome']} clears={status['clears_gate']} "
        f"top_full={top['full']:.6f} top_nest={top['nested']:.6f} disagree={status['disagree_vs_v7']}\n"
        f"GPU used~{used2:.0f}MB — leave IDLE\n"
        f"keep_ir_v7={status['keep_ir_v7']} wrote_csv={status['wrote_csv']}\n"
        f"metrics=metrics_ir_v15_status.json\n"
        f"next_roi={status.get('next_roi')}\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, default=str), flush=True)
    print(f"handoff -> {handoff}", flush=True)


if __name__ == "__main__":
    main()
