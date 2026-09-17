"""Thermal v3: more seeds + GroupKFold OOF fuse + full-data submit models.

Reuse thermal_yolo cache + existing v1/seed123/seed7 ckpts.
Honest holdout = pool-trained ensemble; fuse weights from pool OOF (not holdout).
Submit logits from all-data multi-seed ensemble (uses users 8/9/24 too).
No TTA (hflip/trev hurt). No Kaggle submit.
"""
from __future__ import annotations
import argparse, csv, json, random, time, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)
CKPT = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3"
CACHE = ROOT / "cache" / "thermal_yolo"


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained: bool = True) -> nn.Module:
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def softmax_np(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {
        "acc": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "logits": np.concatenate(logits_all),
        "y": yt,
    }


def train_one(
    X, y, users, train_idx, val_idx, device, seed, tag, ckpt_path,
    epochs=30, batch_size=8, lr=1e-4, patience=12, unfreeze_ep=5,
):
    set_seed(seed)
    model = build(True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ["stem", "layer1"]):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[train_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[train_idx]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(train_idx), True)
    train_loader = DataLoader(
        CachedClipDataset(X, y, users, train_idx, train=True, seed=seed),
        batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        CachedClipDataset(X, y, users, val_idx, train=False, seed=seed),
        batch_size=batch_size * 2, shuffle=False, num_workers=0,
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_state, best_ep = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all @ ep{ep}", flush=True)
        model.train()
        loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        print(
            f"[{tag}] ep{ep:03d} loss={loss_sum/max(n,1):.4f} tr={correct/max(n,1):.3f} "
            f"val={metrics['acc']:.4f} f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save(
                {
                    "model": best_state,
                    "model_fp16": sd16,
                    "val_acc": best_acc,
                    "epoch": ep,
                    "seed": seed,
                    "tag": tag,
                    "logits": metrics["logits"],
                    "y": metrics["y"],
                },
                ckpt_path,
            )
            print(f"  saved best={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    final = evaluate(model, val_loader, device)
    del model
    torch.cuda.empty_cache()
    return {"best_acc": best_acc, "logits": final["logits"], "y": final["y"], "ckpt": str(ckpt_path)}


@torch.no_grad()
def infer_logits(model, cache: Path, device, batch=8) -> np.ndarray:
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
    model.eval()
    out = np.zeros((len(meta), 40), np.float32)
    for i in range(0, len(meta), batch):
        arr = Xt[i : i + batch].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i : i + len(x)] = model(x).float().cpu().numpy()
    return out


def fuse_grid(th_logits, mid_logits, y, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    best = (-1.0, None)
    yt = y[mask]
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pt = softmax_np(th_logits[mask], T)
        pm = softmax_np(mid_logits[mask], T)
        for w in np.linspace(0.0, 1.0, 41):
            pred = (w * pt + (1.0 - w) * pm).argmax(1)
            acc = float((pred == yt).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(mask.sum())})
    return best


def write_sub(path, meta, preds, empty, fb):
    nfb = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i]))
                nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb


def load_state(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["model"], float(blob.get("val_acc", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-seeds", default="42,99,2024")
    ap.add_argument("--alldata-seeds", default="42,99,2024")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--unfreeze-ep", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-gkf", action="store_true")
    ap.add_argument("--skip-new-seeds", action="store_true")
    ap.add_argument("--skip-alldata", action="store_true")
    args = ap.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(CACHE / "train_y.npy")
    users = np.load(CACHE / "train_users.npy")
    X = np.memmap(CACHE / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device}", flush=True)

    mid = np.load(CACHE / "midfuse_aligned_train_logits.npy")
    mid_mask = mid.any(1)

    # ---- existing + new pool seeds for holdout ensemble ----
    member_logits = []
    member_scores = {}
    member_states = []
    member_tags = []

    # v1
    v1_path = ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt"
    state, _ = load_state(v1_path)
    model = build(False).to(device)
    model.load_state_dict(state)
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=16, shuffle=False)
    m = evaluate(model, val_loader, device)
    print(f"[v1] holdout={m['acc']:.4f}", flush=True)
    member_logits.append(m["logits"])
    member_scores["v1"] = m["acc"]
    member_states.append(state)
    member_tags.append("v1")
    del model
    torch.cuda.empty_cache()

    # exact seeds from v2
    for seed in [123, 7]:
        p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / f"exact_seed{seed}.pt"
        state, vacc = load_state(p)
        model = build(False).to(device)
        model.load_state_dict(state)
        m = evaluate(model, val_loader, device)
        tag = f"seed{seed}"
        print(f"[{tag}] holdout={m['acc']:.4f} (ckpt_val={vacc:.4f})", flush=True)
        member_logits.append(m["logits"])
        member_scores[tag] = m["acc"]
        member_states.append(state)
        member_tags.append(tag)
        del model
        torch.cuda.empty_cache()

    new_seeds = [int(s) for s in args.new_seeds.split(",") if s.strip()]
    if not args.skip_new_seeds:
        for seed in new_seeds:
            tag = f"pool_seed{seed}"
            ck = CKPT / f"{tag}.pt"
            if ck.exists():
                print(f"[{tag}] reuse {ck}", flush=True)
                state, vacc = load_state(ck)
                model = build(False).to(device)
                model.load_state_dict(state)
                m = evaluate(model, val_loader, device)
                member_logits.append(m["logits"])
                member_scores[tag] = m["acc"]
                member_states.append(state)
                member_tags.append(tag)
                del model
                torch.cuda.empty_cache()
            else:
                out = train_one(
                    X, y, users, pool_idx, hold_idx, device, seed, tag, ck,
                    epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, unfreeze_ep=args.unfreeze_ep,
                )
                state, _ = load_state(ck)
                member_logits.append(out["logits"])
                member_scores[tag] = out["best_acc"]
                member_states.append(state)
                member_tags.append(tag)

    ens = np.mean(np.stack(member_logits, 0), 0)
    yt = y[hold_idx]
    ens_acc = float((ens.argmax(1) == yt).mean())
    ens_f1 = float(f1_score(yt, ens.argmax(1), average="macro", zero_division=0))
    print(f"POOL-ENS holdout={ens_acc:.4f} f1={ens_f1:.4f} members={member_scores}", flush=True)
    np.savez(CKPT / "holdout_ensemble_logits.npz", logits=ens, y=yt, tags=np.array(member_tags), scores=np.array([member_scores[t] for t in member_tags]))

    # holdout-tuned fuse (optimistic)
    hold_blend_acc, hold_blend_cfg = fuse_grid(ens, mid[hold_idx], yt, mask=mid_mask[hold_idx])
    print(f"HOLD-TUNE blend={hold_blend_acc:.4f} cfg={hold_blend_cfg}", flush=True)

    # ---- GroupKFold OOF on pool ----
    oof = np.zeros((len(y), 40), np.float32)
    oof_mask = np.zeros(len(y), dtype=bool)
    fold_scores = []
    if not args.skip_gkf:
        gkf = GroupKFold(n_splits=args.folds)
        for fold, (tr, va) in enumerate(gkf.split(pool_idx, y[pool_idx], groups=users[pool_idx])):
            tag = f"gkf_fold{fold}"
            ck = CKPT / f"{tag}.pt"
            tr_idx = pool_idx[tr]
            va_idx = pool_idx[va]
            if ck.exists():
                print(f"[{tag}] reuse {ck}", flush=True)
                state, vacc = load_state(ck)
                model = build(False).to(device)
                model.load_state_dict(state)
                loader = DataLoader(CachedClipDataset(X, y, users, va_idx, train=False), batch_size=16, shuffle=False)
                m = evaluate(model, loader, device)
                oof[va_idx] = m["logits"]
                oof_mask[va_idx] = True
                fold_scores.append(m["acc"])
                print(f"[{tag}] oof_acc={m['acc']:.4f}", flush=True)
                del model
                torch.cuda.empty_cache()
            else:
                out = train_one(
                    X, y, users, tr_idx, va_idx, device, seed=1000 + fold, tag=tag, ckpt_path=ck,
                    epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, unfreeze_ep=args.unfreeze_ep,
                )
                oof[va_idx] = out["logits"]
                oof_mask[va_idx] = True
                fold_scores.append(out["best_acc"])
        np.savez(CKPT / "oof_logits.npz", logits=oof, mask=oof_mask, y=y, users=users, fold_scores=np.array(fold_scores))
        oof_acc = float((oof[oof_mask].argmax(1) == y[oof_mask]).mean())
        print(f"GKF OOF acc={oof_acc:.4f} folds={fold_scores}", flush=True)
        # OOF fuse (honest for public)
        oof_mid_mask = oof_mask & mid_mask
        oof_blend_acc, oof_blend_cfg = fuse_grid(oof, mid, y, mask=oof_mid_mask)
        print(f"OOF-TUNE blend={oof_blend_acc:.4f} cfg={oof_blend_cfg}", flush=True)
        # apply OOF weights on holdout for honest estimate
        w, T = oof_blend_cfg["w"], oof_blend_cfg["T"]
        hz = mid_mask[hold_idx]
        pred = (w * softmax_np(ens[hz], T) + (1 - w) * softmax_np(mid[hold_idx][hz], T)).argmax(1)
        honest_hold_blend = float((pred == yt[hz]).mean())
        print(f"HONEST holdout with OOF weights={honest_hold_blend:.4f} w={w} T={T}", flush=True)
    else:
        oof_acc = None
        oof_blend_cfg = hold_blend_cfg
        honest_hold_blend = hold_blend_acc
        oof_blend_acc = None

    # ---- full-data seeds for submission (max train signal) ----
    alldata_seeds = [int(s) for s in args.alldata_seeds.split(",") if s.strip()]
    test_logit_list = []
    alldata_tags = []
    if not args.skip_alldata:
        # Val = users 22,23 (not competition holdout); train includes holdout users 8/9/24 for max LB signal.
        val_users_all = {22, 23}
        all_val_idx = np.where(np.isin(users, list(val_users_all)))[0]
        all_tr_idx = np.where(~np.isin(users, list(val_users_all)))[0]
        print(f"alldata train={len(all_tr_idx)} val={len(all_val_idx)} val_users={sorted(val_users_all)}", flush=True)
        for seed in alldata_seeds:
            tag = f"all_seed{seed}"
            ck = CKPT / f"{tag}.pt"
            if not ck.exists():
                train_one(
                    X, y, users, all_tr_idx, all_val_idx, device, seed, tag, ck,
                    epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, unfreeze_ep=args.unfreeze_ep,
                )
            state, vacc = load_state(ck)
            model = build(False).to(device)
            model.load_state_dict(state)
            tl = infer_logits(model, CACHE, device)
            test_logit_list.append(tl)
            alldata_tags.append(tag)
            print(f"[{tag}] inferred test ckpt_val={vacc:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
    else:
        # fallback: infer with pool ensemble members
        for tag, state in zip(member_tags, member_states):
            model = build(False).to(device)
            model.load_state_dict(state)
            test_logit_list.append(infer_logits(model, CACHE, device))
            alldata_tags.append(tag)
            del model
            torch.cuda.empty_cache()
            print(f"inferred {tag}", flush=True)

    test_logits = np.mean(np.stack(test_logit_list, 0), 0)
    np.save(CKPT / "test_logits.npy", test_logits)

    # choose fuse cfg: prefer OOF-tuned
    fuse_cfg = oof_blend_cfg if oof_blend_cfg is not None else hold_blend_cfg
    w, T = fuse_cfg["w"], fuse_cfg["T"]
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    fused = (w * softmax_np(test_logits, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)
    video_pred = test_logits.argmax(1)

    meta = json.loads((CACHE / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((CACHE / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    out = ROOT / "submission_thermal_v3.csv"
    out_vid = ROOT / "submission_thermal_v3_video.csv"
    write_sub(out_vid, meta, video_pred, empty, fb)
    nfb = write_sub(out, meta, fused, empty, fb)

    # pack best pool member fp16
    best_i = int(np.argmax([member_scores[t] for t in member_tags]))
    primary = member_states[best_i]
    fp16 = CKPT / "model_fp16.pt"
    torch.save(
        {
            "model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in primary.items()},
            "val_acc": member_scores[member_tags[best_i]],
            "ensemble_acc": ens_acc,
            "tag": member_tags[best_i],
        },
        fp16,
    )
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    report = {
        "holdout_acc_ensemble": ens_acc,
        "holdout_macro_f1_ensemble": ens_f1,
        "member_scores": member_scores,
        "holdout_tune_blend": hold_blend_cfg,
        "holdout_acc_blend_tuned_on_holdout": hold_blend_acc,
        "oof_acc": oof_acc,
        "oof_fold_scores": fold_scores,
        "oof_tune_blend": oof_blend_cfg,
        "oof_blend_acc": oof_blend_acc,
        "honest_holdout_blend_oof_weights": honest_hold_blend,
        "fuse_used_for_submission": fuse_cfg,
        "alldata_tags": alldata_tags,
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "submission": str(out),
        "submission_video": str(out_vid),
        "empty_fallback": nfb,
        "v2_blend_holdout": 0.6639676113360324,
        "delta_honest_vs_v2": (honest_hold_blend - 0.6639676113360324) if honest_hold_blend is not None else None,
        "delta_holdtune_vs_v2": hold_blend_acc - 0.6639676113360324,
        "method": "v3: more pool seeds + GKF OOF fuse + alldata multi-seed submit; MidFuse late-fuse",
        "holdout_users": list(DEFAULT_HOLD_OUT_USERS),
        "notes": [
            "Do not auto-submit; human upload submission_thermal_v3.csv",
            "Submit fuse weights from OOF (not holdout) to reduce public LB gap",
            "All-data seeds used for test logits (train includes holdout users)",
            "No TTA (hflip/trev previously hurt)",
        ],
    }
    (ROOT / "metrics_thermal_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()

