"""Thermal v2b: v1-like Kinetics FT + multi-seed + honest MidFuse late-fuse."""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score

from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained=True):
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def to_ncthw(x):
    return ((x - K_MEAN.to(x.device)) / K_STD.to(x.device)).permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def eval_logits(model, loader, device, tta=False):
    model.eval()
    ys, outs = [], []
    for x, y, _u, _i in loader:
        x = x.to(device, non_blocking=True)
        logits = model(to_ncthw(x))
        if tta:
            logits = 0.5 * (logits + model(to_ncthw(x.flip(-1))))
        outs.append(logits.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def softmax_np(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


def train_seed(X, y, users, pool_idx, hold_idx, device, args, seed, ckpt_path):
    set_seed(seed)
    tag = f"seed{seed}"
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True)
    train_loader = DataLoader(
        CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed),
        batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        CachedClipDataset(X, y, users, hold_idx, train=False, seed=seed),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
    )
    model = build(pretrained=True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ("stem", "layer1")):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    # Cosine over full run; on unfreeze rebuild opt with lower lr and fresh cosine remainder
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    best_acc, best_state, best_ep, history = -1.0, None, 0, []
    unfrozen = False
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if (not unfrozen) and ep == args.unfreeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            # differential LR: backbone lower
            bb, hd = [], []
            for n, p in model.named_parameters():
                (hd if "fc" in n else bb).append(p)
            opt = torch.optim.AdamW(
                [{"params": bb, "lr": args.lr * 0.2}, {"params": hd, "lr": args.lr * 0.5}],
                weight_decay=1e-4,
            )
            unfrozen = True
            print(f"[{tag}] unfroze ALL @ ep{ep}", flush=True)

        # cosine decay from current param-group base
        progress = (ep - 1) / max(args.epochs - 1, 1)
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * max(cos, 0.1)

        model.train()
        loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = to_ncthw(xb.to(device))
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)

        logits_v, yt = eval_logits(model, val_loader, device, tta=False)
        acc = float((logits_v.argmax(1) == yt).mean())
        f1 = float(f1_score(yt, logits_v.argmax(1), average="macro", zero_division=0))
        # also tta score for logging (select on no-tta like v1 for stability)
        logits_t, _ = eval_logits(model, val_loader, device, tta=True)
        acc_t = float((logits_t.argmax(1) == yt).mean())
        row = {"epoch": ep, "tr_loss": loss_sum / max(n, 1), "tr_acc": correct / max(n, 1),
               "val_acc": acc, "val_f1": f1, "val_acc_tta": acc_t}
        history.append(row)
        print(f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
              f"val={acc:.4f} tta={acc_t:.4f} f1={f1:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc,
                        "val_acc_tta": acc_t, "epoch": ep, "seed": seed, "args": vars(args)}, ckpt_path)
            print(f"  saved best acc={best_acc:.4f}", flush=True)
        if args.patience > 0 and ep - best_ep >= args.patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break

    model.load_state_dict(best_state)
    logits_v, yt = eval_logits(model, val_loader, device, tta=False)
    logits_t, _ = eval_logits(model, val_loader, device, tta=True)
    print(f"[{tag}] BEST={best_acc:.4f} final_tta={float((logits_t.argmax(1)==yt).mean()):.4f} "
          f"took={time.time()-t0:.1f}s", flush=True)
    return {"best_acc": best_acc, "logits": logits_v, "logits_tta": logits_t, "y": yt,
            "state": best_state, "history": history, "model": model}


def align_midfuse_train(meta_th, mid_logits, meta_v2):
    def make_key(m):
        uid = int(m["user_id"]); lab = int(m["label"])
        trial = str(m.get("trial") or Path(str(m.get("clip_dir", m.get("pred_dir", "")))).name)
        action = str(m.get("action_name", ""))
        return (uid, lab, trial, action)

    idx = {}
    for i, m in enumerate(meta_v2):
        k = make_key(m)
        idx[k] = i
        idx[(k[0], k[1], k[2])] = i
    out = np.zeros((len(meta_th), mid_logits.shape[1]), np.float32)
    mask = np.zeros(len(meta_th), dtype=bool)
    for i, m in enumerate(meta_th):
        k = make_key(m)
        j = idx.get(k) or idx.get((k[0], k[1], k[2]))
        if j is None:
            continue
        out[i] = mid_logits[j]; mask[i] = True
    return out, mask


def infer_test(model, cache, device, tta=False):
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), 16, 112, 112, 3))
    model.eval()
    logits = np.zeros((len(meta), 40), np.float32)
    with torch.no_grad():
        for i in range(0, len(meta), 8):
            arr = Xt[i:i+8].astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
            out = model(to_ncthw(x))
            if tta:
                out = 0.5 * (out + model(to_ncthw(x.flip(-1))))
            logits[i:i+len(x)] = out.float().cpu().numpy()
    return logits


def write_sub(path, meta, preds, empty, fb):
    nfb = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i])); nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--unfreeze-ep", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--include-v1", action="store_true", help="include existing v1 ckpt in ensemble")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "cache" / "thermal_yolo"
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device}", flush=True)

    ckpt_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    seed_logits, seed_scores, seed_states = [], [], []
    for seed in seeds:
        out = train_seed(X, y, users, pool_idx, hold_idx, device, args, seed,
                         ckpt_dir / f"holdout_seed{seed}.pt")
        seed_logits.append(out["logits"])
        seed_scores.append(out["best_acc"])
        seed_states.append(out["state"])
        del out["model"]; torch.cuda.empty_cache()

    # optionally include v1
    if args.include_v1:
        v1 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt"
        if v1.exists():
            blob = torch.load(v1, map_location="cpu", weights_only=False)
            model = build(pretrained=False).to(device)
            model.load_state_dict(blob["model"])
            val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False),
                                    batch_size=16, shuffle=False)
            logits_v, yt = eval_logits(model, val_loader, device, tta=False)
            acc = float((logits_v.argmax(1) == yt).mean())
            print(f"[v1] holdout={acc:.4f}", flush=True)
            seed_logits.append(logits_v); seed_scores.append(acc); seed_states.append(blob["model"])
            del model; torch.cuda.empty_cache()

    ens = np.mean(seed_logits, axis=0)
    yt = y[hold_idx]
    ens_acc = float((ens.argmax(1) == yt).mean())
    ens_f1 = float(f1_score(yt, ens.argmax(1), average="macro", zero_division=0))
    print(f"ENSEMBLE holdout={ens_acc:.4f} f1={ens_f1:.4f} seeds={seed_scores}", flush=True)
    np.savez(ckpt_dir / "holdout_ensemble_logits.npz", logits=ens,
             seed_logits=np.stack(seed_logits), y=yt, seed_scores=np.array(seed_scores))

    best_i = int(np.argmax(seed_scores))
    primary = seed_states[best_i]
    torch.save({"model": primary,
                "model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in primary.items()},
                "val_acc": seed_scores[best_i], "ensemble_acc": ens_acc,
                "seed_scores": seed_scores}, ckpt_dir / "holdout_best.pt")

    # MidFuse honest blend
    mid_raw = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_train_logits_honest.npy")
    meta_th = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    meta_v2 = json.loads((TRACK / "baselines" / "skeleton_imu_v2" / "cache" / "train_meta.json").read_text(encoding="utf-8"))
    mid_all, mid_mask = align_midfuse_train(meta_th, mid_raw, meta_v2)
    mid_h = mid_all[hold_idx]; nz = mid_mask[hold_idx]
    best = (-1.0, None)
    for T in [0.5, 1.0, 1.5, 2.0]:
        for w in np.linspace(0, 1, 21):
            pred = (w * softmax_np(ens[nz], T) + (1 - w) * softmax_np(mid_h[nz], T)).argmax(1)
            acc = float((pred == yt[nz]).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(nz.sum())})
    print(f"BLEND holdout={best[0]:.4f} cfg={best[1]} mid_alone={float((mid_h[nz].argmax(1)==yt[nz]).mean()):.4f}", flush=True)

    # test infer all seeds (+ optional v1)
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    test_seeds = []
    for seed in seeds:
        blob = torch.load(ckpt_dir / f"holdout_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = build(pretrained=False).to(device); model.load_state_dict(blob["model"])
        test_seeds.append(infer_test(model, cache, device, tta=False))
        del model; torch.cuda.empty_cache()
        print(f"inferred seed{seed}", flush=True)
    if args.include_v1:
        blob = torch.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt",
                          map_location="cpu", weights_only=False)
        model = build(pretrained=False).to(device); model.load_state_dict(blob["model"])
        test_seeds.append(infer_test(model, cache, device, tta=False))
        del model; torch.cuda.empty_cache()
    test_logits = np.mean(test_seeds, axis=0)
    np.save(ckpt_dir / "test_logits.npy", test_logits)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    w, T = best[1]["w"], best[1]["T"]
    fused = (w * softmax_np(test_logits, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)

    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    out_video = ROOT / "submission_thermal_v2_video.csv"
    out_blend = ROOT / "submission_thermal_v2.csv"
    write_sub(out_video, meta, test_logits.argmax(1), empty, fb)
    nfb = write_sub(out_blend, meta, fused, empty, fb)
    print(f"wrote {out_blend} nfb={nfb}", flush=True)

    fp16 = ckpt_dir / "model_fp16.pt"
    torch.save({"model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in primary.items()},
                "val_acc": seed_scores[best_i], "ensemble_acc": ens_acc}, fp16)
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    promote_acc = max(ens_acc, best[0])
    promoted = False
    if promote_acc >= 0.60:
        dest = TRACK / "submission.csv"
        dest.write_text(out_blend.read_text(encoding="utf-8"), encoding="utf-8")
        promoted = True
        print("PROMOTED", dest, promote_acc, flush=True)

    report = {
        "holdout_acc_ensemble": ens_acc,
        "holdout_macro_f1_ensemble": ens_f1,
        "seed_scores": seed_scores,
        "blend": best[1],
        "holdout_acc_blend": best[0],
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "submission": str(out_blend),
        "submission_video": str(out_video),
        "promoted": promoted,
        "v1_holdout": 0.5754,
        "delta_blend_vs_v1": best[0] - 0.5754,
        "delta_ens_vs_v1": ens_acc - 0.5754,
    }
    (ROOT / "metrics_thermal_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ckpt_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
