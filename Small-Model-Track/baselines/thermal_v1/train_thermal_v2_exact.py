"""Exact v1 Kinetics FT recipe, multi-seed, then ensemble + MidFuse blend."""
from __future__ import annotations
import argparse, csv, json, random, time, shutil
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


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


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


def train_one_seed(X, y, users, pool_idx, hold_idx, device, seed, epochs=30, batch_size=8, lr=1e-4, patience=10):
    set_seed(seed)
    tag = f"seed{seed}"
    model = build(True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ["stem", "layer1"]):
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(pool_idx), True)
    train_loader = DataLoader(CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed),
                              batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False, seed=seed),
                            batch_size=batch_size * 2, shuffle=False, num_workers=0)
    ckpt_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"exact_seed{seed}.pt"
    best_acc, best_state, best_ep, history = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == 6:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            # rebuild sched for remaining epochs (fix v1 bug)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all", flush=True)
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb); loss = crit(logits, yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, "tr_loss": loss_sum/max(n,1), "tr_acc": correct/max(n,1),
               "val_acc": metrics["acc"], "val_f1": metrics["macro_f1"]}
        history.append(row)
        print(f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
              f"val={row['val_acc']:.4f} f1={row['val_f1']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc,
                        "epoch": ep, "seed": seed, "logits": metrics["logits"], "y": metrics["y"]}, ckpt_path)
            print(f"  saved best={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True); break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    final = evaluate(model, val_loader, device)
    return {"best_acc": best_acc, "logits": final["logits"], "y": final["y"],
            "state": best_state, "ckpt": str(ckpt_path), "model": model}


def softmax_np(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50)); return e / e.sum(1, keepdims=True)


def align_mid(meta_th, mid, meta_v2):
    def key(m):
        uid, lab = int(m["user_id"]), int(m["label"])
        trial = str(m.get("trial") or Path(str(m.get("clip_dir", m.get("pred_dir","")))).name)
        action = str(m.get("action_name", ""))
        return (uid, lab, trial, action)
    idx = {}
    for i, m in enumerate(meta_v2):
        k = key(m); idx[k] = i; idx[(k[0], k[1], k[2])] = i
    out = np.zeros((len(meta_th), mid.shape[1]), np.float32)
    mask = np.zeros(len(meta_th), dtype=bool)
    for i, m in enumerate(meta_th):
        k = key(m); j = idx.get(k) or idx.get((k[0], k[1], k[2]))
        if j is None: continue
        out[i] = mid[j]; mask[i] = True
    return out, mask


def infer(model, cache, device):
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
    model.eval(); logits = np.zeros((len(meta), 40), np.float32)
    with torch.no_grad():
        for i in range(0, len(meta), 8):
            arr = Xt[i:i+8].astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
            x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
            logits[i:i+len(x)] = model(x).float().cpu().numpy()
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
    ap.add_argument("--seeds", default="123,7")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "cache" / "thermal_yolo"
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_logits, all_scores, all_states, all_tags = [], [], [], []

    # include v1
    v1 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt"
    blob = torch.load(v1, map_location="cpu", weights_only=False)
    model = build(False).to(device); model.load_state_dict(blob["model"])
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=16, shuffle=False)
    m = evaluate(model, val_loader, device)
    print(f"[v1] holdout={m['acc']:.4f}", flush=True)
    all_logits.append(m["logits"]); all_scores.append(m["acc"]); all_states.append(blob["model"]); all_tags.append("v1")
    del model; torch.cuda.empty_cache()

    for seed in seeds:
        out = train_one_seed(X, y, users, pool_idx, hold_idx, device, seed, epochs=args.epochs, batch_size=args.batch_size)
        all_logits.append(out["logits"]); all_scores.append(out["best_acc"])
        all_states.append(out["state"]); all_tags.append(f"seed{seed}")
        del out["model"]; torch.cuda.empty_cache()

    ens = np.mean(all_logits, axis=0)
    yt = y[hold_idx]
    ens_acc = float((ens.argmax(1) == yt).mean())
    ens_f1 = float(f1_score(yt, ens.argmax(1), average="macro", zero_division=0))
    print(f"ENSEMBLE={ens_acc:.4f} f1={ens_f1:.4f} members={list(zip(all_tags, all_scores))}", flush=True)

    ckpt_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2"
    np.savez(ckpt_dir / "holdout_ensemble_logits.npz", logits=ens, y=yt,
             seed_scores=np.array(all_scores), tags=np.array(all_tags))

    # midfuse blend
    mid_raw = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_train_logits_honest.npy")
    meta_th = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    meta_v2 = json.loads((TRACK / "baselines" / "skeleton_imu_v2" / "cache" / "train_meta.json").read_text(encoding="utf-8"))
    mid_all, mid_mask = align_mid(meta_th, mid_raw, meta_v2)
    mid_h, nz = mid_all[hold_idx], mid_mask[hold_idx]
    best = (-1.0, None)
    for T in [0.5, 1.0, 1.5, 2.0]:
        for w in np.linspace(0, 1, 21):
            pred = (w * softmax_np(ens[nz], T) + (1 - w) * softmax_np(mid_h[nz], T)).argmax(1)
            acc = float((pred == yt[nz]).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(nz.sum())})
    print(f"BLEND={best[0]:.4f} cfg={best[1]}", flush=True)

    # test infer
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    test_logits_list = []
    # v1
    model = build(False).to(device); model.load_state_dict(all_states[0])
    test_logits_list.append(infer(model, cache, device)); del model; torch.cuda.empty_cache()
    for seed in seeds:
        blob = torch.load(ckpt_dir / f"exact_seed{seed}.pt", map_location="cpu", weights_only=False)
        model = build(False).to(device); model.load_state_dict(blob["model"])
        test_logits_list.append(infer(model, cache, device)); del model; torch.cuda.empty_cache()
        print(f"inferred seed{seed}", flush=True)
    test_logits = np.mean(test_logits_list, axis=0)
    np.save(ckpt_dir / "test_logits.npy", test_logits)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    w, T = best[1]["w"], best[1]["T"]
    fused = (w * softmax_np(test_logits, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    out = ROOT / "submission_thermal_v2.csv"
    out_vid = ROOT / "submission_thermal_v2_video.csv"
    write_sub(out_vid, meta, test_logits.argmax(1), empty, fb)
    nfb = write_sub(out, meta, fused, empty, fb)

    best_i = int(np.argmax(all_scores))
    primary = all_states[best_i]
    fp16 = ckpt_dir / "model_fp16.pt"
    torch.save({"model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in primary.items()},
                "val_acc": all_scores[best_i], "ensemble_acc": ens_acc}, fp16)
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    promote_acc = max(ens_acc, best[0])
    promoted = False
    if promote_acc >= 0.60:
        dest = TRACK / "submission.csv"
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        promoted = True
        print("PROMOTED", dest, promote_acc, flush=True)

    report = {
        "holdout_acc_ensemble": ens_acc,
        "holdout_macro_f1_ensemble": ens_f1,
        "member_scores": dict(zip(all_tags, all_scores)),
        "blend": best[1],
        "holdout_acc_blend": best[0],
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "submission": str(out),
        "submission_video": str(out_vid),
        "promoted": promoted,
        "empty_fallback": nfb,
        "v1_holdout": 0.5754,
        "delta_blend_vs_v1": best[0] - 0.5754,
        "delta_ens_vs_v1": ens_acc - 0.5754,
        "method": "exact_v1_recipe multi-seed + v1 ckpt ensemble + honest MidFuse late-fuse",
    }
    (ROOT / "metrics_thermal_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
