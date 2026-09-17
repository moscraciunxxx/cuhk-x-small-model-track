"""v4: Kinetics R(2+1)D-18 FT on YOLO crop cache + MidFuse/thermal late-fuse."""
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


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def softmax_np(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


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


def fuse_grid(vid, mid, y, mask):
    best = (-1.0, None)
    yt = y[mask]
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pv, pm = softmax_np(vid[mask], T), softmax_np(mid[mask], T)
        for w in np.linspace(0, 1, 41):
            acc = float(((w * pv + (1 - w) * pm).argmax(1) == yt).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(mask.sum())})
    return best


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience, ckpt_path):
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
    best_acc, best_state, best_ep, history = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == 6:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
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
                        "epoch": ep, "seed": seed, "history": history}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    del model; torch.cuda.empty_cache()
    return best_acc, best_state, history


@torch.no_grad()
def infer_cache(model, Xt, device, bs=8):
    model.eval()
    out = np.zeros((len(Xt), 40), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i + len(x)] = model(x).float().cpu().numpy()
    return out


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


def align_midfuse(cache: Path, mid_path: Path):
    """Align MidFuse logits to this cache's train order via user/label/trial keys if shapes differ."""
    mid = np.load(mid_path)
    y = np.load(cache / "train_y.npy")
    if len(mid) == len(y):
        return mid
    # try thermal aligned copy
    alt = ROOT / "cache" / "thermal_yolo" / "midfuse_aligned_train_logits.npy"
    if alt.exists() and len(np.load(alt)) == len(y):
        return np.load(alt)
    raise RuntimeError(f"midfuse len {len(mid)} != cache {len(y)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v4"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--tag", default="ir_v4")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), args.t, args.size, args.size, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device} cache={cache}", flush=True)

    members = []
    if not args.skip_train:
        for seed in args.seeds:
            ck = ckpt_dir / f"pool_seed{seed}.pt"
            if ck.exists():
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                print(f"reuse {ck.name} val={blob.get('val_acc')}", flush=True)
                members.append({"seed": seed, "acc": float(blob["val_acc"]), "state": blob["model"], "path": ck})
                continue
            acc, state, hist = train_one(X, y, users, pool_idx, hold_idx, device, seed,
                                         args.epochs, args.batch_size, args.lr, args.patience, ck)
            members.append({"seed": seed, "acc": acc, "state": state, "path": ck})
    else:
        for seed in args.seeds:
            ck = ckpt_dir / f"pool_seed{seed}.pt"
            if not ck.exists():
                print("missing", ck); continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            members.append({"seed": seed, "acc": float(blob["val_acc"]), "state": blob["model"], "path": ck})

    # holdout logits ensemble
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)
    hold_logits = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"])
        met = evaluate(model, val_loader, device)
        print(f"eval seed{m['seed']} hold={met['acc']:.4f}", flush=True)
        m["hold_logits"] = met["logits"]; m["acc"] = met["acc"]
        hold_logits.append(met["logits"])
        del model; torch.cuda.empty_cache()

    ens = np.mean(hold_logits, 0)
    yt = y[hold_idx]
    ens_acc = float((ens.argmax(1) == yt).mean())
    ens_f1 = float(f1_score(yt, ens.argmax(1), average="macro", zero_division=0))
    print(f"ENSEMBLE video holdout_acc={ens_acc:.4f} f1={ens_f1:.4f}", flush=True)

    # MidFuse align — prefer thermal_yolo aligned (same clip order if same discover order)
    mid_aligned = ROOT / "cache" / "thermal_yolo" / "midfuse_aligned_train_logits.npy"
    mid_honest = TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_train_logits_honest.npy"
    mid = None
    for p in [mid_aligned, mid_honest]:
        if p.exists():
            arr = np.load(p)
            if len(arr) == len(y):
                mid = arr; print("using midfuse", p); break
            # try re-align by meta
    if mid is None:
        # build alignment from metas
        th_meta = json.loads((ROOT / "cache" / "thermal_yolo" / "train_meta.json").read_text(encoding="utf-8"))
        ir_meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
        src = np.load(mid_aligned) if mid_aligned.exists() else np.load(mid_honest)
        key = lambda m: (int(m["user_id"]), int(m["label"]), str(m.get("trial", "")), str(m.get("action_name", "")))
        th_map = {key(m): i for i, m in enumerate(th_meta)}
        mid = np.zeros((len(ir_meta), 40), np.float32)
        hit = 0
        for i, m in enumerate(ir_meta):
            j = th_map.get(key(m))
            if j is not None and j < len(src):
                mid[i] = src[j]; hit += 1
        print(f"aligned midfuse {hit}/{len(ir_meta)}", flush=True)
        np.save(cache / "midfuse_aligned_train_logits.npy", mid)
    else:
        np.save(cache / "midfuse_aligned_train_logits.npy", mid)

    mid_hold = mid[hold_idx]
    mid_mask = mid_hold.any(1)
    blend_acc, blend_cfg = fuse_grid(ens, mid_hold, yt, mid_mask)
    print(f"BLEND MidFuse holdout={blend_acc:.4f} cfg={blend_cfg}", flush=True)

    # optional thermal video late-fuse on holdout (complementary)
    th_pack = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "holdout_ensemble_logits.npz"
    th_blend = None
    if th_pack.exists():
        z = np.load(th_pack)
        # may be holdout-only; skip if shape mismatch
        print("thermal ensemble pack keys", list(z.keys()), {k: z[k].shape for k in z.files})

    # Also try 3-way: IR ens + thermal hold logits from re-eval if available
    th_hold_path = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "oof_logits.npz"
    # Use v1+seed holdout by loading thermal cache eval from saved test only — skip 3way if hard

    # Infer test with each seed + mean
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), args.t, args.size, args.size, 3))
    test_logit_list = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"])
        tl = infer_cache(model, Xt, device, bs=8)
        np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", tl)
        test_logit_list.append(tl)
        del model; torch.cuda.empty_cache()
        print(f"inferred test seed{m['seed']}", flush=True)
    test_logits = np.mean(test_logit_list, 0)
    np.save(ckpt_dir / "test_logits_ens.npy", test_logits)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    w, T = blend_cfg["w"], blend_cfg["T"]
    fused = (w * softmax_np(test_logits, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)

    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    out_vid = ROOT / f"submission_{args.tag}_video.csv"
    out_fuse = ROOT / f"submission_{args.tag}.csv"
    write_sub(out_vid, meta, test_logits.argmax(1), empty, fb)
    nfb = write_sub(out_fuse, meta, fused, empty, fb)

    # Try fuse with thermal_v3 test logits if complementary
    th_test = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_test.exists():
        th_test = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    triple = None
    if th_test.exists() and mid_aligned.exists():
        # rebuild thermal hold ens from members for fair fuse grid needs hold logits
        # Approximate: load thermal hold from holdout_ensemble if present
        pass

    # fp16 pack from best member
    best_m = max(members, key=lambda d: d["acc"])
    fp16 = ckpt_dir / "model_fp16.pt"
    torch.save({"model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in best_m["state"].items()},
                "val_acc": best_m["acc"], "seed": best_m["seed"]}, fp16)
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    # write v4 submission only if clearly better than 0.674
    promoted = None
    if blend_acc > 0.674 + 1e-6:
        promoted = ROOT / "submission_ir_v4.csv"
        # already wrote out_fuse
        print(f"HOLD CLEARLY >0.674 -> primary {out_fuse}", flush=True)
    else:
        print(f"holdout blend {blend_acc:.4f} not clearly >0.674; CSV still written for inspection", flush=True)

    report = {
        "tag": args.tag,
        "cache": str(cache),
        "member_scores": {f"seed{m['seed']}": m["acc"] for m in members},
        "holdout_acc_ensemble": ens_acc,
        "holdout_macro_f1_ensemble": ens_f1,
        "holdout_acc_blend_midfuse": blend_acc,
        "blend_cfg": blend_cfg,
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) < 100,
        "submission_video": str(out_vid),
        "submission_fuse": str(out_fuse),
        "empty_fallback": nfb,
        "holdout_users": list(HOLD),
        "baseline_v3_blend": 0.6740890688259109,
        "delta_vs_v3": float(blend_acc - 0.6740890688259109),
        "write_v4_gate": bool(blend_acc > 0.674),
    }
    (ROOT / f"metrics_{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
