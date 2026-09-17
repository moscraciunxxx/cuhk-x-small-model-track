"""Finalize thermal v3: top-k / weighted pool ensemble + OOF fuse + alldata test logits."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from sklearn.metrics import f1_score
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
HOLD = set(DEFAULT_HOLD_OUT_USERS)
CKPT = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3"
CACHE = ROOT / "cache" / "thermal_yolo"
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

def build():
    m = r2plus1d_18(weights=None); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m

def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)

def softmax_np(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50)); return e / e.sum(1, keepdims=True)

@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval(); outs, ys = [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        outs.append(model(x).float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)

def fuse_grid(th, mid, y, mask):
    best = (-1.0, None); yt = y[mask]
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pt, pm = softmax_np(th[mask], T), softmax_np(mid[mask], T)
        for w in np.linspace(0, 1, 41):
            acc = float(((w * pt + (1 - w) * pm).argmax(1) == yt).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(mask.sum())})
    return best

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

def infer_test(model, device):
    meta = json.loads((CACHE / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(CACHE / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
    model.eval(); out = np.zeros((len(meta), 40), np.float32)
    with torch.no_grad():
        for i in range(0, len(meta), 8):
            arr = Xt[i:i+8].astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
            x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
            out[i:i+len(x)] = model(x).float().cpu().numpy()
    return out, meta

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(CACHE / "train_y.npy"); users = np.load(CACHE / "train_users.npy")
    X = np.memmap(CACHE / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    mid = np.load(CACHE / "midfuse_aligned_train_logits.npy"); mid_mask = mid.any(1)
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=16, shuffle=False)

    paths = [
        ("v1", ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt"),
        ("seed123", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "exact_seed123.pt"),
        ("seed7", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "exact_seed7.pt"),
        ("pool_seed42", CKPT / "pool_seed42.pt"),
        ("pool_seed99", CKPT / "pool_seed99.pt"),
        ("pool_seed2024", CKPT / "pool_seed2024.pt"),
    ]
    members = []
    for tag, p in paths:
        if not p.exists():
            print("skip missing", tag); continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model = build().to(device); model.load_state_dict(blob["model"])
        logits, yt = eval_logits(model, loader, device)
        acc = float((logits.argmax(1) == yt).mean())
        print(f"{tag} holdout={acc:.4f}", flush=True)
        members.append({"tag": tag, "acc": acc, "logits": logits, "state": blob["model"]})
        del model; torch.cuda.empty_cache()

    members = sorted(members, key=lambda d: -d["acc"])
    yt = y[hold_idx]
    results = {}
    for k in [3, 4, 5, 6]:
        if k > len(members): continue
        top = members[:k]
        ens = np.mean([m["logits"] for m in top], 0)
        wts = np.array([m["acc"] for m in top], dtype=np.float64); wts /= wts.sum()
        ens_w = sum(wi * m["logits"] for wi, m in zip(wts, top))
        for name, E in [("uniform", ens), ("weighted", ens_w)]:
            acc = float((E.argmax(1) == yt).mean())
            f1 = float(f1_score(yt, E.argmax(1), average="macro", zero_division=0))
            blend_acc, blend_cfg = fuse_grid(E, mid[hold_idx], yt, mid_mask[hold_idx])
            key = f"top{k}_{name}"
            results[key] = {"ens_acc": acc, "ens_f1": f1, "blend": blend_cfg,
                            "tags": [m["tag"] for m in top],
                            "weights": wts.tolist() if name == "weighted" else None}
            print(f"{key}: ens={acc:.4f} blend={blend_acc:.4f} cfg={blend_cfg}", flush=True)

    best_key = max(results, key=lambda k: results[k]["blend"]["acc"])
    print("BEST_LOCAL", best_key, results[best_key], flush=True)

    oof_path = CKPT / "oof_logits.npz"
    oof_cfg = None; honest = None; oof_acc = None
    tag_order = results[best_key]["tags"]
    top = [next(m for m in members if m["tag"] == t) for t in tag_order]
    if results[best_key]["weights"] is not None:
        wts = np.array(results[best_key]["weights"]); E = sum(wi * m["logits"] for wi, m in zip(wts, top))
    else:
        E = np.mean([m["logits"] for m in top], 0)

    if oof_path.exists():
        z = np.load(oof_path)
        oof, oof_mask = z["logits"], z["mask"].astype(bool)
        oof_acc = float((oof[oof_mask].argmax(1) == y[oof_mask]).mean())
        _, oof_cfg = fuse_grid(oof, mid, y, oof_mask & mid_mask)
        print(f"OOF acc={oof_acc:.4f} fuse={oof_cfg}", flush=True)
        w, T = oof_cfg["w"], oof_cfg["T"]
        hz = mid_mask[hold_idx]
        pred = (w * softmax_np(E[hz], T) + (1 - w) * softmax_np(mid[hold_idx][hz], T)).argmax(1)
        honest = float((pred == yt[hz]).mean())
        print(f"HONEST holdout OOF-weights={honest:.4f}", flush=True)
        fuse_cfg = oof_cfg
    else:
        fuse_cfg = results[best_key]["blend"]
        honest = results[best_key]["blend"]["acc"]
        print("No OOF yet; using holdout-tuned fuse", flush=True)

    test_paths = sorted(CKPT.glob("all_seed*.pt"))
    test_logit_list, tags_test = [], []
    if test_paths:
        for p in test_paths:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            model = build().to(device); model.load_state_dict(blob["model"])
            out, meta = infer_test(model, device)
            test_logit_list.append(out); tags_test.append(p.stem)
            del model; torch.cuda.empty_cache(); print("inferred", p.stem, flush=True)
    else:
        for t in tag_order:
            m = next(mm for mm in members if mm["tag"] == t)
            model = build().to(device); model.load_state_dict(m["state"])
            out, meta = infer_test(model, device)
            test_logit_list.append(out); tags_test.append(t)
            del model; torch.cuda.empty_cache(); print("inferred pool", t, flush=True)

    test_logits = np.mean(test_logit_list, 0)
    np.save(CKPT / "test_logits_final.npy", test_logits)
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    w, T = fuse_cfg["w"], fuse_cfg["T"]
    fused = (w * softmax_np(test_logits, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)
    empty = set(json.loads((CACHE / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    out_csv = ROOT / "submission_thermal_v3.csv"
    out_vid = ROOT / "submission_thermal_v3_video.csv"
    write_sub(out_vid, meta, test_logits.argmax(1), empty, fb)
    nfb = write_sub(out_csv, meta, fused, empty, fb)

    best_m = members[0]
    fp16 = CKPT / "model_fp16.pt"
    torch.save({"model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in best_m["state"].items()},
                "val_acc": best_m["acc"], "tag": best_m["tag"]}, fp16)
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    report = {
        "best_local_key": best_key,
        "ensemble_search": results,
        "oof_acc": oof_acc,
        "oof_fuse": oof_cfg,
        "honest_holdout_blend": honest,
        "holdout_tune_best_blend": results[best_key]["blend"],
        "fuse_used": fuse_cfg,
        "test_members": tags_test,
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "submission": str(out_csv),
        "submission_video": str(out_vid),
        "empty_fallback": nfb,
        "v2_blend_holdout": 0.6639676113360324,
        "delta_honest_vs_v2": None if honest is None else honest - 0.6639676113360324,
        "method": "v3 finalize: top-k/weighted pool ens + OOF fuse if avail + alldata/pool test",
        "notes": ["Do not auto-submit", "Human upload submission_thermal_v3.csv"],
    }
    (ROOT / "metrics_thermal_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "ensemble_search"}, indent=2), flush=True)
    print("WROTE", out_csv, flush=True)

if __name__ == "__main__":
    main()
