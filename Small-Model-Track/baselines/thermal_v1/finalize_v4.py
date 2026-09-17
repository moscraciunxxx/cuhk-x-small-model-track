"""Finalize v4: IR video ens + optional Thermal ens + MidFuse late-fuse; gate submission_*_v4.csv."""
from __future__ import annotations
import csv, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)
V3_BLEND = 0.6740890688259109


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


def fuse2(a, b, y, mask):
    best = (-1.0, None); yt = y[mask]
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        pa, pb = softmax_np(a[mask], T), softmax_np(b[mask], T)
        for w in np.linspace(0, 1, 41):
            acc = float(((w * pa + (1 - w) * pb).argmax(1) == yt).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(mask.sum())})
    return best


def fuse3(a, b, c, y, mask):
    best = (-1.0, None); yt = y[mask]
    for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, 21):
            for wb in np.linspace(0, 1 - wa, max(1, int((1 - wa) * 20) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum())})
    return best


def write_sub(path, meta, preds, empty, fb):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i]))
            else:
                pred = int(preds[i])
            w.writerow([p, pred])


def align_mid_to_cache(cache: Path):
    y = np.load(cache / "train_y.npy")
    for p in [
        cache / "midfuse_aligned_train_logits.npy",
        ROOT / "cache" / "thermal_yolo" / "midfuse_aligned_train_logits.npy",
        TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_train_logits_honest.npy",
    ]:
        if p.exists() and len(np.load(p)) == len(y):
            return np.load(p), str(p)
    th_meta = json.loads((ROOT / "cache" / "thermal_yolo" / "train_meta.json").read_text(encoding="utf-8"))
    ir_meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    src = np.load(ROOT / "cache" / "thermal_yolo" / "midfuse_aligned_train_logits.npy")
    key = lambda m: (int(m["user_id"]), int(m["label"]), str(m.get("trial", "")), str(m.get("action_name", "")))
    th_map = {key(m): i for i, m in enumerate(th_meta)}
    mid = np.zeros((len(ir_meta), 40), np.float32); hit = 0
    for i, m in enumerate(ir_meta):
        j = th_map.get(key(m))
        if j is not None:
            mid[i] = src[j]; hit += 1
    np.save(cache / "midfuse_aligned_train_logits.npy", mid)
    print(f"aligned mid {hit}/{len(ir_meta)}")
    return mid, "aligned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v4"))
    ap.add_argument("--tag", default="ir_v4")
    args = ap.parse_args()
    cache, ckpt_dir = Path(args.cache_dir), Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)

    members = []
    for p in sorted(ckpt_dir.glob("pool_seed*.pt")):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model = build().to(device); model.load_state_dict(blob["model"])
        logits, yt = eval_logits(model, loader, device)
        acc = float((logits.argmax(1) == yt).mean())
        print(f"{p.stem} hold={acc:.4f}", flush=True)
        members.append({"tag": p.stem, "acc": acc, "logits": logits, "state": blob["model"]})
        del model; torch.cuda.empty_cache()
    ens = np.mean([m["logits"] for m in members], 0)
    ens_acc = float((ens.argmax(1) == yt).mean())
    print(f"IR ens hold={ens_acc:.4f}", flush=True)

    mid, mid_src = align_mid_to_cache(cache)
    mid_h = mid[hold_idx]; mask = mid_h.any(1)
    b2_acc, b2 = fuse2(ens, mid_h, yt, mask)
    print(f"IR+Mid fuse={b2_acc:.4f} {b2}", flush=True)

    th_cache = ROOT / "cache" / "thermal_yolo"
    th_y = np.load(th_cache / "train_y.npy"); th_u = np.load(th_cache / "train_users.npy")
    th_hold = np.where(np.isin(th_u, list(HOLD)))[0]
    ir_meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    th_meta = json.loads((th_cache / "train_meta.json").read_text(encoding="utf-8"))
    key = lambda m: (int(m["user_id"]), int(m["label"]), str(m.get("trial", "")), str(m.get("action_name", "")))
    th_paths = [
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt",
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "exact_seed123.pt",
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "exact_seed7.pt",
    ]
    th_X = np.memmap(th_cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(th_y), 16, 112, 112, 3))
    th_loader = DataLoader(CachedClipDataset(th_X, th_y, th_u, th_hold, train=False), batch_size=12, shuffle=False)
    th_logs = []
    for p in th_paths:
        if not p.exists():
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model = build().to(device); model.load_state_dict(blob["model"])
        logits, _ = eval_logits(model, th_loader, device)
        th_logs.append(logits); del model; torch.cuda.empty_cache()
        print("loaded thermal", p.name, flush=True)
    th_ens = np.mean(th_logs, 0)
    th_hold_meta = [th_meta[i] for i in th_hold]
    th_map = {key(m): j for j, m in enumerate(th_hold_meta)}
    th_on_ir = np.zeros_like(ens); mapped = 0
    for j, ii in enumerate(hold_idx):
        k = key(ir_meta[ii])
        if k in th_map:
            th_on_ir[j] = th_ens[th_map[k]]; mapped += 1
    print(f"mapped thermal hold onto IR hold {mapped}/{len(hold_idx)}", flush=True)
    th_mask = th_on_ir.any(1) & mask
    b3_acc, b3 = fuse3(ens, th_on_ir, mid_h, yt, th_mask)
    print(f"IR+Thermal+Mid fuse={b3_acc:.4f} {b3}", flush=True)
    bt_acc, bt = fuse2(th_on_ir, mid_h, yt, th_mask)
    print(f"Thermal+Mid (recheck)={bt_acc:.4f} {bt}", flush=True)
    bi_acc, bi = fuse2(ens, th_on_ir, yt, th_on_ir.any(1))
    print(f"IR+Thermal video={bi_acc:.4f} {bi}", flush=True)

    cands = [
        ("ir_mid", b2_acc, b2, "2way_ir_mid"),
        ("triple", b3_acc, b3, "3way"),
        ("th_mid", bt_acc, bt, "2way_th_mid"),
        ("ir_th", bi_acc, bi, "2way_ir_th"),
        ("ir_only", ens_acc, {"w": 1.0, "T": 1.0, "acc": ens_acc}, "ir_only"),
    ]
    best_name, best_acc, best_cfg, best_kind = max(cands, key=lambda x: x[1])
    print("BEST", best_name, best_acc, best_cfg, flush=True)

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    ir_test = np.load(ckpt_dir / "test_logits_ens.npy") if (ckpt_dir / "test_logits_ens.npy").exists() else None
    if ir_test is None:
        Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
        outs = []
        for m in members:
            model = build().to(device); model.load_state_dict(m["state"]); model.eval()
            o = np.zeros((len(meta), 40), np.float32)
            with torch.no_grad():
                for i in range(0, len(meta), 8):
                    arr = Xt[i:i+8].astype(np.float32) / 255.0
                    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
                    x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
                    o[i:i+len(x)] = model(x).float().cpu().numpy()
            outs.append(o); del model; torch.cuda.empty_cache()
        ir_test = np.mean(outs, 0); np.save(ckpt_dir / "test_logits_ens.npy", ir_test)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    if not th_test_p.exists():
        th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    th_test = np.load(th_test_p) if th_test_p.exists() else None

    if best_kind == "3way" and th_test is not None:
        wa, wb, wc, T = best_cfg["wa"], best_cfg["wb"], best_cfg["wc"], best_cfg["T"]
        pred = (wa * softmax_np(ir_test, T) + wb * softmax_np(th_test, T) + wc * softmax_np(mid_test, T)).argmax(1)
    elif best_kind == "2way_ir_mid":
        w, T = best_cfg["w"], best_cfg["T"]
        pred = (w * softmax_np(ir_test, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)
    elif best_kind == "2way_th_mid" and th_test is not None:
        w, T = best_cfg["w"], best_cfg["T"]
        pred = (w * softmax_np(th_test, T) + (1 - w) * softmax_np(mid_test, T)).argmax(1)
    elif best_kind == "2way_ir_th" and th_test is not None:
        w, T = best_cfg["w"], best_cfg["T"]
        pred = (w * softmax_np(ir_test, T) + (1 - w) * softmax_np(th_test, T)).argmax(1)
    else:
        pred = ir_test.argmax(1)

    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    out_inspect = ROOT / f"submission_{args.tag}.csv"
    write_sub(out_inspect, meta, pred, empty, fb)
    write_sub(ROOT / f"submission_{args.tag}_video.csv", meta, ir_test.argmax(1), empty, fb)

    out_v4 = None
    if best_acc > V3_BLEND + 1e-6:
        out_v4 = ROOT / "submission_ir_thermal_mid_v4.csv"
        write_sub(out_v4, meta, pred, empty, fb)
        print("WROTE gated v4", out_v4, "hold", best_acc, flush=True)
    else:
        print("NOT gated (need >0.674); inspect CSV only", out_inspect, flush=True)

    report = {
        "ir_ens_hold": ens_acc,
        "members": {m["tag"]: m["acc"] for m in members},
        "ir_mid": b2, "triple": b3, "th_mid": bt, "ir_th": bi,
        "best_name": best_name, "best_acc": best_acc, "best_cfg": best_cfg,
        "v3_blend": V3_BLEND, "delta": float(best_acc - V3_BLEND),
        "submission_inspect": str(out_inspect),
        "submission_v4": str(out_v4) if out_v4 else None,
        "mid_src": mid_src,
    }
    (ROOT / f"metrics_{args.tag}_finalize.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
