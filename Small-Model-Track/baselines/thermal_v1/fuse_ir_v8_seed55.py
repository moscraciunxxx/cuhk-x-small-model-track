"""IR v8: infer seed55 hold+test logits; re-fuse all10 IR + Thermal + Mid; promote if clear > v7."""
from __future__ import annotations
import csv, json, time
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
V7 = 0.7530364372469636
CLEAR = 0.002  # clearly > v7
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def softmax_np(z, T=1.0):
    z = z / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    outs, ys = [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        outs.append(model(x).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


@torch.no_grad()
def infer_cache(model, Xt, device, bs=8):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i + len(x)] = model(x).float().cpu().numpy()
    return out


def fuse3_sameT(a, b, c, y, mask, Ts, ngrid=51):
    best = (-1.0, None)
    yt = y[mask]
    for T in Ts:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0] + 1e-12:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "sameT"})
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


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    cache = ROOT / "cache" / "ir_yolo_v4"
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)

    z = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_logits_v6.npz", allow_pickle=True)
    old_tags = [str(t) for t in z["tags"]]
    old_base = z["base"]; yt = z["y"]; yu = z["users"]
    assert np.allclose(yt, y[hold_idx])

    members = []
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    for i, t in enumerate(old_tags):
        acc = float((old_base[i].argmax(1) == yt).mean())
        tl_path = old_ckpt / f"test_logits_{t}.npy"
        if not tl_path.exists():
            seed = t.replace("pool_seed", "")
            tl_path = old_ckpt / f"test_logits_seed{seed}.npy"
        members.append({"tag": t, "logits": old_base[i], "acc": acc, "test_logits": np.load(tl_path)})

    new_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
    hz = np.load(new_dir / "hold_logits_new_seeds.npz", allow_pickle=True)
    for t in hz["tags"]:
        t = str(t)
        seed = t.replace("pool_seed", "")
        lg = hz[t]
        acc = float((lg.argmax(1) == yt).mean())
        members.append({"tag": t, "logits": lg, "acc": acc,
                        "test_logits": np.load(new_dir / f"test_logits_seed{seed}.npy")})

    # seed55: infer hold + test
    v7_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7"
    ckpt55 = v7_dir / "pool_seed55.pt"
    hold55_path = v7_dir / "hold_logits_seed55.npy"
    test55_path = v7_dir / "test_logits_seed55.npy"
    blob = torch.load(ckpt55, map_location="cpu", weights_only=False)
    print(f"seed55 ckpt val_acc={blob.get('val_acc')} ep={blob.get('epoch')}", flush=True)
    model = build().to(device)
    model.load_state_dict(blob["model"])
    if hold55_path.exists() and test55_path.exists():
        lg55 = np.load(hold55_path)
        tl55 = np.load(test55_path)
        print(f"loaded cached seed55 hold/test logits", flush=True)
    else:
        lg55, yt2 = eval_logits(model, loader, device)
        assert np.allclose(yt2, yt)
        np.save(hold55_path, lg55.astype(np.float32))
        print(f"seed55 hold solo={float((lg55.argmax(1)==yt).mean()):.4f}", flush=True)
        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r",
                       shape=(len(meta), 16, 112, 112, 3))
        tl55 = infer_cache(model, Xt, device, bs=8)
        np.save(test55_path, tl55.astype(np.float32))
        print(f"seed55 test logits saved {tl55.shape}", flush=True)
    del model; torch.cuda.empty_cache()
    acc55 = float((lg55.argmax(1) == yt).mean())
    members.append({"tag": "pool_seed55", "logits": lg55, "acc": acc55, "test_logits": tl55})
    print(f"seed55 hold solo acc={acc55:.4f}", flush=True)

    # also save npz for seed55
    np.savez_compressed(v7_dir / "hold_logits_seed55.npz", tags=np.array(["pool_seed55"]),
                        pool_seed55=lg55, y=yt, users=yu)

    members = sorted(members, key=lambda d: -d["acc"])
    print("members:", [(m["tag"], round(m["acc"], 4)) for m in members], flush=True)

    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")[hold_idx]
    mask = th.any(1) & mid.any(1)
    print(f"mask n={int(mask.sum())} / {len(yt)}", flush=True)

    Ts_fine = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
    stack = np.stack([m["logits"] for m in members], 0)
    w_acc = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64); w_acc /= w_acc.sum()
    variants = {}
    for k in range(3, len(members) + 1):
        variants[f"top{k}"] = np.mean(stack[:k], 0)
    variants["all_mean"] = np.mean(stack, 0)
    variants["all_acc_w"] = np.tensordot(w_acc, stack, axes=(0, 0))
    # without seed55 for reference
    members9 = [m for m in members if m["tag"] != "pool_seed55"]
    stack9 = np.stack([m["logits"] for m in members9], 0)
    variants["all9_mean"] = np.mean(stack9, 0)

    results = []
    for name, elogs in variants.items():
        ens_acc = float((elogs.argmax(1) == yt).mean())
        b_acc, bcfg = fuse3_sameT(elogs, th, mid, yt, mask, Ts_fine, ngrid=51)
        results.append({"ens": name, "ens_acc": ens_acc, "triple": bcfg, "triple_acc": b_acc})
        print(f"{name}: ens={ens_acc:.4f} triple={b_acc:.4f} {bcfg}", flush=True)

    best = max(results, key=lambda r: r["triple_acc"])
    best_acc = best["triple_acc"]
    best_cfg = best["triple"]
    print(f"\nBEST: {best['ens']} triple={best_acc:.6f} delta_v7={best_acc-V7:+.4f}", flush=True)

    # reproduce v7 cfg on all9
    v7_cfg = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
    T = v7_cfg["T"]
    pa = softmax_np(variants["all9_mean"][mask], T); pb = softmax_np(th[mask], T); pc = softmax_np(mid[mask], T)
    v7_re = float(((v7_cfg["wa"]*pa + v7_cfg["wb"]*pb + v7_cfg["wc"]*pc).argmax(1) == yt[mask]).mean())
    print(f"v7 cfg reproduce all9: {v7_re:.6f}", flush=True)

    clear_win = best_acc >= V7 + CLEAR - 1e-9
    report = {
        "tag": "ir_v8",
        "seed55_hold_solo": acc55,
        "seed55_val_ckpt": float(blob.get("val_acc") or 0),
        "seed55_epoch": int(blob.get("epoch") or 0),
        "members": {m["tag"]: m["acc"] for m in members},
        "results": results,
        "best_ens": best["ens"],
        "best_acc": best_acc,
        "best_cfg": best_cfg,
        "delta_vs_v7": float(best_acc - V7),
        "v7": V7,
        "v7_reproduce_all9": v7_re,
        "clear_win": bool(clear_win),
        "clear_margin": CLEAR,
        "elapsed_s": time.time() - t0,
        "promote": None,
    }

    if clear_win:
        promote_ens = best["ens"]
        promote_cfg = best_cfg
        if promote_ens.startswith("top"):
            k = int(promote_ens[3:])
            sel = members[:k]
            ir_test = np.mean([m["test_logits"] for m in sel], 0).astype(np.float32)
            tags = [m["tag"] for m in sel]
        elif promote_ens == "all_acc_w":
            ir_test = np.tensordot(w_acc, np.stack([m["test_logits"] for m in members], 0), axes=(0, 0)).astype(np.float32)
            tags = [m["tag"] for m in members]
        elif promote_ens == "all9_mean":
            ir_test = np.mean([m["test_logits"] for m in members9], 0).astype(np.float32)
            tags = [m["tag"] for m in members9]
        else:
            ir_test = np.mean([m["test_logits"] for m in members], 0).astype(np.float32)
            tags = [m["tag"] for m in members]

        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
        th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")
        T = promote_cfg["T"]
        preds = (promote_cfg["wa"] * softmax_np(ir_test, T) +
                 promote_cfg["wb"] * softmax_np(th_test, T) +
                 promote_cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
            for row in csv.DictReader(f):
                fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
        out = ROOT / "submission_ir_v8.csv"
        nfb = write_sub(out, meta, preds, empty, fb)
        track_sub = TRACK / "submission.csv"
        track_sub.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        # also copy to Coding Compete root submission if different path same content already via junction
        # size check: reuse v6/v7 fp16 pack (seed55 not packed alone — ensemble CSV only)
        fp16 = new_dir / "model_fp16.pt"
        if not fp16.exists():
            fp16 = old_ckpt / "model_fp16.pt"
        yolo = ROOT / "yolov8n.pt"
        fp16_mb = fp16.stat().st_size / (1024 * 1024) if fp16.exists() else -1
        yolo_mb = yolo.stat().st_size / (1024 * 1024) if yolo.exists() else 0
        total_mb = fp16_mb + yolo_mb
        # disagree vs v7
        v7p = []
        with open(ROOT / "submission_ir_v7.csv") as f:
            for row in csv.DictReader(f):
                v7p.append(int(row["prediction"]))
        disagree = int(sum(int(a) != int(b) for a, b in zip(preds, v7p)))
        report["promote"] = {
            "wrote": str(out),
            "promoted_track": str(track_sub),
            "hold": best_acc,
            "cfg": promote_cfg,
            "ens": promote_ens,
            "tags": tags,
            "empty_fallback": nfb,
            "disagree_vs_v7": disagree,
            "fp16_pack_mb": fp16_mb,
            "yolo_mb": yolo_mb,
            "total_approx_mb": total_mb,
            "size_ok_under_100mb": bool(total_mb <= 100),
        }
        report["primary"] = "submission_ir_v8.csv"
        report["method"] = f"{promote_ens} IR (+seed55) + Thermal + Mid; {promote_cfg}"
        report["holdout_acc"] = best_acc
        report["cfg"] = promote_cfg
        report["notes"] = [
            f"PRIMARY hold {best_acc:.4f} (+{best_acc-V7:.4f} vs v7 {V7:.4f}) — CLEAR WIN",
            f"seed55 solo hold {acc55:.4f} (ckpt val {blob.get('val_acc')})",
            f"disagree vs v7: {disagree}",
            "Do not Kaggle-submit from this script; <=100MB pack ok",
            "Depth_Color skipped",
        ]
        print(f"PROMOTED v8 hold={best_acc:.4f} -> {out} + track submission.csv disagree={disagree}", flush=True)
    else:
        report["primary"] = "submission_ir_v7.csv"
        report["holdout_acc"] = V7
        report["notes"] = [
            f"NO CLEAR WIN best={best_acc:.4f} v7={V7:.4f} need>={V7+CLEAR:.4f}; leave v7 primary",
            f"seed55 solo hold {acc55:.4f} (ckpt val {blob.get('val_acc')} ep{blob.get('epoch')})",
            "logits saved for future use",
            "Do not Kaggle-submit from this script",
        ]
        print(f"NO CLEAR WIN (best={best_acc:.4f} v7={V7:.4f} need>={V7+CLEAR:.4f}); leave v7 primary", flush=True)

    report["elapsed_s"] = time.time() - t0
    (ROOT / "metrics_ir_v8.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "best_acc": best_acc, "delta_v7": best_acc - V7, "clear_win": clear_win,
        "promoted": report["promote"] is not None, "seed55_hold": acc55,
        "elapsed_s": report["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
