"""Combine v5 (6 seeds) + new v6 seeds (1,777,333); search IR ens + triple fuse; rewrite v6 if better."""
from __future__ import annotations
import csv, json
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
V5 = 0.7388663967611336
V6_ROBUST = 0.7449392712550608
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


def fuse3(a, b, c, y, mask, ngrid=26):
    best = (-1.0, None)
    yt = y[mask]
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum())})
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
    device = torch.device("cuda")
    cache = ROOT / "cache" / "ir_yolo_v4"
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)

    # load cached old 6
    z = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_logits_v6.npz", allow_pickle=True)
    old_tags = [str(t) for t in z["tags"]]
    old_base = z["base"]; yt = z["y"]
    members = []
    for i, t in enumerate(old_tags):
        acc = float((old_base[i].argmax(1) == yt).mean())
        members.append({"tag": t, "logits": old_base[i], "acc": acc, "src": "v5"})

    # eval new seeds
    new_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
    for p in sorted(new_dir.glob("pool_seed*.pt")):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model = build().to(device); model.load_state_dict(blob["model"])
        lg, yt2 = eval_logits(model, loader, device)
        assert np.allclose(yt2, yt)
        acc = float((lg.argmax(1) == yt).mean())
        print(f"new {p.stem} hold={acc:.4f}", flush=True)
        members.append({"tag": p.stem, "logits": lg, "acc": acc, "src": "v6new", "state": blob["model"],
                        "test_logits": np.load(new_dir / f"test_logits_seed{p.stem.replace('pool_seed','')}.npy")})
        del model; torch.cuda.empty_cache()

    # also attach old test logits
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    for m in members:
        if m["src"] == "v5":
            m["test_logits"] = np.load(old_ckpt / f"test_logits_{m['tag']}.npy")

    members = sorted(members, key=lambda d: -d["acc"])
    print("ALL members:", [(m["tag"], round(m["acc"], 4), m["src"]) for m in members], flush=True)

    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")[hold_idx]
    mask = th.any(1) & mid.any(1)

    results = []
    # try top-k and all
    for k in range(3, len(members) + 1):
        ens = np.mean([m["logits"] for m in members[:k]], 0)
        ens_acc = float((ens.argmax(1) == yt).mean())
        b3_acc, b3 = fuse3(ens, th, mid, yt, mask)
        results.append({"k": k, "tags": [m["tag"] for m in members[:k]], "ens_acc": ens_acc,
                        "triple_acc": b3_acc, "cfg": b3})
        print(f"top{k} ens={ens_acc:.4f} triple={b3_acc:.4f} {b3}", flush=True)

    # also all9 equal
    ens_all = np.mean([m["logits"] for m in members], 0)
    # acc-weighted all
    w = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64); w /= w.sum()
    ens_w = np.tensordot(w, np.stack([m["logits"] for m in members], 0), axes=(0, 0))
    for name, elogs in [("all_mean", ens_all), ("all_acc_w", ens_w)]:
        ea = float((elogs.argmax(1) == yt).mean())
        ba, bc = fuse3(elogs, th, mid, yt, mask)
        results.append({"k": name, "tags": [m["tag"] for m in members], "ens_acc": ea, "triple_acc": ba, "cfg": bc})
        print(f"{name} ens={ea:.4f} triple={ba:.4f} {bc}", flush=True)

    best = max(results, key=lambda r: r["triple_acc"])
    print(f"\nBEST combine {best['k']} triple={best['triple_acc']:.6f} delta_v5={best['triple_acc']-V5:+.4f} delta_v6r={best['triple_acc']-V6_ROBUST:+.4f}", flush=True)

    report = {
        "members": {m["tag"]: m["acc"] for m in members},
        "results": results,
        "best": best,
        "v5": V5,
        "v6_robust_prev": V6_ROBUST,
        "improved_vs_v6_robust": bool(best["triple_acc"] > V6_ROBUST + 1e-6),
        "improved_vs_v5": bool(best["triple_acc"] > V5 + 0.002),
    }

    if best["triple_acc"] > V6_ROBUST + 1e-6:
        # rewrite primary submission
        tags = best["tags"]
        if best["k"] == "all_acc_w":
            ir_test = np.tensordot(w, np.stack([m["test_logits"] for m in members], 0), axes=(0, 0)).astype(np.float32)
            tag_sel = [m["tag"] for m in members]
        elif best["k"] == "all_mean":
            ir_test = np.mean([m["test_logits"] for m in members], 0).astype(np.float32)
            tag_sel = [m["tag"] for m in members]
        else:
            sel = [m for m in members if m["tag"] in tags]
            # preserve top-k order
            sel = sorted(sel, key=lambda d: -d["acc"])[: int(best["k"])]
            ir_test = np.mean([m["test_logits"] for m in sel], 0).astype(np.float32)
            tag_sel = [m["tag"] for m in sel]

        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
        if not th_p.exists():
            th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
        th_test = np.load(th_p)
        cfg = best["cfg"]
        T = cfg["T"]
        preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T) + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
            for row in csv.DictReader(f):
                fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
        out = ROOT / "submission_ir_v6.csv"
        nfb = write_sub(out, meta, preds, empty, fb)
        np.save(new_dir / "test_logits_ens_combined.npy", ir_test)
        report["wrote"] = str(out)
        report["ir_tags_used"] = tag_sel
        report["empty_fallback"] = nfb
        print(f"WROTE {out} hold={best['triple_acc']:.4f} tags={tag_sel}", flush=True)
    else:
        print("No improvement over robust v6 0.7449; keep existing submission_ir_v6.csv", flush=True)

    # update metrics
    prev = {}
    mp = ROOT / "metrics_ir_v6.json"
    if mp.exists():
        prev = json.loads(mp.read_text(encoding="utf-8"))
    prev["new_seeds"] = report
    if report.get("wrote"):
        prev["holdout_acc"] = best["triple_acc"]
        prev["cfg"] = best["cfg"]
        prev["delta_vs_v5"] = float(best["triple_acc"] - V5)
        prev["method"] = f"top-{best['k']} IR seeds incl new + Thermal + Mid finer same-T"
        prev["ir_top4"] = best.get("tags")  # may be more than 4
        prev["notes"] = [
            f"PRIMARY hold {best['triple_acc']:.4f} (+{best['triple_acc']-V5:.4f} vs v5)",
            f"New seeds 1/777/333 alone weak but combined selection chose k={best['k']}",
            "perT overfit rejected earlier",
            "Do not Kaggle-submit from this script",
            "Depth_Color skipped",
        ]
    mp.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v6_combine.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "improved": report["improved_vs_v6_robust"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
