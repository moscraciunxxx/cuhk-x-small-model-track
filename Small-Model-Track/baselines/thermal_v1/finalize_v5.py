"""v5 finalize: multi-seed IR ens (+ optional TTA) + Thermal + MidFuse late-fuse; write submission_ir_v5.csv."""
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
V4_BLEND = 0.7145748987854251
V4_TRIPLE = 0.7307692307692307


def build():
    m = r2plus1d_18(weights=None)
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
def eval_logits(model, loader, device, tta=False):
    model.eval()
    outs, ys = [], []
    for x, y, _u, _i in loader:
        x = x.to(device)
        def _fwd(xx):
            xx = normalize(xx).permute(0, 2, 1, 3, 4).contiguous()
            return model(xx).float()
        logits = _fwd(x)
        if tta:
            # horizontal flip on spatial dims of (B,T,C,H,W) after our dataset gives (B,T,H,W,C) -> we have (B,T,C,H,W) after permute inside
            x_flip = torch.flip(x, dims=[-1])  # flip W on (B,T,H,W,C)
            logits = 0.5 * (logits + _fwd(x_flip))
        outs.append(logits.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


@torch.no_grad()
def infer_array(model, Xt, device, bs=8, tta=False):
    model.eval()
    n = len(Xt)
    o = np.zeros((n, NUM_CLASSES), np.float32)
    for i in range(0, n, bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(device)  # B,T,H,W,C
        def _fwd(xx):
            xx = xx.permute(0, 1, 4, 2, 3).contiguous()  # B,T,C,H,W
            xx = normalize(xx).permute(0, 2, 1, 3, 4).contiguous()
            return model(xx).float()
        logits = _fwd(x)
        if tta:
            x_flip = torch.flip(x, dims=[-2])  # flip W: (B,T,H,W,C) -> dim -2 is W? shape B,T,H,W,C so W is -2
            # B,T,H,W,C: dims (-2)=W
            logits = 0.5 * (logits + _fwd(torch.flip(x, dims=[-2])))
        o[i:i + len(x)] = logits.cpu().numpy()
    return o


def fuse2(a, b, y, mask):
    best = (-1.0, None)
    yt = y[mask]
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        pa, pb = softmax_np(a[mask], T), softmax_np(b[mask], T)
        for w in np.linspace(0, 1, 41):
            acc = float(((w * pa + (1 - w) * pb).argmax(1) == yt).mean())
            if acc > best[0]:
                best = (acc, {"w": float(w), "T": float(T), "acc": acc, "n": int(mask.sum())})
    return best


def fuse3(a, b, c, y, mask):
    best = (-1.0, None)
    yt = y[mask]
    for T in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, 21):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * 20)) + 1)):
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
    mid = np.zeros((len(ir_meta), 40), np.float32)
    hit = 0
    for i, m in enumerate(ir_meta):
        j = th_map.get(key(m))
        if j is not None:
            mid[i] = src[j]
            hit += 1
    np.save(cache / "midfuse_aligned_train_logits.npy", mid)
    print(f"aligned mid {hit}/{len(ir_meta)}", flush=True)
    return mid, "aligned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"))
    ap.add_argument("--tag", default="ir_v5")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    cache, ckpt_dir = Path(args.cache_dir), Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy")
    users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)

    members = []
    for p in sorted(ckpt_dir.glob("pool_seed*.pt")):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model = build().to(device)
        model.load_state_dict(blob["model"])
        logits, yt = eval_logits(model, loader, device, tta=args.tta)
        acc = float((logits.argmax(1) == yt).mean())
        print(f"{p.stem} hold={acc:.4f} tta={args.tta}", flush=True)
        members.append({"tag": p.stem, "acc": acc, "logits": logits, "state": blob["model"], "seed": blob.get("seed", p.stem)})
        del model
        torch.cuda.empty_cache()

    if not members:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")

    # mean ens + weighted-by-acc ens + softmax-mean ens
    stack = np.stack([m["logits"] for m in members], 0)
    ens_logit = stack.mean(0)
    w_acc = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64)
    w_acc = w_acc / w_acc.sum()
    ens_w = np.tensordot(w_acc, stack, axes=(0, 0))
    ens_sm = np.mean([softmax_np(m["logits"], 1.0) for m in members], 0)
    # convert sm back to log-prob for fuse grids that expect logits-ish; keep as probs with T=1 path via log
    ens_sm_logit = np.log(np.clip(ens_sm, 1e-8, 1.0))

    def acc_of(logits):
        return float((logits.argmax(1) == yt).mean())

    cands_ens = {
        "logit_mean": (ens_logit, acc_of(ens_logit)),
        "acc_weighted": (ens_w, acc_of(ens_w)),
        "softmax_mean": (ens_sm_logit, float((ens_sm.argmax(1) == yt).mean())),
    }
    for k, (lg, ac) in cands_ens.items():
        print(f"IR ens {k} hold={ac:.4f}", flush=True)
    best_ens_name = max(cands_ens, key=lambda k: cands_ens[k][1])
    ens, ens_acc = cands_ens[best_ens_name]
    print(f"BEST IR ens={best_ens_name} {ens_acc:.4f}", flush=True)

    mid, mid_src = align_mid_to_cache(cache)
    mid_h = mid[hold_idx]
    mask = mid_h.any(1)
    b2_acc, b2 = fuse2(ens, mid_h, yt, mask)
    print(f"IR+Mid fuse={b2_acc:.4f} {b2}", flush=True)

    # thermal hold ens mapped onto IR hold
    th_cache = ROOT / "cache" / "thermal_yolo"
    th_y = np.load(th_cache / "train_y.npy")
    th_u = np.load(th_cache / "train_users.npy")
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
        model = build().to(device)
        model.load_state_dict(blob["model"])
        logits, _ = eval_logits(model, th_loader, device, tta=False)
        th_logs.append(logits)
        del model
        torch.cuda.empty_cache()
        print("loaded thermal", p.name, flush=True)
    th_ens = np.mean(th_logs, 0)
    th_hold_meta = [th_meta[i] for i in th_hold]
    th_map = {key(m): j for j, m in enumerate(th_hold_meta)}
    th_on_ir = np.zeros_like(ens)
    mapped = 0
    for j, ii in enumerate(hold_idx):
        k = key(ir_meta[ii])
        if k in th_map:
            th_on_ir[j] = th_ens[th_map[k]]
            mapped += 1
    print(f"mapped thermal hold onto IR hold {mapped}/{len(hold_idx)}", flush=True)
    th_mask = th_on_ir.any(1) & mask
    b3_acc, b3 = fuse3(ens, th_on_ir, mid_h, yt, th_mask)
    print(f"IR+Thermal+Mid fuse={b3_acc:.4f} {b3}", flush=True)
    bt_acc, bt = fuse2(th_on_ir, mid_h, yt, th_mask)
    bi_acc, bi = fuse2(ens, th_on_ir, yt, th_on_ir.any(1))
    print(f"Thermal+Mid={bt_acc:.4f} {bt}", flush=True)
    print(f"IR+Thermal={bi_acc:.4f} {bi}", flush=True)

    # compromise: pull toward MidFuse a bit vs holdout-optimal (public-gap hedge)
    # average holdout-opt weights with more-mid variant
    compromise = {
        "wa": float(0.5 * b3["wa"] + 0.5 * max(b3["wa"] - 0.1, 0.35)),
        "wb": float(0.5 * b3["wb"] + 0.5 * b3["wb"]),
        "wc": None,
        "T": float(0.5 * (b3["T"] + 2.5)),
    }
    # renormalize wa,wb,wc with wc getting residual + a little boost
    wa_c = compromise["wa"]
    wb_c = min(b3["wb"], 0.25)
    wc_c = 1.0 - wa_c - wb_c
    if wc_c < 0.25:
        # boost mid
        s = wa_c + wb_c
        wa_c, wb_c = wa_c / s * 0.7, wb_c / s * 0.7
        wc_c = 0.3
    compromise.update({"wa": wa_c, "wb": wb_c, "wc": wc_c})
    pa, pb, pc = softmax_np(ens[th_mask], compromise["T"]), softmax_np(th_on_ir[th_mask], compromise["T"]), softmax_np(mid_h[th_mask], compromise["T"])
    comp_acc = float(((wa_c * pa + wb_c * pb + wc_c * pc).argmax(1) == yt[th_mask]).mean())
    compromise["acc"] = comp_acc
    print(f"compromise triple hold={comp_acc:.4f} {compromise}", flush=True)

    cands = [
        ("triple", b3_acc, b3, "3way"),
        ("ir_mid", b2_acc, b2, "2way_ir_mid"),
        ("compromise", comp_acc, compromise, "3way_comp"),
        ("ir_th", bi_acc, bi, "2way_ir_th"),
        ("th_mid", bt_acc, bt, "2way_th_mid"),
        ("ir_only", ens_acc, {"w": 1.0, "T": 1.0, "acc": ens_acc}, "ir_only"),
    ]
    best_name, best_acc, best_cfg, best_kind = max(cands, key=lambda x: x[1])
    print("BEST", best_name, best_acc, best_cfg, flush=True)

    # ---- test inference ----
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
    outs = []
    for m in members:
        model = build().to(device)
        model.load_state_dict(m["state"])
        o = infer_array(model, Xt, device, bs=8, tta=args.tta)
        np.save(ckpt_dir / f"test_logits_{m['tag']}.npy", o)
        outs.append(o)
        del model
        torch.cuda.empty_cache()
        print(f"inferred test {m['tag']}", flush=True)
    stack_t = np.stack(outs, 0)
    if best_ens_name == "acc_weighted":
        ir_test = np.tensordot(w_acc, stack_t, axes=(0, 0)).astype(np.float32)
    elif best_ens_name == "softmax_mean":
        ir_test = np.log(np.clip(np.mean([softmax_np(o, 1.0) for o in outs], 0), 1e-8, 1.0)).astype(np.float32)
    else:
        ir_test = stack_t.mean(0).astype(np.float32)
    np.save(ckpt_dir / "test_logits_ens.npy", ir_test)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    if not th_test_p.exists():
        th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    th_test = np.load(th_test_p)

    def pred_from(kind, cfg):
        if kind.startswith("3way"):
            T = cfg["T"]
            return (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T) + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
        if kind == "2way_ir_mid":
            return (cfg["w"] * softmax_np(ir_test, cfg["T"]) + (1 - cfg["w"]) * softmax_np(mid_test, cfg["T"])).argmax(1)
        if kind == "2way_ir_th":
            return (cfg["w"] * softmax_np(ir_test, cfg["T"]) + (1 - cfg["w"]) * softmax_np(th_test, cfg["T"])).argmax(1)
        if kind == "2way_th_mid":
            return (cfg["w"] * softmax_np(th_test, cfg["T"]) + (1 - cfg["w"]) * softmax_np(mid_test, cfg["T"])).argmax(1)
        return ir_test.argmax(1)

    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    pred_primary = pred_from(best_kind if best_kind != "3way_comp" else "3way", best_cfg)
    # Always also emit holdout-best triple and IR+Mid and compromise
    pred_triple = pred_from("3way", b3)
    pred_irmid = pred_from("2way_ir_mid", b2)
    pred_comp = pred_from("3way", compromise)

    out_primary = ROOT / "submission_ir_v5.csv"
    out_video = ROOT / "submission_ir_v5_video.csv"
    out_irmid = ROOT / "submission_ir_v5_irmid.csv"
    out_comp = ROOT / "submission_ir_v5_compromise.csv"
    out_triple = ROOT / "submission_ir_v5_triple.csv"

    nfb = write_sub(out_primary, meta, pred_primary, empty, fb)
    write_sub(out_video, meta, ir_test.argmax(1), empty, fb)
    write_sub(out_irmid, meta, pred_irmid, empty, fb)
    write_sub(out_comp, meta, pred_comp, empty, fb)
    write_sub(out_triple, meta, pred_triple, empty, fb)
    print(f"WROTE {out_primary} best={best_name} hold={best_acc:.4f}", flush=True)

    # fp16 pack from best member
    best_m = max(members, key=lambda d: d["acc"])
    fp16 = ckpt_dir / "model_fp16.pt"
    torch.save({
        "model_fp16": {k: (v.half() if v.is_floating_point() else v) for k, v in best_m["state"].items()},
        "val_acc": best_m["acc"],
        "seed": best_m.get("seed"),
    }, fp16)
    fp16_mb = fp16.stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    report = {
        "tag": args.tag,
        "tta": bool(args.tta),
        "cache": str(cache),
        "ckpt_dir": str(ckpt_dir),
        "member_scores": {m["tag"]: m["acc"] for m in members},
        "ens_method": best_ens_name,
        "holdout_acc_ensemble": ens_acc,
        "ens_candidates": {k: v[1] for k, v in cands_ens.items()},
        "holdout_acc_ir_mid": b2_acc,
        "ir_mid_cfg": b2,
        "holdout_acc_triple": b3_acc,
        "triple_cfg": b3,
        "holdout_acc_compromise": comp_acc,
        "compromise_cfg": compromise,
        "holdout_acc_ir_th": bi_acc,
        "ir_th_cfg": bi,
        "best_name": best_name,
        "best_acc": best_acc,
        "best_cfg": best_cfg,
        "delta_vs_v4_blend": float(best_acc - V4_BLEND),
        "delta_vs_v4_triple": float(best_acc - V4_TRIPLE),
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) < 100,
        "submission_primary": str(out_primary),
        "submission_video": str(out_video),
        "submission_irmid": str(out_irmid),
        "submission_compromise": str(out_comp),
        "submission_triple": str(out_triple),
        "empty_fallback": nfb,
        "holdout_users": list(HOLD),
        "mid_src": mid_src,
        "notes": [
            "PRIMARY=holdout-best fuse (usually IR+Thermal+Mid triple)",
            "submission_ir_v5_compromise.csv hedges MidFuse weight for public-gap",
            "submission_ir_v5_irmid.csv is IR+Mid only (matches prior public style)",
            "Do not Kaggle-submit from this script; CSV ready for next window",
            "Depth_Color skipped (holdout~0.21, YOLO~19%)",
        ],
    }
    (ROOT / "metrics_ir_v5.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
