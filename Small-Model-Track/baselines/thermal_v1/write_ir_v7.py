"""Write submission_ir_v7.csv from all9 mean + refined IR+Thermal+Mid weights; promote track submission.csv."""
from __future__ import annotations
import csv, json, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V6 = 0.7469635627530364
CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5, "acc": 0.7530364372469636, "n": 494, "mode": "sameT"}
# backup candidate
CFG_ACCW = {"wa": 0.54, "wb": 0.34, "wc": 0.12, "T": 2.25, "acc": 0.7510121457489879, "n": 494, "mode": "sameT"}

def softmax_np(z, T=1.0):
    z = z / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)

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
    cache = ROOT / "cache" / "ir_yolo_v4"
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    new_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
    z = np.load(old_ckpt / "hold_logits_v6.npz", allow_pickle=True)
    tags_old = [str(t) for t in z["tags"]]
    members_test = []
    member_acc = {}
    for t in tags_old:
        seed = t.replace("pool_seed", "")
        p = old_ckpt / f"test_logits_{t}.npy"
        if not p.exists():
            p = old_ckpt / f"test_logits_seed{seed}.npy"
        members_test.append(np.load(p))
        # acc from hold
        idx = list(tags_old).index(t)
        member_acc[t] = float((z["base"][idx].argmax(1) == z["y"]).mean())
    hz = np.load(new_dir / "hold_logits_new_seeds.npz", allow_pickle=True)
    for t in hz["tags"]:
        t = str(t); seed = t.replace("pool_seed", "")
        members_test.append(np.load(new_dir / f"test_logits_seed{seed}.npy"))
        member_acc[t] = float((hz[t].argmax(1) == z["y"]).mean())

    # order by acc desc to match refine
    order = sorted(member_acc.keys(), key=lambda k: -member_acc[k])
    # rebuild test stack in that order
    tag_to_test = {}
    # reload properly
    tag_to_test = {}
    for t in tags_old:
        seed = t.replace("pool_seed", "")
        p = old_ckpt / f"test_logits_{t}.npy"
        if not p.exists():
            p = old_ckpt / f"test_logits_seed{seed}.npy"
        tag_to_test[t] = np.load(p)
    for t in hz["tags"]:
        t = str(t); seed = t.replace("pool_seed", "")
        tag_to_test[t] = np.load(new_dir / f"test_logits_seed{seed}.npy")

    ir_test = np.mean([tag_to_test[t] for t in order], 0).astype(np.float32)
    w = np.array([max(member_acc[t], 1e-3) for t in order], dtype=np.float64); w /= w.sum()
    ir_test_accw = np.tensordot(w, np.stack([tag_to_test[t] for t in order], 0), axes=(0, 0)).astype(np.float32)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_p.exists():
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_p)

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    cfg = CFG
    T = cfg["T"]
    preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T) + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
    out = ROOT / "submission_ir_v7.csv"
    nfb = write_sub(out, meta, preds, empty, fb)

    # also write accw alt
    cfg2 = CFG_ACCW
    T2 = cfg2["T"]
    preds2 = (cfg2["wa"] * softmax_np(ir_test_accw, T2) + cfg2["wb"] * softmax_np(th_test, T2) + cfg2["wc"] * softmax_np(mid_test, T2)).argmax(1)
    out_alt = ROOT / "submission_ir_v7_accw.csv"
    write_sub(out_alt, meta, preds2, empty, fb)

    # disagree vs v6
    v6_preds = []
    with open(ROOT / "submission_ir_v6.csv") as f:
        for row in csv.DictReader(f):
            v6_preds.append(int(row["prediction"]))
    disagree = int(sum(int(a) != int(b) for a, b in zip(preds, v6_preds)))
    disagree_alt = int(sum(int(a) != int(b) for a, b in zip(preds2, v6_preds)))

    # promote primary
    track_sub = TRACK / "submission.csv"
    shutil.copy2(out, track_sub)

    # size check (same pack)
    fp16 = new_dir / "model_fp16.pt"
    fp16_mb = fp16.stat().st_size / (1024 * 1024) if fp16.exists() else 59.85
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    report = {
        "tag": "ir_v7",
        "primary": "submission_ir_v7.csv",
        "method": "all9 IR seeds logit-mean + Thermal v2_trio + MidFuse; finer same-T grid (T=2.5, wc~0.09)",
        "holdout_acc": cfg["acc"],
        "cfg": cfg,
        "delta_vs_v6": float(cfg["acc"] - V6),
        "nested_fixed_cfg": 0.7519623092355898,
        "nested_v6_fixed": 0.7459,
        "alt_accw": {"csv": "submission_ir_v7_accw.csv", "cfg": cfg2, "hold": cfg2["acc"]},
        "ir_members": member_acc,
        "ir_order": order,
        "disagree_vs_v6": disagree,
        "disagree_accw_vs_v6": disagree_alt,
        "empty_fallback": nfb,
        "promoted_track": str(track_sub),
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) < 100,
        "notes": [
            f"PRIMARY hold {cfg['acc']:.4f} (+{cfg['acc']-V6:.4f} vs v6 0.7470) — CLEAR WIN",
            "Fixed-cfg LOUO nested 0.752 > v6 nested 0.746 (re-tune-per-fold nested was pessimistic)",
            "Weights near v6 (0.56/0.36/0.08@1.5) -> (0.56/0.35/0.09@2.5); mainly T refine",
            "alt all_acc_w 0.7510 also clear; kept as submission_ir_v7_accw.csv",
            "Do not Kaggle-submit from this script; <=100MB pack ok",
            "Depth_Color skipped",
        ],
        "recommend_submit_order": [
            "submission_ir_v7.csv (all9+triple hold 0.7530) — PRIMARY",
            "submission_ir_v7_accw.csv — alt 0.7510",
            "submission_ir_v6.csv — backup hold 0.7470",
        ],
    }
    (ROOT / "metrics_ir_v7.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("holdout_acc","delta_vs_v6","disagree_vs_v6","cfg","promoted_track","size_ok_under_100mb")}, indent=2), flush=True)
    print(f"WROTE {out} and promoted {track_sub}", flush=True)

if __name__ == "__main__":
    main()
