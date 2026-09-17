"""Infer Thermal YOLO+R2Plus1D; MidFuse fallback for empty Thermal test clips."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from model_r2p1d import build_model_r2p1d, model_size_mb

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
MID = TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "thermal_yolo"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "submission_thermal_v1.csv"))
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--min-holdout-acc", type=float, default=0.55)
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    if args.ckpt:
        ckpt = Path(args.ckpt)
    else:
        cands = sorted((ROOT / "checkpoints").rglob("holdout_train.pt"), key=lambda p: p.stat().st_mtime)
        ckpt = cands[-1]
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("ckpt", ckpt, "val_acc", blob.get("val_acc"), "val_f1", blob.get("val_f1"))
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    X = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r", shape=(len(meta), args.t, args.size, args.size, 3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_r2p1d(40, in_ch=3, base=args.base).to(device)
    model.load_state_dict(blob["model"]); model.eval()
    print("size_mb", round(model_size_mb(model), 2))
    logits = np.zeros((len(meta), 40), np.float32)
    with torch.no_grad():
        for i in range(0, len(meta), 32):
            arr = X[i:i+32].astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
            logits[i:i+len(x)] = model(x).cpu().numpy()
    preds = logits.argmax(1)
    fb = {}
    with MID.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    rows=[]; nfb=0
    for i, m in enumerate(meta):
        path = m["path"] if m["path"].endswith("/") else m["path"] + "/"
        if m.get("empty") or m["sample_id"] in empty or m.get("n_frames", 1) == 0:
            pred = fb.get(path, int(preds[i])); nfb += 1
        else:
            pred = int(preds[i])
        rows.append((path, pred))
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "prediction"]); w.writerows(rows)
    np.save(ckpt.parent / "test_logits.npy", logits)
    print("wrote", out, "n", len(rows), "fallback", nfb)
    hold = float(blob.get("val_acc") or 0)
    if args.promote and hold >= args.min_holdout_acc:
        dest = TRACK / "submission.csv"
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print("PROMOTED", dest)
    else:
        print("not promoted holdout_acc", hold, "threshold", args.min_holdout_acc)

if __name__ == "__main__":
    main()
