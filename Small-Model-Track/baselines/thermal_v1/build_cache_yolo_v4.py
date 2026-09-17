"""v4 improved YOLO person-crop cache: IR / Depth_Color / Thermal.

Improvements vs v1:
- IR PNG support + percentile contrast stretch (L->RGB)
- Multi-frame detection (quartile + mid), pick highest-conf box
- Lower conf for Depth; motion-energy bbox fallback; center crop last
- Track detect stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from dataset import (
    discover_test_clips,
    discover_train_clips,
    list_frame_paths,
    sample_indices,
    save_json,
    TRAIN_ROOT,
    TEST_ROOT,
)

ROOT = Path(__file__).resolve().parent


def list_frame_paths_v4(clip_dir: Path, modality: str) -> list[Path]:
    m = modality.lower()
    frames = sorted(clip_dir.glob("*.png")) + sorted(clip_dir.glob("*.jpg"))
    frames = sorted(set(frames), key=lambda p: p.name)
    return frames


def discover_train_clips_v4(modality: str):
    root = TRAIN_ROOT / modality
    clips = []
    if not root.exists():
        raise FileNotFoundError(root)
    for act in sorted(root.iterdir()):
        if not act.is_dir():
            continue
        try:
            aid = int(act.name.split("_")[0])
        except ValueError:
            continue
        for user in sorted(act.iterdir()):
            if not user.is_dir() or not user.name.startswith("user"):
                continue
            uid = int(user.name.replace("user", ""))
            for trial in sorted(user.iterdir()):
                if not trial.is_dir():
                    continue
                frames = list_frame_paths_v4(trial, modality)
                if len(frames) == 0:
                    continue
                clips.append(
                    {
                        "clip_dir": str(trial),
                        "label": aid,
                        "user_id": uid,
                        "action_name": act.name,
                        "trial": trial.name,
                        "n_frames": len(frames),
                        "modality": modality,
                    }
                )
    return clips


def discover_test_clips_v4(modality: str):
    clips = []
    for s in sorted(TEST_ROOT.glob("SM_test_*")):
        d = s / modality
        frames = list_frame_paths_v4(d, modality) if d.exists() else []
        rel = f"small_model_track_test/{s.name}/"
        clips.append(
            {
                "sample_id": s.name,
                "path": rel,
                "clip_dir": str(d),
                "n_frames": len(frames),
                "empty": len(frames) == 0,
                "modality": modality,
            }
        )
    return clips


def open_rgb(path: Path, modality: str) -> Image.Image:
    img = Image.open(path)
    if modality.lower() == "ir" or img.mode == "L":
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return img.convert("RGB")


def motion_box(frame_paths: list[Path], modality: str, idxs: list[int]) -> tuple[float, float, float, float] | None:
    """Return normalized xyxy from motion energy centroid, or None."""
    try:
        a = np.asarray(open_rgb(frame_paths[idxs[0]], modality).convert("L"), dtype=np.float32)
        b = np.asarray(open_rgb(frame_paths[idxs[-1]], modality).convert("L"), dtype=np.float32)
        if a.shape != b.shape:
            return None
        d = np.abs(a - b)
        if float(d.mean()) < 1.5:
            return None
        thr = np.percentile(d, 85)
        mask = d >= thr
        ys, xs = np.where(mask)
        if len(xs) < 50:
            return None
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        h, w = a.shape
        # pad
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - 0.2 * bw)
        y1 = max(0, y1 - 0.2 * bh)
        x2 = min(w - 1, x2 + 0.2 * bw)
        y2 = min(h - 1, y2 + 0.2 * bh)
        return x1 / w, y1 / h, x2 / w, y2 / h
    except Exception:
        return None


def detect_norm_box(frame_paths: list[Path], modality: str, model: YOLO, conf: float) -> tuple[tuple[float, float, float, float], str]:
    n = len(frame_paths)
    if n == 0:
        return (0.075, 0.075, 0.925, 0.925), "empty"
    idxs = sorted(set([n // 4, n // 2, (3 * n) // 4, max(0, n - 1)]))
    best = None  # (conf, nx1,ny1,nx2,ny2)
    for i in idxs:
        try:
            img = open_rgb(frame_paths[i], modality)
        except Exception:
            continue
        w, h = img.size
        res = model.predict(img, classes=[0], verbose=False, conf=conf)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        # prefer high conf * area
        score = confs * np.sqrt(np.maximum(areas, 1.0))
        j = int(score.argmax())
        x1, y1, x2, y2 = xyxy[j]
        cand = (float(confs[j]), x1 / w, y1 / h, x2 / w, y2 / h)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is not None:
        return best[1:], "yolo"
    mb = motion_box(frame_paths, modality, idxs)
    if mb is not None:
        return mb, "motion"
    return (0.075, 0.075, 0.925, 0.925), "center"


def apply_box(img: Image.Image, nx1, ny1, nx2, ny2, pad: float = 0.15) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = nx1 * w, ny1 * h, nx2 * w, ny2 * h
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    x1 = max(0, x1 - pad * bw)
    y1 = max(0, y1 - pad * bh)
    x2 = min(w, x2 + pad * bw)
    y2 = min(h, y2 + pad * bh)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 16)
    half = side / 2
    left = int(max(0, min(w - side, cx - half)))
    top = int(max(0, min(h - side, cy - half)))
    side = int(min(side, w - left, h - top))
    return img.crop((left, top, left + side, top + side))


def resize_clip(frame_paths, t: int, size: int, model: YOLO, modality: str, conf: float, stats: dict) -> np.ndarray:
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    n = len(frame_paths)
    if n == 0:
        stats["empty"] = stats.get("empty", 0) + 1
        return out
    idxs = sample_indices(n, t, train=False)
    box, src = detect_norm_box(frame_paths, modality, model, conf)
    stats[src] = stats.get(src, 0) + 1
    nx1, ny1, nx2, ny2 = box
    for i, fi in enumerate(idxs):
        try:
            img = open_rgb(frame_paths[int(fi)], modality)
        except Exception:
            continue
        crop = apply_box(img, nx1, ny1, nx2, ny2).resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(crop, dtype=np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="IR", choices=["IR", "Depth_Color", "Thermal"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--weights", type=str, default=str(ROOT / "yolov8n.pt"))
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--tag", type=str, default="v4")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.conf is None:
        args.conf = 0.05 if args.modality == "Depth_Color" else 0.10

    out_dir = ROOT / "cache" / f"{args.modality.lower()}_yolo_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    model.predict(np.zeros((112, 112, 3), dtype=np.uint8), classes=[0], verbose=False, device=args.device)

    train_clips = discover_train_clips_v4(args.modality)
    test_clips = discover_test_clips_v4(args.modality)
    print(f"modality={args.modality} conf={args.conf} train={len(train_clips)} test={len(test_clips)} out={out_dir}", flush=True)

    stats = {}
    x_path = out_dir / f"train_x_t{args.t}_s{args.size}.npy"
    if args.force or not x_path.exists():
        N = len(train_clips)
        X = np.memmap(x_path, dtype=np.uint8, mode="w+", shape=(N, args.t, args.size, args.size, 3))
        y = np.zeros(N, dtype=np.int64)
        users = np.zeros(N, dtype=np.int64)
        for i, c in enumerate(tqdm(train_clips, desc=f"yolo-train-{args.modality}")):
            frames = list_frame_paths_v4(Path(c["clip_dir"]), args.modality)
            X[i] = resize_clip(frames, args.t, args.size, model, args.modality, args.conf, stats)
            y[i] = c["label"]
            users[i] = c["user_id"]
            if i % 40 == 0:
                X.flush()
        X.flush()
        np.save(out_dir / "train_y.npy", y)
        np.save(out_dir / "train_users.npy", users)
        save_json(out_dir / "train_meta.json", train_clips)
        save_json(out_dir / "detect_stats_train.json", stats)
        print("train detect stats", stats, flush=True)
    else:
        print("skip existing train cache", x_path)

    stats_t = {}
    tx = out_dir / f"test_x_t{args.t}_s{args.size}.npy"
    if args.force or not tx.exists():
        Nt = len(test_clips)
        X = np.memmap(tx, dtype=np.uint8, mode="w+", shape=(Nt, args.t, args.size, args.size, 3))
        empty = []
        for i, c in enumerate(tqdm(test_clips, desc=f"yolo-test-{args.modality}")):
            frames = list_frame_paths_v4(Path(c["clip_dir"]), args.modality) if not c["empty"] else []
            X[i] = resize_clip(frames, args.t, args.size, model, args.modality, args.conf, stats_t)
            if c["empty"] or len(frames) == 0:
                empty.append(c["sample_id"])
            if i % 20 == 0:
                X.flush()
        X.flush()
        save_json(out_dir / "test_empty.json", empty)
        save_json(out_dir / "test_meta.json", test_clips)
        save_json(out_dir / "detect_stats_test.json", stats_t)
        print("empty", len(empty), "test detect stats", stats_t, flush=True)
    print("done", out_dir)


if __name__ == "__main__":
    main()
