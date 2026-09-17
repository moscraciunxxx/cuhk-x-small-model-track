"""YOLO person-crop frame cache (matches public LB notebooks approach)."""
from __future__ import annotations

import argparse
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
)

ROOT = Path(__file__).resolve().parent


def crop_person(img: Image.Image, model: YOLO, pad: float = 0.15) -> Image.Image:
    w, h = img.size
    res = model.predict(img, classes=[0], verbose=False, conf=0.15)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        # center square fallback
        side = int(min(w, h) * 0.85)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))
    # largest person box
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(areas.argmax())]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - pad * bw)
    y1 = max(0, y1 - pad * bh)
    x2 = min(w, x2 + pad * bw)
    y2 = min(h, y2 + pad * bh)
    # square
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    half = side / 2
    left = int(max(0, min(w - side, cx - half)))
    top = int(max(0, min(h - side, cy - half)))
    side = int(min(side, w - left, h - top))
    return img.crop((left, top, left + side, top + side))


def resize_clip(frame_paths, t: int, size: int, model: YOLO) -> np.ndarray:
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    n = len(frame_paths)
    if n == 0:
        return out
    idxs = sample_indices(n, t, train=False)
    # detect box on middle frame, apply to all (stable crop)
    mid = frame_paths[int(idxs[len(idxs) // 2])]
    try:
        mid_img = Image.open(mid).convert("RGB")
        box_img = crop_person(mid_img, model)
        # recover box coords roughly by matching - simpler: recompute crop each frame with same relative? 
        # For speed/stability: get xyxy once from mid and apply scaled to each frame size
        res = model.predict(mid_img, classes=[0], verbose=False, conf=0.15)
        boxes = res[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            x1, y1, x2, y2 = xyxy[int(areas.argmax())]
            mw, mh = mid_img.size
            # normalized box
            nx1, ny1, nx2, ny2 = x1 / mw, y1 / mh, x2 / mw, y2 / mh
        else:
            nx1, ny1, nx2, ny2 = 0.1, 0.1, 0.9, 0.9
    except Exception:
        nx1, ny1, nx2, ny2 = 0.1, 0.1, 0.9, 0.9

    for i, fi in enumerate(idxs):
        try:
            img = Image.open(frame_paths[int(fi)]).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        x1, y1, x2, y2 = nx1 * w, ny1 * h, nx2 * w, ny2 * h
        bw, bh = x2 - x1, y2 - y1
        pad = 0.15
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
        crop = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(crop, dtype=np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="Depth_Color", choices=["Depth_Color", "Thermal"])
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--weights", type=str, default=str(ROOT / "yolov8n.pt"))
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    out_dir = ROOT / "cache" / f"{args.modality.lower()}_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    # warmup
    model.predict(np.zeros((112, 112, 3), dtype=np.uint8), classes=[0], verbose=False, device=args.device)

    train_clips = discover_train_clips(args.modality)
    test_clips = discover_test_clips(args.modality)
    print(f"train={len(train_clips)} test={len(test_clips)} out={out_dir}")

    x_path = out_dir / f"train_x_t{args.t}_s{args.size}.npy"
    if not x_path.exists():
        N = len(train_clips)
        X = np.memmap(x_path, dtype=np.uint8, mode="w+", shape=(N, args.t, args.size, args.size, 3))
        y = np.zeros(N, dtype=np.int64)
        users = np.zeros(N, dtype=np.int64)
        for i, c in enumerate(tqdm(train_clips, desc="yolo-train")):
            frames = list_frame_paths(Path(c["clip_dir"]), args.modality)
            X[i] = resize_clip(frames, args.t, args.size, model)
            y[i] = c["label"]
            users[i] = c["user_id"]
            if i % 50 == 0:
                X.flush()
        X.flush()
        np.save(out_dir / "train_y.npy", y)
        np.save(out_dir / "train_users.npy", users)
        save_json(out_dir / "train_meta.json", train_clips)

    tx = out_dir / f"test_x_t{args.t}_s{args.size}.npy"
    if not tx.exists():
        Nt = len(test_clips)
        X = np.memmap(tx, dtype=np.uint8, mode="w+", shape=(Nt, args.t, args.size, args.size, 3))
        empty = []
        for i, c in enumerate(tqdm(test_clips, desc="yolo-test")):
            frames = list_frame_paths(Path(c["clip_dir"]), args.modality) if not c["empty"] else []
            X[i] = resize_clip(frames, args.t, args.size, model)
            if c["empty"] or len(frames) == 0:
                empty.append(c["sample_id"])
            if i % 20 == 0:
                X.flush()
        X.flush()
        save_json(out_dir / "test_empty.json", empty)
        save_json(out_dir / "test_meta.json", test_clips)
        print("empty", len(empty))
    print("done")


if __name__ == "__main__":
    main()
