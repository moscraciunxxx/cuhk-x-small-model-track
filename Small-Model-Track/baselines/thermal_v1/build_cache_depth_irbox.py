"""Build Depth_Color YOLO cache using IR-detected person boxes (synced 640x480).
IR detect ~99% YOLO vs Depth ~65%; transfer normalized boxes across modalities.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from build_cache_yolo_v4 import (
    discover_train_clips_v4, discover_test_clips_v4, list_frame_paths_v4,
    open_rgb, detect_norm_box, apply_box, motion_box,
)
from dataset import sample_indices, save_json

ROOT = Path(__file__).resolve().parent


def resize_clip_with_box(frame_paths, t, size, box, modality):
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    n = len(frame_paths)
    if n == 0:
        return out
    idxs = sample_indices(n, t, train=False)
    nx1, ny1, nx2, ny2 = box
    for i, fi in enumerate(idxs):
        try:
            img = open_rgb(frame_paths[int(fi)], modality)
        except Exception:
            continue
        crop = apply_box(img, nx1, ny1, nx2, ny2).resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(crop, dtype=np.uint8)
    return out


def pair_depth_clip(ir_clip: dict) -> Path:
    """Map IR train clip_dir -> Depth_Color clip_dir."""
    p = Path(ir_clip["clip_dir"])
    # .../IR/<act>/userX/<trial> -> .../Depth_Color/<act>/userX/<trial>
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "IR":
            parts[i] = "Depth_Color"
            break
    else:
        raise RuntimeError(f"no IR segment in {p}")
    return Path(*parts)


def pair_test_depth(sample_id: str, test_root_ir_clip: dict) -> Path:
    p = Path(test_root_ir_clip["clip_dir"])  # .../SM_test_xxxx/IR
    return p.parent / "Depth_Color"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--weights", type=str, default=str(ROOT / "yolov8n.pt"))
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.10)  # IR conf
    ap.add_argument("--tag", type=str, default="v4_irbox")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "cache" / f"depth_color_yolo_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    model.predict(np.zeros((112, 112, 3), dtype=np.uint8), classes=[0], verbose=False, device=args.device)

    ir_train = discover_train_clips_v4("IR")
    ir_test = discover_test_clips_v4("IR")
    print(f"IR train={len(ir_train)} test={len(ir_test)} out={out_dir} conf={args.conf}", flush=True)

    # TRAIN
    x_path = out_dir / f"train_x_t{args.t}_s{args.size}.npy"
    stats = {"yolo_ir": 0, "motion_ir": 0, "center": 0, "depth_missing": 0, "empty": 0}
    if args.force or not x_path.exists():
        N = len(ir_train)
        X = np.memmap(x_path, dtype=np.uint8, mode="w+", shape=(N, args.t, args.size, args.size, 3))
        y = np.zeros(N, dtype=np.int64)
        users = np.zeros(N, dtype=np.int64)
        meta = []
        boxes_save = []
        for i, c in enumerate(tqdm(ir_train, desc="depth-from-irbox-train")):
            ir_frames = list_frame_paths_v4(Path(c["clip_dir"]), "IR")
            box, src = detect_norm_box(ir_frames, "IR", model, args.conf)
            key = "yolo_ir" if src == "yolo" else ("motion_ir" if src == "motion" else "center")
            stats[key] = stats.get(key, 0) + 1
            dc_dir = pair_depth_clip(c)
            dc_frames = list_frame_paths_v4(dc_dir, "Depth_Color") if dc_dir.exists() else []
            if len(dc_frames) == 0:
                stats["depth_missing"] += 1
                X[i] = 0
            else:
                X[i] = resize_clip_with_box(dc_frames, args.t, args.size, box, "Depth_Color")
            y[i] = c["label"]
            users[i] = c["user_id"]
            mc = dict(c)
            mc["modality"] = "Depth_Color"
            mc["clip_dir"] = str(dc_dir)
            mc["box_src"] = src
            mc["box"] = [float(x) for x in box]
            mc["n_frames"] = len(dc_frames)
            meta.append(mc)
            boxes_save.append({"box": list(box), "src": src})
            if i % 40 == 0:
                X.flush()
        X.flush()
        np.save(out_dir / "train_y.npy", y)
        np.save(out_dir / "train_users.npy", users)
        save_json(out_dir / "train_meta.json", meta)
        save_json(out_dir / "detect_stats_train.json", stats)
        np.save(out_dir / "train_boxes.npy", np.array([b["box"] for b in boxes_save], dtype=np.float32))
        print("train stats", stats, flush=True)
    else:
        print("skip train", x_path)

    # TEST
    stats_t = {"yolo_ir": 0, "motion_ir": 0, "center": 0, "depth_missing": 0, "empty": 0}
    tx = out_dir / f"test_x_t{args.t}_s{args.size}.npy"
    if args.force or not tx.exists():
        Nt = len(ir_test)
        X = np.memmap(tx, dtype=np.uint8, mode="w+", shape=(Nt, args.t, args.size, args.size, 3))
        empty = []
        meta = []
        for i, c in enumerate(tqdm(ir_test, desc="depth-from-irbox-test")):
            ir_dir = Path(c["clip_dir"])
            ir_frames = list_frame_paths_v4(ir_dir, "IR") if ir_dir.exists() else []
            if len(ir_frames) == 0:
                box, src = (0.075, 0.075, 0.925, 0.925), "center"
                stats_t["empty"] += 1
            else:
                box, src = detect_norm_box(ir_frames, "IR", model, args.conf)
            key = "yolo_ir" if src == "yolo" else ("motion_ir" if src == "motion" else "center")
            stats_t[key] = stats_t.get(key, 0) + 1
            dc_dir = ir_dir.parent / "Depth_Color"
            dc_frames = list_frame_paths_v4(dc_dir, "Depth_Color") if dc_dir.exists() else []
            if len(dc_frames) == 0:
                stats_t["depth_missing"] += 1
                X[i] = 0
                empty.append(c["sample_id"])
            else:
                X[i] = resize_clip_with_box(dc_frames, args.t, args.size, box, "Depth_Color")
            mc = dict(c)
            mc["modality"] = "Depth_Color"
            mc["clip_dir"] = str(dc_dir)
            mc["box_src"] = src
            mc["box"] = [float(x) for x in box]
            mc["n_frames"] = len(dc_frames)
            meta.append(mc)
            if i % 20 == 0:
                X.flush()
        X.flush()
        save_json(out_dir / "test_empty.json", empty)
        save_json(out_dir / "test_meta.json", meta)
        save_json(out_dir / "detect_stats_test.json", stats_t)
        print("test stats", stats_t, "empty", len(empty), flush=True)
    print("done", out_dir)


if __name__ == "__main__":
    main()

