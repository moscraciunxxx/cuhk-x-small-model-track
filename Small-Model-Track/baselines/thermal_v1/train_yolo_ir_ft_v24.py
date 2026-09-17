"""Finetune YOLOv8n on IR person pseudo-boxes (domain adapt), then rebuild light IR crop cache.
Angle 2 of ir_v24. Pseudo-labels from frozen yolov8n detections on IR train frames (quartile samples).
"""
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from build_cache_yolo_v4 import discover_train_clips_v4, list_frame_paths_v4

ROOT = Path(__file__).resolve().parent


def ir_to_rgb_u8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo + 1:
        lo, hi = float(arr.min()), float(arr.max() + 1)
    arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def export_pseudo(out_dir: Path, conf: float = 0.15, max_per_clip: int = 4, seed: int = 0):
    rng = random.Random(seed)
    model = YOLO(str(ROOT / "yolov8n.pt"))
    clips = discover_train_clips_v4("IR")
    img_dir = out_dir / "images" / "train"
    lbl_dir = out_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for ci, c in enumerate(tqdm(clips, desc="pseudo-IR")):
        frames = list_frame_paths_v4(Path(c["clip_dir"]), "IR")
        if not frames:
            continue
        # quartile + mid indices
        idxs = sorted(set([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]))
        if len(idxs) > max_per_clip:
            idxs = sorted(rng.sample(idxs, max_per_clip))
        for fi in idxs:
            p = frames[fi]
            rgb = ir_to_rgb_u8(p)
            h, w = rgb.shape[:2]
            res = model.predict(rgb, classes=[0], conf=conf, verbose=False, device="0")
            boxes = []
            if res and len(res[0].boxes):
                for b in res[0].boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                    if bw > 0.02 and bh > 0.02:
                        boxes.append((0, cx, cy, bw, bh))
            if not boxes:
                continue
            stem = f"u{c['user_id']}_a{c['label']}_t{c['trial']}_f{fi}"
            Image.fromarray(rgb).save(img_dir / f"{stem}.jpg", quality=92)
            with open(lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                for cls, cx, cy, bw, bh in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    n_box += 1
            n_img += 1
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    meta = {"n_img": n_img, "n_box": n_box, "conf": conf}
    (out_dir / "pseudo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta, flush=True)
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()
    dset = ROOT / "cache" / "ir_yolo_ft_pseudo_v24"
    yaml = export_pseudo(dset, conf=args.conf)
    if args.export_only:
        return
    model = YOLO(str(ROOT / "yolov8n.pt"))
    run = model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(ROOT / "checkpoints" / "yolo_ir_ft_v24"),
        name="ft",
        exist_ok=True,
        patience=8,
        lr0=1e-3,
        workers=0,
        verbose=True,
    )
    best = Path(run.save_dir) / "weights" / "best.pt"
    dest = ROOT / "checkpoints" / "yolo_ir_ft_v24" / "yolov8n_ir_ft_best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dest)
        print("copied", dest, flush=True)
    print("done YOLO-FT", flush=True)


if __name__ == "__main__":
    main()
