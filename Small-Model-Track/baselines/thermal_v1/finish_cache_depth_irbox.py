"""Fast finish: train meta from IR discovery (no re-detect); build test with IR boxes."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from build_cache_yolo_v4 import (
    discover_train_clips_v4, discover_test_clips_v4, list_frame_paths_v4,
    detect_norm_box,
)
from build_cache_depth_irbox import pair_depth_clip, resize_clip_with_box
from dataset import save_json

ROOT = Path(__file__).resolve().parent
out_dir = ROOT / "cache" / "depth_color_yolo_v4_irbox"
t, size, conf = 16, 112, 0.10

ir_train = discover_train_clips_v4("IR")
y = np.load(out_dir / "train_y.npy")
users = np.load(out_dir / "train_users.npy")
assert len(ir_train) == len(y) == len(users), (len(ir_train), len(y), len(users))

# train meta without boxes (alignment keys only) — X already built
meta = []
for i, c in enumerate(ir_train):
    dc_dir = pair_depth_clip(c)
    dc_n = len(list_frame_paths_v4(dc_dir, "Depth_Color")) if dc_dir.exists() else 0
    mc = {
        "clip_dir": str(dc_dir),
        "label": int(c["label"]),
        "user_id": int(c["user_id"]),
        "action_name": c["action_name"],
        "trial": c["trial"],
        "n_frames": dc_n,
        "modality": "Depth_Color",
        "box_src": "ir_transfer_v4",
    }
    assert int(y[i]) == int(c["label"]) and int(users[i]) == int(c["user_id"])
    meta.append(mc)
save_json(out_dir / "train_meta.json", meta)
# approximate train detect stats from IR cache (known ~99% yolo)
save_json(out_dir / "detect_stats_train.json", {"note": "boxes from IR transfer; see ir_yolo_v4 stats", "n": len(meta)})
print("train meta written", len(meta), flush=True)

model = YOLO(str(ROOT / "yolov8n.pt"))
model.predict(np.zeros((112, 112, 3), dtype=np.uint8), classes=[0], verbose=False, device="0")
ir_test = discover_test_clips_v4("IR")
stats_t = {"yolo_ir": 0, "motion_ir": 0, "center": 0, "depth_missing": 0, "empty": 0}
tx = out_dir / f"test_x_t{t}_s{size}.npy"
Nt = len(ir_test)
X = np.memmap(tx, dtype=np.uint8, mode="w+", shape=(Nt, t, size, size, 3))
empty, tmeta = [], []
for i, c in enumerate(tqdm(ir_test, desc="test-irbox")):
    ir_dir = Path(c["clip_dir"])
    ir_frames = list_frame_paths_v4(ir_dir, "IR") if ir_dir.exists() else []
    if len(ir_frames) == 0:
        box, src = (0.075, 0.075, 0.925, 0.925), "center"
        stats_t["empty"] += 1
    else:
        box, src = detect_norm_box(ir_frames, "IR", model, conf)
    box = tuple(float(x) for x in box)
    key = "yolo_ir" if src == "yolo" else ("motion_ir" if src == "motion" else "center")
    stats_t[key] = stats_t.get(key, 0) + 1
    dc_dir = ir_dir.parent / "Depth_Color"
    dc_frames = list_frame_paths_v4(dc_dir, "Depth_Color") if dc_dir.exists() else []
    if len(dc_frames) == 0:
        stats_t["depth_missing"] += 1
        X[i] = 0
        empty.append(c["sample_id"])
    else:
        X[i] = resize_clip_with_box(dc_frames, t, size, box, "Depth_Color")
    tmeta.append({
        "sample_id": c["sample_id"], "path": c["path"], "clip_dir": str(dc_dir),
        "n_frames": len(dc_frames), "empty": len(dc_frames) == 0,
        "modality": "Depth_Color", "box_src": src, "box": list(box),
    })
    if i % 20 == 0:
        X.flush()
X.flush()
save_json(out_dir / "test_empty.json", empty)
save_json(out_dir / "test_meta.json", tmeta)
save_json(out_dir / "detect_stats_test.json", stats_t)
print("test stats", stats_t, "empty", len(empty), flush=True)
print("DONE", out_dir, flush=True)
