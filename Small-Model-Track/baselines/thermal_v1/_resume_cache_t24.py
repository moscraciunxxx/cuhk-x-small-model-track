"""Resume IR YOLO T=24 cache from partial memmap, then test split, then train R2P1D-18."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from build_cache_yolo_v4 import (
    discover_train_clips_v4, discover_test_clips_v4, list_frame_paths_v4, resize_clip
)
from dataset import save_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "cache" / "ir_yolo_v4_t24"
T, SIZE = 24, 112
START = 339  # first zero index; clips 0..338 filled

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "yolov8n.pt"))
    model.predict(np.zeros((112, 112, 3), dtype=np.uint8), classes=[0], verbose=False, device="0")
    train_clips = discover_train_clips_v4("IR")
    test_clips = discover_test_clips_v4("IR")
    N = len(train_clips)
    assert N == 2933, N
    x_path = OUT / f"train_x_t{T}_s{SIZE}.npy"
    X = np.memmap(x_path, dtype=np.uint8, mode="r+", shape=(N, T, SIZE, SIZE, 3))
    y = np.zeros(N, dtype=np.int64)
    users = np.zeros(N, dtype=np.int64)
    # rebuild labels for all; refill crops from START
    stats = {}
    for i, c in enumerate(tqdm(train_clips, desc="resume-train-IR")):
        y[i] = c["label"]; users[i] = c["user_id"]
        if i < START:
            continue
        frames = list_frame_paths_v4(Path(c["clip_dir"]), "IR")
        X[i] = resize_clip(frames, T, SIZE, model, "IR", 0.10, stats)
        if i % 40 == 0:
            X.flush()
    X.flush()
    np.save(OUT / "train_y.npy", y)
    np.save(OUT / "train_users.npy", users)
    save_json(OUT / "train_meta.json", train_clips)
    save_json(OUT / "detect_stats_train.json", {"resumed_from": START, **{k: stats.get(k) for k in stats}})
    print("train done", flush=True)

    stats_t = {}
    tx = OUT / f"test_x_t{T}_s{SIZE}.npy"
    Nt = len(test_clips)
    Xt = np.memmap(tx, dtype=np.uint8, mode="w+", shape=(Nt, T, SIZE, SIZE, 3))
    empty = []
    for i, c in enumerate(tqdm(test_clips, desc="yolo-test-IR")):
        frames = list_frame_paths_v4(Path(c["clip_dir"]), "IR") if not c["empty"] else []
        Xt[i] = resize_clip(frames, T, SIZE, model, "IR", 0.10, stats_t)
        if c["empty"] or len(frames) == 0:
            empty.append(c["sample_id"])
        if i % 20 == 0:
            Xt.flush()
    Xt.flush()
    save_json(OUT / "test_empty.json", empty)
    save_json(OUT / "test_meta.json", test_clips)
    save_json(OUT / "detect_stats_test.json", stats_t)
    print("empty", len(empty), "done", OUT, flush=True)

if __name__ == "__main__":
    main()


