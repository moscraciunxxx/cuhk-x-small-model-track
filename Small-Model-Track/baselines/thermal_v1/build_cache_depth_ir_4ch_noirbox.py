"""4ch non-IR-box Depth_Color YOLO crop + IR gray. Different crop from depth_ir_4ch_v31 (IR-box)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IR = ROOT / "cache" / "ir_yolo_v4"
DEPTH = ROOT / "cache" / "depth_color_yolo_v4"
IRBOX = ROOT / "cache" / "depth_color_yolo_v4_irbox"
OUT = ROOT / "cache" / "depth_ir_4ch_noirbox"
T, S, C_OUT = 16, 112, 4


def _keys(meta):
    return [(int(m["user_id"]), int(m["label"]), str(m.get("trial", ""))) for m in meta]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ir_tr = json.loads((IR / "train_meta.json").read_text(encoding="utf-8"))
    dp_tr = json.loads((DEPTH / "train_meta.json").read_text(encoding="utf-8"))
    ir_te = json.loads((IR / "test_meta.json").read_text(encoding="utf-8"))
    dp_te = json.loads((DEPTH / "test_meta.json").read_text(encoding="utf-8"))
    if [m["sample_id"] for m in ir_te] != [m["sample_id"] for m in dp_te]:
        raise SystemExit("test sample_id mismatch")
    ntr, nte = len(ir_tr), len(ir_te)
    dmap = {k: i for i, k in enumerate(_keys(dp_tr))}
    ir_keys = _keys(ir_tr)
    miss = [i for i, k in enumerate(ir_keys) if k not in dmap]
    print(f"noirbox 4ch n_train={ntr} n_test={nte} depth_hits={ntr - len(miss)} fallback_irbox={len(miss)}", flush=True)

    ir_x = np.memmap(IR / f"train_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(ntr, T, S, S, 3))
    d_x = np.memmap(DEPTH / f"train_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(len(dp_tr), T, S, S, 3))
    fb_x = np.memmap(IRBOX / f"train_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(ntr, T, S, S, 3))
    out = np.memmap(OUT / f"train_x_t{T}_s{S}.npy", dtype=np.uint8, mode="w+", shape=(ntr, T, S, S, C_OUT))
    bs = 32
    for i in range(0, ntr, bs):
        sl = slice(i, min(i + bs, ntr))
        chunk = np.zeros((sl.stop - sl.start, T, S, S, C_OUT), np.uint8)
        gray = np.asarray(ir_x[sl]).mean(axis=-1).round().astype(np.uint8)
        for j, gi in enumerate(range(sl.start, sl.stop)):
            k = ir_keys[gi]
            if k in dmap:
                chunk[j, ..., :3] = d_x[dmap[k]]
            else:
                chunk[j, ..., :3] = fb_x[gi]
            chunk[j, ..., 3] = gray[j]
        out[sl] = chunk
        print(f"  train {sl.stop}/{ntr}", flush=True)
    out.flush()
    del ir_x, d_x, fb_x, out

    ir_t = np.memmap(IR / f"test_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(nte, T, S, S, 3))
    d_t = np.memmap(DEPTH / f"test_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(nte, T, S, S, 3))
    out_t = np.memmap(OUT / f"test_x_t{T}_s{S}.npy", dtype=np.uint8, mode="w+", shape=(nte, T, S, S, C_OUT))
    for i in range(0, nte, bs):
        sl = slice(i, min(i + bs, nte))
        db = np.asarray(d_t[sl])
        gray = np.asarray(ir_t[sl]).mean(axis=-1).round().astype(np.uint8)
        out_t[sl] = np.concatenate([db, gray[..., None]], axis=-1)
        print(f"  test {sl.stop}/{nte}", flush=True)
    out_t.flush()
    del ir_t, d_t, out_t

    shutil.copy2(IR / "train_y.npy", OUT / "train_y.npy")
    shutil.copy2(IR / "train_users.npy", OUT / "train_users.npy")
    shutil.copy2(IR / "train_meta.json", OUT / "train_meta.json")
    shutil.copy2(IR / "test_meta.json", OUT / "test_meta.json")
    shutil.copy2(IR / "test_empty.json", OUT / "test_empty.json")
    info = {
        "arch_input": "Depth_Color RGB non-IR-box YOLO crop + IR gray",
        "in_ch": 4, "t": T, "size": S,
        "differs_from": "cache/depth_ir_4ch_v31 IR-box crop T16 s112",
        "crop": "non-irbox depth_color_yolo_v4",
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T, S, S, C_OUT],
        "test_shape": [nte, T, S, S, C_OUT],
        "fallback_irbox_idx": miss,
        "aligned_to": "cache/ir_yolo_v4",
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
