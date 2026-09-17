"""4ch native T24: Depth IR-box sampled at T=24 from raw frames + IR T24 gray."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IR = ROOT / "cache" / "ir_yolo_v4_t24"
DEPTH = ROOT / "cache" / "depth_color_yolo_v4_irbox_t24"
OUT = ROOT / "cache" / "depth_ir_4ch_t24_native"
T, S, C_OUT = 24, 112, 4


def _keys_train(meta):
    return [(int(m["user_id"]), int(m["label"]), str(m.get("trial", ""))) for m in meta]


def _stack(n, ir_path, depth_path, out_path):
    ir = np.memmap(ir_path, dtype=np.uint8, mode="r", shape=(n, T, S, S, 3))
    d = np.memmap(depth_path, dtype=np.uint8, mode="r", shape=(n, T, S, S, 3))
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(n, T, S, S, C_OUT))
    bs = 32
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        db = np.asarray(d[sl])
        gray = np.asarray(ir[sl]).mean(axis=-1).round().astype(np.uint8)
        out[sl] = np.concatenate([db, gray[..., None]], axis=-1)
        print(f"  stacked {sl.stop}/{n}", flush=True)
    out.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ir_tr = json.loads((IR / "train_meta.json").read_text(encoding="utf-8"))
    dp_tr = json.loads((DEPTH / "train_meta.json").read_text(encoding="utf-8"))
    if _keys_train(ir_tr) != _keys_train(dp_tr):
        raise SystemExit("train keys mismatch")
    ir_te = json.loads((IR / "test_meta.json").read_text(encoding="utf-8"))
    dp_te = json.loads((DEPTH / "test_meta.json").read_text(encoding="utf-8"))
    if [m["sample_id"] for m in ir_te] != [m["sample_id"] for m in dp_te]:
        raise SystemExit("test sample_id mismatch")
    ntr, nte = len(ir_tr), len(ir_te)
    print(f"native T24 4ch n_train={ntr} n_test={nte}", flush=True)
    _stack(ntr, IR / f"train_x_t{T}_s{S}.npy", DEPTH / f"train_x_t{T}_s{S}.npy", OUT / f"train_x_t{T}_s{S}.npy")
    _stack(nte, IR / f"test_x_t{T}_s{S}.npy", DEPTH / f"test_x_t{T}_s{S}.npy", OUT / f"test_x_t{T}_s{S}.npy")
    shutil.copy2(IR / "train_y.npy", OUT / "train_y.npy")
    shutil.copy2(IR / "train_users.npy", OUT / "train_users.npy")
    shutil.copy2(IR / "train_meta.json", OUT / "train_meta.json")
    shutil.copy2(IR / "test_meta.json", OUT / "test_meta.json")
    shutil.copy2(IR / "test_empty.json", OUT / "test_empty.json")
    info = {
        "arch_input": "native Depth_Color IR-box T24 from raw frames + IR T24 gray",
        "in_ch": 4, "t": T, "size": S,
        "differs_from": "cache/depth_ir_4ch_v31 T=16 s=112 IR-box; not T16 upsample",
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T, S, S, C_OUT],
        "test_shape": [nte, T, S, S, C_OUT],
        "aligned_to": "cache/ir_yolo_v4",
        "native_t24_depth": True,
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
