"""Stack Depth_Color RGB (IR-box YOLO crop) + IR luminance into 4-channel T16 s112 cache.

Aligned to ir_yolo_v4 clip order so hold_idx matches the late-fuse stack.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IR = ROOT / "cache" / "ir_yolo_v4"
DEPTH = ROOT / "cache" / "depth_color_yolo_v4_irbox"
OUT = ROOT / "cache" / "depth_ir_4ch_v31"
T, S, C_OUT = 16, 112, 4


def _keys_train(meta):
    return [(int(m["user_id"]), int(m["label"]), str(m.get("trial", ""))) for m in meta]


def _stack(depth_path: Path, ir_path: Path, n: int, out_path: Path):
    d = np.memmap(depth_path, dtype=np.uint8, mode="r", shape=(n, T, S, S, 3))
    ir = np.memmap(ir_path, dtype=np.uint8, mode="r", shape=(n, T, S, S, 3))
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(n, T, S, S, C_OUT))
    bs = 64
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        db = np.asarray(d[sl])
        ib = np.asarray(ir[sl])
        gray = ib.mean(axis=-1).round().astype(np.uint8)
        chunk = np.concatenate([db, gray[..., None]], axis=-1)
        out[sl] = chunk
        print(f"  stacked {min(i + bs, n)}/{n}", flush=True)
    out.flush()
    del d, ir, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ir_tr = json.loads((IR / "train_meta.json").read_text(encoding="utf-8"))
    dp_tr = json.loads((DEPTH / "train_meta.json").read_text(encoding="utf-8"))
    if _keys_train(ir_tr) != _keys_train(dp_tr):
        raise SystemExit("train Depth irbox keys != IR keys")
    ir_te = json.loads((IR / "test_meta.json").read_text(encoding="utf-8"))
    dp_te = json.loads((DEPTH / "test_meta.json").read_text(encoding="utf-8"))
    if [m["sample_id"] for m in ir_te] != [m["sample_id"] for m in dp_te]:
        raise SystemExit("test sample_id order mismatch")

    ntr, nte = len(ir_tr), len(ir_te)
    print(f"4ch cache n_train={ntr} n_test={nte} shape=(N,{T},{S},{S},{C_OUT})", flush=True)
    _stack(DEPTH / f"train_x_t{T}_s{S}.npy", IR / f"train_x_t{T}_s{S}.npy", ntr, OUT / f"train_x_t{T}_s{S}.npy")
    _stack(DEPTH / f"test_x_t{T}_s{S}.npy", IR / f"test_x_t{T}_s{S}.npy", nte, OUT / f"test_x_t{T}_s{S}.npy")
    shutil.copy2(IR / "train_y.npy", OUT / "train_y.npy")
    shutil.copy2(IR / "train_users.npy", OUT / "train_users.npy")
    shutil.copy2(IR / "train_meta.json", OUT / "train_meta.json")
    shutil.copy2(IR / "test_meta.json", OUT / "test_meta.json")
    shutil.copy2(IR / "test_empty.json", OUT / "test_empty.json")
    x = np.memmap(OUT / f"train_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(ntr, T, S, S, C_OUT))
    xt = np.memmap(OUT / f"test_x_t{T}_s{S}.npy", dtype=np.uint8, mode="r", shape=(nte, T, S, S, C_OUT))
    info = {
        "arch_input": "Depth_Color RGB + IR gray",
        "in_ch": C_OUT,
        "t": T,
        "size": S,
        "n_train": ntr,
        "n_test": nte,
        "train_shape": list(x.shape),
        "test_shape": list(xt.shape),
        "aligned_to": "cache/ir_yolo_v4",
        "depth_src": str(DEPTH),
        "ir_src": str(IR),
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)
    assert x.shape[-1] == 4 and xt.shape[-1] == 4
    assert xt.shape[0] == 405


if __name__ == "__main__":
    main()
