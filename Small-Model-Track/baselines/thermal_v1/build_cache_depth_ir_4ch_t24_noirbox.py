"""4ch T24 non-IR-box: Depth_Color YOLO crop T16->T24 upsample + IR T24 gray.

Different from IR-box T16, non-IR-box T16, and native T24 IR-box.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IR = ROOT / "cache" / "ir_yolo_v4_t24"
DEPTH = ROOT / "cache" / "depth_color_yolo_v4"
IRBOX = ROOT / "cache" / "depth_color_yolo_v4_irbox"
IR16 = ROOT / "cache" / "ir_yolo_v4"
OUT = ROOT / "cache" / "depth_ir_4ch_t24_noirbox"
T_IR, T_D, S, C_OUT = 24, 16, 112, 4


def _keys(meta):
    return [(int(m["user_id"]), int(m["label"]), str(m.get("trial", ""))) for m in meta]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ir_tr = json.loads((IR / "train_meta.json").read_text(encoding="utf-8"))
    dp_tr = json.loads((DEPTH / "train_meta.json").read_text(encoding="utf-8"))
    ir16_tr = json.loads((IR16 / "train_meta.json").read_text(encoding="utf-8"))
    if _keys(ir_tr) != _keys(ir16_tr):
        raise SystemExit("IR T24 keys != IR T16 keys")
    ntr, nte = len(ir_tr), len(json.loads((IR / "test_meta.json").read_text(encoding="utf-8")))
    dmap = {k: i for i, k in enumerate(_keys(dp_tr))}
    ir_keys = _keys(ir_tr)
    miss = [i for i, k in enumerate(ir_keys) if k not in dmap]
    print(f"t24 noirbox 4ch n_train={ntr} n_test={nte} fallback_irbox={len(miss)}", flush=True)
    idx = np.linspace(0, T_D - 1, T_IR).round().astype(np.int64)
    ir_x = np.memmap(IR / f"train_x_t{T_IR}_s{S}.npy", dtype=np.uint8, mode="r", shape=(ntr, T_IR, S, S, 3))
    d_x = np.memmap(DEPTH / f"train_x_t{T_D}_s{S}.npy", dtype=np.uint8, mode="r", shape=(len(dp_tr), T_D, S, S, 3))
    fb_x = np.memmap(IRBOX / f"train_x_t{T_D}_s{S}.npy", dtype=np.uint8, mode="r", shape=(ntr, T_D, S, S, 3))
    out = np.memmap(OUT / f"train_x_t{T_IR}_s{S}.npy", dtype=np.uint8, mode="w+", shape=(ntr, T_IR, S, S, C_OUT))
    bs = 32
    for i in range(0, ntr, bs):
        sl = slice(i, min(i + bs, ntr))
        chunk = np.zeros((sl.stop - sl.start, T_IR, S, S, C_OUT), np.uint8)
        gray = np.asarray(ir_x[sl]).mean(axis=-1).round().astype(np.uint8)
        for j, gi in enumerate(range(sl.start, sl.stop)):
            k = ir_keys[gi]
            src = d_x[dmap[k]] if k in dmap else fb_x[gi]
            chunk[j, ..., :3] = np.asarray(src)[idx]
            chunk[j, ..., 3] = gray[j]
        out[sl] = chunk
        print(f"  train {sl.stop}/{ntr}", flush=True)
    out.flush()
    ir_te = json.loads((IR / "test_meta.json").read_text(encoding="utf-8"))
    dp_te = json.loads((DEPTH / "test_meta.json").read_text(encoding="utf-8"))
    if [m["sample_id"] for m in ir_te] != [m["sample_id"] for m in dp_te]:
        raise SystemExit("test sample_id mismatch")
    ir_t = np.memmap(IR / f"test_x_t{T_IR}_s{S}.npy", dtype=np.uint8, mode="r", shape=(nte, T_IR, S, S, 3))
    d_t = np.memmap(DEPTH / f"test_x_t{T_D}_s{S}.npy", dtype=np.uint8, mode="r", shape=(nte, T_D, S, S, 3))
    out_t = np.memmap(OUT / f"test_x_t{T_IR}_s{S}.npy", dtype=np.uint8, mode="w+", shape=(nte, T_IR, S, S, C_OUT))
    for i in range(0, nte, bs):
        sl = slice(i, min(i + bs, nte))
        db = np.asarray(d_t[sl])[:, idx]
        gray = np.asarray(ir_t[sl]).mean(axis=-1).round().astype(np.uint8)
        out_t[sl] = np.concatenate([db, gray[..., None]], axis=-1)
        print(f"  test {sl.stop}/{nte}", flush=True)
    out_t.flush()
    shutil.copy2(IR / "train_y.npy", OUT / "train_y.npy")
    shutil.copy2(IR / "train_users.npy", OUT / "train_users.npy")
    shutil.copy2(IR / "train_meta.json", OUT / "train_meta.json")
    shutil.copy2(IR / "test_meta.json", OUT / "test_meta.json")
    shutil.copy2(IR / "test_empty.json", OUT / "test_empty.json")
    info = {
        "arch_input": "non-IR-box Depth T16->T24 upsample + IR T24 gray",
        "in_ch": 4, "t": T_IR, "size": S,
        "differs_from": "IR-box T16; non-IR-box T16; native T24 IR-box",
        "crop": "non-irbox T24",
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T_IR, S, S, C_OUT],
        "test_shape": [nte, T_IR, S, S, C_OUT],
        "fallback_irbox_n": len(miss),
        "aligned_to": "cache/ir_yolo_v4",
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
