"""4ch Depth_Color (IR-box T16 upsampled) + IR T24. Different T from depth_ir_4ch_v31 (T16)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
IR = ROOT / "cache" / "ir_yolo_v4_t24"
DEPTH = ROOT / "cache" / "depth_color_yolo_v4_irbox"
OUT = ROOT / "cache" / "depth_ir_4ch_t24"
T_IR, T_D, S, C_OUT = 24, 16, 112, 4


def _keys_train(meta):
    return [(int(m["user_id"]), int(m["label"]), str(m.get("trial", ""))) for m in meta]


def _stack(n: int, ir_path: Path, depth_path: Path, out_path: Path, n_ir_t: int, n_d_t: int):
    ir = np.memmap(ir_path, dtype=np.uint8, mode="r", shape=(n, n_ir_t, S, S, 3))
    d = np.memmap(depth_path, dtype=np.uint8, mode="r", shape=(n, n_d_t, S, S, 3))
    out = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(n, n_ir_t, S, S, C_OUT))
    idx = np.linspace(0, n_d_t - 1, n_ir_t).round().astype(np.int64)
    bs = 32
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        db = np.asarray(d[sl])[:, idx]
        ib = np.asarray(ir[sl])
        gray = ib.mean(axis=-1).round().astype(np.uint8)
        out[sl] = np.concatenate([db, gray[..., None]], axis=-1)
        print(f"  stacked {min(i + bs, n)}/{n}", flush=True)
    out.flush()
    del ir, d, out


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
    print(f"4ch T24 cache n_train={ntr} n_test={nte} vs old T16 IR-box", flush=True)
    _stack(ntr, IR / f"train_x_t{T_IR}_s{S}.npy", DEPTH / f"train_x_t{T_D}_s{S}.npy",
           OUT / f"train_x_t{T_IR}_s{S}.npy", T_IR, T_D)
    _stack(nte, IR / f"test_x_t{T_IR}_s{S}.npy", DEPTH / f"test_x_t{T_D}_s{S}.npy",
           OUT / f"test_x_t{T_IR}_s{S}.npy", T_IR, T_D)
    shutil.copy2(IR / "train_y.npy", OUT / "train_y.npy")
    shutil.copy2(IR / "train_users.npy", OUT / "train_users.npy")
    shutil.copy2(IR / "train_meta.json", OUT / "train_meta.json")
    shutil.copy2(IR / "test_meta.json", OUT / "test_meta.json")
    shutil.copy2(IR / "test_empty.json", OUT / "test_empty.json")
    x = np.memmap(OUT / f"train_x_t{T_IR}_s{S}.npy", dtype=np.uint8, mode="r", shape=(ntr, T_IR, S, S, C_OUT))
    xt = np.memmap(OUT / f"test_x_t{T_IR}_s{S}.npy", dtype=np.uint8, mode="r", shape=(nte, T_IR, S, S, C_OUT))
    info = {
        "arch_input": "Depth_Color RGB IR-box T16->T24 upsample + IR T24 gray",
        "in_ch": C_OUT,
        "t": T_IR,
        "size": S,
        "differs_from": "cache/depth_ir_4ch_v31 T=16 s=112",
        "n_train": ntr,
        "n_test": nte,
        "train_shape": list(x.shape),
        "test_shape": list(xt.shape),
        "aligned_to": "cache/ir_yolo_v4",
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)
    assert x.shape == (ntr, 24, 112, 112, 4)
    assert xt.shape == (405, 24, 112, 112, 4)


if __name__ == "__main__":
    main()
