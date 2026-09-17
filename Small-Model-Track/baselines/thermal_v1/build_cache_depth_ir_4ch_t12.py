"""4ch T=12: subsample IR-box T16 Depth+IR cache. Unused T vs T8/T16/T24 members."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cache" / "depth_ir_4ch_v31"
OUT = ROOT / "cache" / "depth_ir_4ch_t12"
T12, T16, S, C = 12, 16, 112, 4
IDX = np.round(np.linspace(0, T16 - 1, T12)).astype(np.int64)


def _copy(n, src_name, dst_name, t_src):
    src = np.memmap(SRC / src_name, dtype=np.uint8, mode="r", shape=(n, t_src, S, S, C))
    dst = np.memmap(OUT / dst_name, dtype=np.uint8, mode="w+", shape=(n, T12, S, S, C))
    bs = 64
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        dst[sl] = np.asarray(src[sl])[:, IDX]
        print(f"  {dst_name} {sl.stop}/{n}", flush=True)
    dst.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "cache_meta.json").read_text(encoding="utf-8"))
    ntr, nte = int(meta["n_train"]), int(meta["n_test"])
    _copy(ntr, "train_x_t16_s112.npy", "train_x_t12_s112.npy", T16)
    _copy(nte, "test_x_t16_s112.npy", "test_x_t12_s112.npy", T16)
    for name in ("train_y.npy", "train_users.npy", "train_meta.json", "test_meta.json", "test_empty.json"):
        shutil.copy2(SRC / name, OUT / name)
    info = {
        "arch_input": "IR-box Depth RGB + IR gray, T=12 subsample of T16",
        "in_ch": 4, "t": T12, "size": S,
        "frame_idx": IDX.tolist(),
        "differs_from": "T16 IR-box; T16 non-IR-box; T24 IR-box; T24 non-IR-box; T=8 IR-box",
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T12, S, S, C],
        "test_shape": [nte, T12, S, S, C],
        "aligned_to": "cache/ir_yolo_v4",
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
