"""4ch T=14: subsample IR-box T16. Unused T vs T8/T10/T12/T16/T20/T24."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cache" / "depth_ir_4ch_v31"
OUT = ROOT / "cache" / "depth_ir_4ch_t14"
T14, T16, S, C = 14, 16, 112, 4
IDX = np.round(np.linspace(0, T16 - 1, T14)).astype(np.int64)


def _copy(n, src_name, dst_name):
    src = np.memmap(SRC / src_name, dtype=np.uint8, mode="r", shape=(n, T16, S, S, C))
    dst = np.memmap(OUT / dst_name, dtype=np.uint8, mode="w+", shape=(n, T14, S, S, C))
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
    _copy(ntr, "train_x_t16_s112.npy", "train_x_t14_s112.npy")
    _copy(nte, "test_x_t16_s112.npy", "test_x_t14_s112.npy")
    for name in ("train_y.npy", "train_users.npy", "train_meta.json", "test_meta.json", "test_empty.json"):
        shutil.copy2(SRC / name, OUT / name)
    info = {
        "arch_input": "IR-box Depth RGB + IR gray, T=14 subsample of T16",
        "in_ch": 4, "t": T14, "size": S, "frame_idx": IDX.tolist(),
        "differs_from": "T8; T10; T12; T16 IR-box; T20/T24; non-IR-box views",
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T14, S, S, C], "test_shape": [nte, T14, S, S, C],
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
