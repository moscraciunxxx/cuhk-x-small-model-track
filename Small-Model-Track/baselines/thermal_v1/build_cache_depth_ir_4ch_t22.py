"""4ch T=22: subsample native T24 IR-box. Unused T vs T8/10/12/14/16/18/20/24; not in v66 mix."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cache" / "depth_ir_4ch_t24_native"
OUT = ROOT / "cache" / "depth_ir_4ch_t22"
T22, T24, S, C = 22, 24, 112, 4
IDX = np.round(np.linspace(0, T24 - 1, T22)).astype(np.int64)


def _copy(n, src_name, dst_name):
    src = np.memmap(SRC / src_name, dtype=np.uint8, mode="r", shape=(n, T24, S, S, C))
    dst = np.memmap(OUT / dst_name, dtype=np.uint8, mode="w+", shape=(n, T22, S, S, C))
    bs = 32
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        dst[sl] = np.asarray(src[sl])[:, IDX]
        print(f"  {dst_name} {sl.stop}/{n}", flush=True)
    dst.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SRC / "cache_meta.json").read_text(encoding="utf-8"))
    ntr, nte = int(meta["n_train"]), int(meta["n_test"])
    _copy(ntr, "train_x_t24_s112.npy", "train_x_t22_s112.npy")
    _copy(nte, "test_x_t24_s112.npy", "test_x_t22_s112.npy")
    for name in ("train_y.npy", "train_users.npy", "train_meta.json", "test_meta.json", "test_empty.json"):
        shutil.copy2(SRC / name, OUT / name)
    info = {
        "arch_input": "IR-box Depth RGB + IR gray, T=22 subsample of native T24",
        "in_ch": 4, "t": T22, "size": S, "frame_idx": IDX.tolist(),
        "not_in_v66_mix": True,
        "not_t18": True,
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T22, S, S, C], "test_shape": [nte, T22, S, S, C],
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
