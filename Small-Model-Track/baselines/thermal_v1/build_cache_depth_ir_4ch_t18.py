"""4ch T=18: subsample native T24 IR-box. Unused T vs T8/10/12/14/16/20/24 (v66 mix uses T10/12/16/24)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cache" / "depth_ir_4ch_t24_native"
OUT = ROOT / "cache" / "depth_ir_4ch_t18"
T18, T24, S, C = 18, 24, 112, 4
IDX = np.round(np.linspace(0, T24 - 1, T18)).astype(np.int64)


def _copy(n, src_name, dst_name):
    src = np.memmap(SRC / src_name, dtype=np.uint8, mode="r", shape=(n, T24, S, S, C))
    dst = np.memmap(OUT / dst_name, dtype=np.uint8, mode="w+", shape=(n, T18, S, S, C))
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
    _copy(ntr, "train_x_t24_s112.npy", "train_x_t18_s112.npy")
    _copy(nte, "test_x_t24_s112.npy", "test_x_t18_s112.npy")
    for name in ("train_y.npy", "train_users.npy", "train_meta.json", "test_meta.json", "test_empty.json"):
        shutil.copy2(SRC / name, OUT / name)
    info = {
        "arch_input": "IR-box Depth RGB + IR gray, T=18 subsample of native T24",
        "in_ch": 4, "t": T18, "size": S, "frame_idx": IDX.tolist(),
        "not_in_v66_mix": "v66 uses T16 IRbox + T16 noirbox + T24 IRbox + T12 IRbox + T10 IRbox",
        "n_train": ntr, "n_test": nte,
        "train_shape": [ntr, T18, S, S, C], "test_shape": [nte, T18, S, S, C],
    }
    (OUT / "cache_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
