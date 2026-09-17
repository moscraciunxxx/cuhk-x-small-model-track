"""Structural check: member is R(2+1)D-34 with 4-channel Depth+IR input, not r2plus1d_18."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from model_r2p1d34 import IN_CH, LAYERS, assert_r2p1d34_4ch, build_r2p1d34_4ch

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "depth_ir_4ch_v31"


def main():
    m = build_r2p1d34_4ch(pretrained=False, progress=False)
    ident = assert_r2p1d34_4ch(m)
    assert ident["arch"] == "r2plus1d_34"
    assert ident["in_ch"] == 4
    assert tuple(ident["layers"]) == LAYERS
    assert ident["nparams"] > 50_000_000
    meta = json.loads((CACHE / "cache_meta.json").read_text(encoding="utf-8"))
    assert int(meta["in_ch"]) == 4
    assert meta["train_shape"][-1] == 4
    assert meta["test_shape"][0] == 405
    x = np.memmap(CACHE / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=tuple(meta["train_shape"]))
    assert x.shape[-1] == 4
    print(f"arch={ident['arch']} in_ch={ident['in_ch']} layers={ident['layers']} nparams={ident['nparams']}", flush=True)
    print(f"cache_in_ch={meta['in_ch']} train_shape={meta['train_shape']} test_shape={meta['test_shape']}", flush=True)
    print("not_r2plus1d_18=1 pass=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
