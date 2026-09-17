"""Build IMU caches aligned to skeleton train/test caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import (
    DEFAULT_T,
    IMU_DIM,
    discover_imu_trials,
    discover_test_imu,
    load_imu_trial,
    load_skel_test_cache,
    load_skel_train_cache,
)

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument(
        "--imu-root",
        type=str,
        default=r"D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\IMU",
    )
    p.add_argument(
        "--test-root",
        type=str,
        default=r"D:\CUHK-X\Small-Model-Track\Testing\data\small_model_track_test",
    )
    p.add_argument("--cache-dir", type=str, default=str(ROOT / "cache"))
    args = p.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    X_skel, y, users, meta = load_skel_train_cache(cache)
    n = len(meta)
    assert n == len(y)
    print(f"skel train n={n} T={X_skel.shape[1]} F={X_skel.shape[2]}")

    imu_map = discover_imu_trials(Path(args.imu_root))
    print(f"discovered IMU trials={len(imu_map)}")

    X_imu = np.zeros((n, args.T, IMU_DIM), dtype=np.float32)
    has_imu = np.zeros((n,), dtype=np.bool_)
    missing = 0
    for i, m in enumerate(tqdm(meta, desc="cache-imu-train")):
        key = (m["action_name"], int(m["user_id"]), m["trial"])
        trial_dir = imu_map.get(key)
        if trial_dir is None:
            missing += 1
            continue
        X_imu[i] = load_imu_trial(Path(trial_dir), T=args.T)
        has_imu[i] = True
    np.savez_compressed(cache / "imu_train.npz", X=X_imu, has_imu=has_imu.astype(np.uint8))
    print(f"Wrote imu_train.npz shape={X_imu.shape} has_imu={has_imu.sum()} missing={missing}")

    # test
    X_test, paths = load_skel_test_cache(cache)
    test_imu = discover_test_imu(Path(args.test_root))
    Xt = np.zeros((len(paths), args.T, IMU_DIM), dtype=np.float32)
    has_t = np.zeros((len(paths),), dtype=np.bool_)
    miss_t = 0
    for i, path in enumerate(tqdm(paths, desc="cache-imu-test")):
        # path like small_model_track_test/SM_test_0001/
        clip = path.rstrip("/").split("/")[-1]
        imu_dir = test_imu.get(clip)
        if imu_dir is None:
            miss_t += 1
            continue
        Xt[i] = load_imu_trial(Path(imu_dir), T=args.T)
        has_t[i] = True
    np.savez_compressed(
        cache / "imu_test.npz",
        X=Xt,
        has_imu=has_t.astype(np.uint8),
        paths=np.array(paths, dtype=object),
    )
    print(f"Wrote imu_test.npz shape={Xt.shape} has_imu={has_t.sum()} missing={miss_t}")


if __name__ == "__main__":
    main()
