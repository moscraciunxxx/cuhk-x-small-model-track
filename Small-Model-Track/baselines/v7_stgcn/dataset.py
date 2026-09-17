"""Dataset helpers for v7 ST-GCN — reuse skeleton_imu_v2 caches."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_V2_DS = Path(__file__).resolve().parent.parent / "skeleton_imu_v2" / "dataset.py"


def _load_v2_dataset():
    spec = importlib.util.spec_from_file_location("skel_imu_v2_dataset", _V2_DS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_v2 = _load_v2_dataset()

COORDS = _v2.COORDS
DEFAULT_HOLD_OUT_USERS = _v2.DEFAULT_HOLD_OUT_USERS
DEFAULT_T = _v2.DEFAULT_T
IMU_DIM = _v2.IMU_DIM
NUM_JOINTS = _v2.NUM_JOINTS
SKEL_DIM = _v2.SKEL_DIM
CachedDualDataset = _v2.CachedDualDataset
CachedSkelDataset = _v2.CachedSkelDataset
load_skel_test_cache = _v2.load_skel_test_cache
load_skel_train_cache = _v2.load_skel_train_cache


def load_imu_caches(cache_dir: Path):
    import numpy as np

    cache_dir = Path(cache_dir)
    imu_train = np.load(cache_dir / "imu_train.npz", allow_pickle=False)
    X_imu = imu_train["X"]
    has_imu = imu_train["has_imu"].astype(bool)
    imu_test = np.load(cache_dir / "imu_test.npz", allow_pickle=False)
    X_imu_te = imu_test["X"]
    has_imu_te = imu_test["has_imu"].astype(bool) if "has_imu" in imu_test.files else None
    return X_imu, has_imu, X_imu_te, has_imu_te
