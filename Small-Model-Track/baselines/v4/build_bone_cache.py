"""Precompute bone feature caches."""
from pathlib import Path
import numpy as np
from bones import precompute_bone_cache
from dataset import load_skel_train_cache, load_skel_test_cache

c = Path("cache")
Xs, y, u, m = load_skel_train_cache(c)
p = c / "bone_train.npz"
if not p.exists():
    print("computing train bone...", flush=True)
    X = precompute_bone_cache(Xs)
    np.savez_compressed(p, X=X)
    print("wrote", p, X.shape, flush=True)
else:
    print("exists", p, np.load(p)["X"].shape, flush=True)

Xt, paths = load_skel_test_cache(c)
pt = c / "bone_test.npz"
if not pt.exists():
    print("computing test bone...", flush=True)
    X = precompute_bone_cache(Xt)
    np.savez_compressed(pt, X=X)
    print("wrote", pt, X.shape, flush=True)
else:
    print("exists", pt, np.load(pt)["X"].shape, flush=True)
