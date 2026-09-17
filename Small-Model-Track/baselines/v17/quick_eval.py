"""Quick nested/holdout eval for v17 KD blends (no test inference)."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import sys
sys.path.insert(0, r"baselines\skeleton_imu_v2")
from dataset import DEFAULT_HOLD_OUT_USERS

ROOT = Path(r"baselines\v17")
NUM = 40
TEMPS = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)

def soft(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / np.maximum(e.sum(1, keepdims=True), 1e-12)).astype(np.float32)

def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())

def apply_eq(ps):
    return (sum(ps) / float(len(ps))).astype(np.float32)

def apply_conf(ps, temp):
    confs = [np.exp(pr.max(1, keepdims=True) / temp) for pr in ps]
    w = np.concatenate(confs, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return sum(w[:, i:i+1] * ps[i] for i in range(len(ps))).astype(np.float32)

def fit_conf(ps, y):
    best = None
    for t in TEMPS:
        a = acc(apply_conf(ps, t).argmax(1), y)
        if best is None or a > best[0]:
            best = (a, t)
    return best[1]

def nested(keys, P, yt, us, fam):
    n = len(yt)
    out = np.zeros((n, NUM), np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(np.arange(n), yt, us):
        if fam == "eq":
            out[va] = apply_eq([P[k][va] for k in keys])
        else:
            t = fit_conf([P[k][tr] for k in keys], yt[tr])
            out[va] = apply_conf([P[k][va] for k in keys], t)
    return out

# load
aliases = {
    "kd": ("oof_kd.npz", "holdout_kd.npz", ["kd"]),
    "kd2": ("oof_kd2.npz", "holdout_kd2.npz", ["kd2", "kd"]),
    "kd_c": ("oof_kd_c.npz", "holdout_kd_c.npz", ["kd_c", "kd"]),
    "kd_alt": ("oof_kd_alt.npz", "holdout_kd_alt.npz", ["kd_alt", "kd"]),
    "kd3": ("oof_kd3.npz", "holdout_kd3.npz", ["kd3", "kd"]),
    "kd_mf_v13": ("oof_kd_mf_v13.npz", "holdout_kd_mf_v13.npz", ["kd_mf_v13", "kd"]),
    "kd_a02": ("oof_kd_a02.npz", "holdout_kd_a02.npz", ["kd_a02", "kd"]),
}
P = {}; HP = {}; y = None; users = None; hy = None
for alias, (of, hf, keys) in aliases.items():
    op, hp = ROOT / of, ROOT / hf
    if not op.exists() or not hp.exists():
        continue
    d = np.load(op)
    k = next(x for x in keys + ["student", "compact"] if x in d.files)
    P[alias] = soft(d[k])
    if y is None:
        y = d["y"].astype(np.int64); users = d["users"].astype(np.int64)
    d2 = np.load(hp)
    k2 = next(x for x in keys + ["student", "compact"] if x in d2.files)
    HP[alias] = soft(d2[k2])
    if hy is None:
        hy = d2["y"].astype(np.int64)

hold = set(int(u) for u in DEFAULT_HOLD_OUT_USERS)
nh = np.array([int(u) not in hold for u in users])
Pn = {k: v[nh] for k, v in P.items()}
yt, us = y[nh], users[nh]
pool = list(P.keys())
print("branches", pool)

methods = []
for r in range(1, len(pool) + 1):
    for combo in combinations(pool, r):
        for fam in ("eq", "conf") if r > 1 else ("eq",):
            name = ("solo_" + combo[0]) if r == 1 else f"{fam}_{'_'.join(combo)}"
            o = nested(combo, Pn, yt, us, fam)
            if fam == "eq":
                h = apply_eq([HP[k] for k in combo])
            else:
                t = fit_conf([Pn[k] for k in combo], yt)
                h = apply_conf([HP[k] for k in combo], t)
            methods.append({
                "name": name,
                "nested_oof": acc(o.argmax(1), yt),
                "holdout": acc(h.argmax(1), hy),
                "keys": list(combo),
                "family": fam,
            })

methods.sort(key=lambda m: (-m["nested_oof"], -m["holdout"]))
WIN_HOLD_ALONE, WIN_HOLD_SOFT, WIN_OOF_SOFT = 0.624, 0.619, 0.660
best = methods[0]
clear = best["holdout"] >= WIN_HOLD_ALONE or (best["holdout"] >= WIN_HOLD_SOFT and best["nested_oof"] >= WIN_OOF_SOFT)
print("SELECTED", best["name"], "nested", round(best["nested_oof"],4), "hold", round(best["holdout"],4), "clear", clear)
print("TOP15:")
for m in methods[:15]:
    cw = m["holdout"] >= WIN_HOLD_ALONE or (m["holdout"] >= WIN_HOLD_SOFT and m["nested_oof"] >= WIN_OOF_SOFT)
    print(f"  {m['name']:40s} nested={m['nested_oof']:.4f} hold={m['holdout']:.4f} clear={cw}")
out = {"selected": best, "clear_win": clear, "top20": methods[:20], "n": len(methods), "branches": pool}
(ROOT / "quick_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote quick_eval.json")
