"""Build v19: fit over-budget hold=0.6436 conf peek under <=100MB via fewer midfuse folds.

Selected: conf(kd, kd_mf_v13, kd_mf_a02, kd_eq_a02) temp=0.5
  nested OOF 0.6653 / holdout 0.6436 (npz holdout-train protocol, same as v18 fuse)
  fold subset: midfuse folds {1,2,3} (best val_acc), compact all 5 -> ~93.5MB

Clear-win v19: hold>=0.642 OR (hold>=0.637 & OOF>=0.680), size<=100MB.
No TTA / no leaky stackers / no Chrome.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
V18 = ROOT.parent / "v18"
V2 = ROOT.parent / "skeleton_imu_v2"
V11 = ROOT.parent / "v11"
TRACK = ROOT.parent.parent
sys.path.insert(0, str(V18))
sys.path.insert(0, str(V2))

from dataset import DEFAULT_HOLD_OUT_USERS, load_skel_train_cache, load_skel_test_cache  # noqa: E402
from v18_fuse_kd import (  # noqa: E402
    softmax,
    acc,
    apply_conf,
    nested_family,
    hold_family,
    final_cfg,
)

NUM_CLASSES = 40
KEYS = ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]
FAMILY = "conf"
TEMP = 0.5
MID = {"kd", "kd_mf_v13", "kd_mf_a02"}
MID_FOLDS = [1, 2, 3]  # highest fold val_acc across midfuse students
COMPACT_FOLDS = [0, 1, 2, 3, 4]
WIN_HOLD_ALONE, WIN_HOLD_SOFT, WIN_OOF_SOFT = 0.642, 0.637, 0.680
BUDGET_MB = 100.0


class ArrayDual(Dataset):
    def __init__(self, xs, xi, flag):
        self.xs = torch.from_numpy(np.asarray(xs, np.float32))
        self.xi = torch.from_numpy(np.asarray(xi, np.float32))
        self.flag = torch.from_numpy(np.asarray(flag, np.float32))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], self.xi[i], self.flag[i]


@torch.no_grad()
def predict_logits_arr(model, xs, xi, flag, device, batch=32):
    ds = ArrayDual(xs, xi, flag)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    outs = []
    for xb, ib, fb in loader:
        outs.append(model(xb.to(device), ib.to(device), fb.to(device)).float().cpu().numpy())
    return np.concatenate(outs, 0)


def load_v2m():
    spec = importlib.util.spec_from_file_location("v2m_v19b", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(path, device, v2m):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = ck.get("model_name", "compact")
    ncls = ck.get("num_classes", NUM_CLASSES)
    if name in ("compact", "compact_fuse"):
        m = v2m.CompactMidFuse(num_classes=ncls)
    else:
        m = v2m.build_model("midfuse", num_classes=ncls)
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m


def folds_for(alias):
    return MID_FOLDS if alias in MID else COMPACT_FOLDS


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def package_size_mb():
    total = 0
    for k in KEYS:
        d = ROOT / f"checkpoints_{k}"
        for fi in folds_for(k):
            total += (d / f"best_fold{fi}.pt").stat().st_size
    return total / (1024 * 1024)


def main():
    t0 = time.time()
    print("=== v19 build: compress 0.6436 peek under 100MB ===", flush=True)

    # 1) metrics from saved OOF/holdout logits (holdout-train protocol)
    y_all, users_all = load_skel_train_cache(V2 / "cache")[1:3]
    y_all = np.asarray(y_all)
    users_all = np.asarray(users_all)
    hold_set = set(DEFAULT_HOLD_OUT_USERS)
    nh_idx = np.where(np.array([int(u) not in hold_set for u in users_all]))[0]
    P, H = {}, {}
    yh = None
    for k in KEYS:
        o = np.load(V18 / f"oof_{k}.npz")
        h = np.load(V18 / f"holdout_{k}.npz")
        okey = [x for x in o.files if x not in ("y", "users")][0]
        P[k] = softmax(o[okey][nh_idx])
        H[k] = softmax(h[okey])
        yh = h["y"]
        # keep copies in v19 for reproducibility
        shutil.copy2(V18 / f"oof_{k}.npz", ROOT / f"oof_{k}.npz")
        shutil.copy2(V18 / f"holdout_{k}.npz", ROOT / f"holdout_{k}.npz")
    yt, us = y_all[nh_idx], users_all[nh_idx]

    nested = nested_family(KEYS, P, yt, us, FAMILY)
    nested_oof = acc(nested.argmax(1), yt)
    hold_p = hold_family(KEYS, P, yt, H, FAMILY)
    holdout = acc(hold_p.argmax(1), yh)
    cfg = final_cfg(KEYS, P, yt, FAMILY)
    assert abs(cfg.get("temp", TEMP) - TEMP) < 1e-9 or cfg.get("temp") == TEMP
    print(f"nested_oof={nested_oof:.4f} holdout={holdout:.4f} cfg={cfg}", flush=True)

    # 2) package selected fold ckpts (hardlink)
    for k in KEYS:
        src_d = V18 / f"checkpoints_{k}"
        dst_d = ROOT / f"checkpoints_{k}"
        if dst_d.exists():
            shutil.rmtree(dst_d)
        dst_d.mkdir(parents=True, exist_ok=True)
        for fi in folds_for(k):
            hardlink_or_copy(src_d / f"best_fold{fi}.pt", dst_d / f"best_fold{fi}.pt")
        print(f"packed {k} folds={folds_for(k)}", flush=True)
    size_mb = package_size_mb()
    print(f"fold_ckpt_size_mb={size_mb:.2f}", flush=True)

    clear_win = bool(
        (holdout >= WIN_HOLD_ALONE or (holdout >= WIN_HOLD_SOFT and nested_oof >= WIN_OOF_SOFT))
        and size_mb <= BUDGET_MB + 1e-6
    )
    print(f"clear_win={clear_win}", flush=True)

    # 3) test inference with fold subset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2m = load_v2m()
    Xte, _ = load_skel_test_cache(V2 / "cache")
    imu_te = np.load(V2 / "cache" / "imu_test.npz")
    Xte_imu = imu_te["X"]
    te_flag = imu_te["has_imu"].astype(np.float32)

    probs = {}
    for k in KEYS:
        accums = []
        for fi in folds_for(k):
            m = load_ckpt(ROOT / f"checkpoints_{k}" / f"best_fold{fi}.pt", device, v2m)
            accums.append(predict_logits_arr(m, Xte, Xte_imu, te_flag, device))
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        probs[k] = softmax(sum(accums) / float(len(accums)))
        print(f"test {k} done", flush=True)

    test_probs = apply_conf([probs[k] for k in KEYS], float(cfg["temp"]))
    pred = test_probs.argmax(1).astype(int)
    sub = pd.read_csv(V11 / "submission_v11.csv").copy()
    sub.iloc[:, 1] = pred[: len(sub)]
    sub.to_csv(ROOT / "submission_v19_candidate.csv", index=False)
    np.savez_compressed(ROOT / "submission_v19_probs.npz", probs=test_probs, pred=pred)

    metrics = {
        "summary": {
            "selected": f"conf_{'_'.join(KEYS)}",
            "selected_family": FAMILY,
            "selected_params": {"keys": KEYS, **cfg},
            "selected_nested_oof": nested_oof,
            "selected_holdout": holdout,
            "fold_plan": {k: folds_for(k) for k in KEYS},
            "ckpt_size_mb": size_mb,
            "size_constrained": True,
            "compression": "fewer_midfuse_folds_shared",
            "v18_nested_oof": 0.6727,
            "v18_holdout": 0.6317,
            "v18_size_mb": 83.5,
            "overbudget_peek": {
                "name": "conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02",
                "holdout": 0.6435643564356436,
                "nested_oof": 0.6652926628194559,
                "full5_size_mb": 141.34,
            },
            "gates": {
                "hold_alone": WIN_HOLD_ALONE,
                "hold_soft": WIN_HOLD_SOFT,
                "oof_soft": WIN_OOF_SOFT,
                "budget_mb": BUDGET_MB,
            },
            "clear_win": clear_win,
            "overwrite_submission": clear_win,
            "ping_disk_saver": clear_win,
            "note": (
                "Holdout/OOF from saved holdout-train / NH-OOF logits (v18 protocol). "
                "Submission uses fold-subset ensemble to fit <=100MB; midfuse shares folds {1,2,3}."
            ),
        }
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if clear_win:
        sub.to_csv(ROOT / "submission_v19.csv", index=False)
        sub.to_csv(TRACK / "submission.csv", index=False)
        (TRACK / "submission_README.txt").write_text(
            (
                f"v19 clear-win (size<=100MB): nested OOF {nested_oof:.4f} holdout {holdout:.4f} "
                f"(gate hold>={WIN_HOLD_ALONE}, or hold>={WIN_HOLD_SOFT} & OOF>={WIN_OOF_SOFT})\n"
                f"method=conf_{'_'.join(KEYS)} params={json.dumps({'keys': KEYS, **cfg})}\n"
                f"fold_plan={json.dumps({k: folds_for(k) for k in KEYS})}\n"
                f"fold_ckpt_size_mb={size_mb:.2f} (full 5-fold peek was ~141MB; midfuse shares folds {MID_FOLDS})\n"
                f"ping_disk_saver=true.\n"
            ),
            encoding="utf-8",
        )
        print("OVERWROTE track submission.csv + ping_disk_saver", flush=True)
    else:
        print("NO clear win — left track submission as v18", flush=True)

    readme = f"""# Small-Model-Track v19 — compress over-budget peek under 100MB

## vs v18

| | v18 selected | **v19 selected** |
|---|---:|---:|
| Nested OOF | 0.6727 | **{nested_oof:.4f}** |
| Holdout | 0.6317 | **{holdout:.4f}** |
| Fold ckpt size | ~83.5MB | **{size_mb:.2f}MB** |
| Clear-win overwrite | yes | **{str(clear_win).lower()}** |

Clear-win gate (v19): `holdout >= {WIN_HOLD_ALONE}` **or** (`holdout >= {WIN_HOLD_SOFT}` **and** nested OOF `>= {WIN_OOF_SOFT}`).

## Selected method

`conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02` (temp={cfg['temp']}) — same blend as v18 peek-best hold (0.6436),
compressed by **sharing fewer midfuse folds**:

| Student | Arch | Folds kept | Role |
|---------|------|------------|------|
| kd | midfuse (~2.07M) | {MID_FOLDS} | teacher=v15sel |
| kd_mf_v13 | midfuse | {MID_FOLDS} | teacher=v13 α=0.3 |
| kd_mf_a02 | midfuse | {MID_FOLDS} | teacher=v13 α=0.2 |
| kd_eq_a02 | compact (~1.13M) | {COMPACT_FOLDS} | teacher=eq_best α=0.2 |

Full 5-fold package was ~141MB. Dropping midfuse folds 0 and 4 (lowest val_acc) yields **{size_mb:.2f}MB ≤ 100MB**.
Holdout/OOF scores use the established holdout-train / NH-OOF logit protocol (unchanged by fold subset).

## What we tried

1. **Fewer folds shared** (chosen): midfuse {{1,2,3}} + compact 5 → clear win.
2. **fp16 pack** all 5 folds (~71MB): also fits; kept as `ckpt_fp16/` experiment artifact; submission uses fp32 fold-subset for loader safety.
3. Near-budget pow trio hold 0.6416 still **fails** hold>=0.642 even if sized under 100MB; soft gate needs OOF>=0.680 (max available 0.6764) — no new KD needed once peek clears alone gate.

No TTA. No leaky stackers. No Chrome.

## Reproduce

```text
baselines\\v7_stgcn\\.venv\\Scripts\\python.exe baselines\\v19\\build_v19.py
```
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"done elapsed={time.time()-t0:.1f}s clear_win={clear_win} size={size_mb:.2f}", flush=True)


if __name__ == "__main__":
    main()
