# Skeleton + IMU HAR baseline (v2) — CUHK-X Small Model Track

Beats v1 holdout **0.479** with a ≤100MB CNN mid-fuse model.

## Results (primary)

| Split | Metric |
|-------|--------|
| **GroupKFold 5-fold mean±std** | **0.505 ± 0.058** |
| Holdout users {8,9,24} | **0.537** |
| v1 holdout (Conv1D) | 0.479 |
| Model | MidFuseNet (skel Conv + IMU Conv), 2,072,072 params, **~8.4 MB** ckpt |
| Submission | `submission_skeleton_imu_v2.csv` (405 rows) |
| Ensemble (optional) | `submission_skeleton_imu_v2_ensemble.csv` (softmax avg of 5 folds) |

Per-fold val_acc: 0.455, 0.572, 0.547, 0.533, 0.419

### Ablations

| Config | CV mean±std | Holdout |
|--------|-------------|---------|
| midfuse + balanced sampler (v2a) | 0.462 ± 0.055 | 0.485 |
| **midfuse, no balanced sampler, ls=0.05 (v2b)** | **0.505 ± 0.058** | **0.537** |
| deepconv skeleton-only | 0.478 ± 0.042 | 0.515 |

Weighted CE + light augment kept; **WeightedRandomSampler hurt** cross-subject CV.

## What’s new vs skeleton_v1

1. Stronger residual temporal Conv branches + MidFuse IMU
2. Proper **GroupKFold by user** (5 folds, users 1–9 & 16–24)
3. IMU mid-fuse: 5 devices × (acc+gyro) = 30 dims, resampled to T=64; gated when missing (28/2931 train trials)
4. Class-weighted CE + label smoothing; optional balanced sampler
5. Temporal augment (noise / shift / time-mask)

## Layout

- `dataset.py`, `model.py`, `build_cache.py`, `train.py`, `infer.py`, `ensemble_infer.py`
- `cache/` — skel + imu npz
- `checkpoints/` — primary `best.pt` + fold ckpts
- `checkpoints_midfuse_v2b/`, `checkpoints_deepconv/` — ablation runs
- `metrics.json` — primary CV metrics
- `.venv` — junction to `skeleton_v1\.venv` (torch 2.6+cu124)

## Setup

```powershell
cd D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2
.\.venv\Scripts\Activate.ps1
```

## Rebuild IMU cache

```powershell
python build_cache.py
```

## Train (winning recipe)

```powershell
python train.py --mode cv --cv-splits 5 --epochs 50 --model midfuse `
  --batch-size 40 --seed 42 --patience 15 --no-balanced-sampler `
  --label-smoothing 0.05 --lr 8e-4
```

## Infer

```powershell
python infer.py --ckpt checkpoints\best.pt --out submission_skeleton_imu_v2.csv
python ensemble_infer.py --ckpt-dir checkpoints --out submission_skeleton_imu_v2_ensemble.csv
```

## Data notes

- IMU CSV: UTF-8-BOM, CN headers; devices WTLA/WTRA/WTC/WTLL/WTRL
- Train IMU 2903 / skel 2931 aligned by (action, user, trial)
- Test IMU present for all 405 clips
