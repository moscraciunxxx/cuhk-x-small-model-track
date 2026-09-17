# Skeleton-only HAR baseline (v1) — CUHK-X Small Model Track

Cross-subject skeleton sequence classifier (Conv1D / GRU / Tiny Transformer).
Model size: a few MB. Metric: accuracy. No large pretrained nets.

## Layout

- `dataset.py` — load train/test skeleton JSON → fixed `T×51` tensors, GroupKFold by user
- `model.py` — `conv1d` (default), `gru`, `transformer`
- `train.py` — hold-out CV (users 8,9,24), optional GroupKFold / full retrain
- `infer.py` — writes `submission_skeleton_v1.csv`
- `checkpoints/` — `best.pt` and fold tags
- `metrics.json` — train metrics

## Data paths (junction)

- Train: `D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\Skeleton\{id}_{Action}\user{N}\{trial}\predictions\*.json`
- Test: `D:\CUHK-X\Small-Model-Track\Testing\data\small_model_track_test\SM_test_XXXX\Skeleton\predictions\*.json`
- Sample: `D:\CUHK-X\Small-Model-Track\Testing\test_file\sample_submission.csv`

JSON schema: list of persons; each has `keypoints` (17×3) and `keypoint_scores` (17).

## Setup (uv)

```powershell
cd D:\CUHK-X\Small-Model-Track\baselines\skeleton_v1
uv venv .venv
.\.venv\Scripts\Activate.ps1
# CUDA torch (adjust cu version if needed)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
```

If the CUDA index line fails, use:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy pandas tqdm scikit-learn
```

## Smoke test

```powershell
python -c "from dataset import discover_train_samples; from pathlib import Path; s=discover_train_samples(Path(r'D:\CUHK-X\Small-Model-Track\Training\data\HAR\data\Skeleton')); print(len(s), s[0])"
python train.py --mode smoke --epochs 2 --batch-size 64
```

## Train (overnight hold-out + all-train)

```powershell
python train.py --mode full --epochs 40 --model conv1d --batch-size 64 --seed 42
```

Hold-out users for local CV: **8, 9, 24**. Never use test-set users.

GroupKFold:

```powershell
python train.py --mode cv --cv-splits 3 --epochs 25
```

## Infer

```powershell
python infer.py --ckpt checkpoints\best.pt --out submission_skeleton_v1.csv
```

## Notes

- Fixed length `T=64` via temporal resampling; root-centered + std-scaled.
- Reproducible seeds (`--seed 42`).
- Constraint: model ≤100MB (this baseline is ≪).

## Cache (recommended)

JSON-per-frame loading is slow. Precompute once:

```powershell
python build_cache.py
```

Then `train.py` / `infer.py` auto-use `cache/train.npz` and `cache/test.npz`.

