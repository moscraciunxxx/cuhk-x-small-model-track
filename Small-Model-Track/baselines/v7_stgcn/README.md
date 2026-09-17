# CUHK-X Small Model Track — baselines/v7_stgcn

Quiet MSI run. Small **ST-GCN** on 17-joint COCO/body17 skeleton + optional IMU mid-fuse; late-fuse with MidFuse v2b.

## Headline

| Item | Value |
|------|------:|
| MidFuse v2b holdout (baseline) | **0.5366** |
| ST-GCN+IMU alone holdout (best) | **0.4792** |
| ST-GCN mean CV (5 GroupKFold) | **0.457 +/- 0.054** |
| Pred agreement vs MidFuse (holdout) | **0.525** |
| Late-fuse holdout (OOF-selected α=0.5) | **0.5426** (+0.006) |
| Late-fuse OOF (non-holdout) | **0.5235** (+0.022 vs v3 0.5015) |
| Clear holdout ≥0.547? | **No** |
| Clearly better OOF? | **Yes** |
| **ping_disk_saver** | **true** |
| Track `submission.csv` | **updated** to late-fuse |

## Joint order / adjacency

Train JSON has 17×3 keypoints. Same COCO-like order as `v4/bones.py`:

0 nose, 1 L_eye, 2 R_eye, 3 L_ear, 4 R_ear, 5 L_sho, 6 R_sho, 7 L_elb, 8 R_elb, 9 L_wri, 10 R_wri, 11 L_hip, 12 R_hip, 13 L_kne, 14 R_kne, 15 L_ank, 16 R_ank.

`graph.py` builds hop-normalized adjacency for ST-GCN.

## Model

- Input: joint xyz + frame velocity (C=6), T=64, V=17
- ST-GCN blocks (channels 64→64→128→128→256) + IMU Conv1D mid-fuse (~1.45M params / ≪50MB)
- Optional `stgcn_2s` joint+bone (implemented, not primary)
- Cache/venv: junctions to `skeleton_imu_v2`

## Protocol notes

- Holdout users `{8,9,24}`; GroupKFold by user
- Late-fuse α selected on **non-holdout OOF only** (no holdout peeking). Peeked α=0.4 would give 0.5485 holdout — **not used**
- Gate: holdout ≥0.547 **or** clearly better OOF vs v3 → OOF path fires

## Layout

- `graph.py`, `model.py`, `model_2s.py`, `dataset.py`, `train.py`, `infer.py`
- `fuse_eval.py`, `nested_latefuse.py`, `nested_latefuse_submit.py`
- `checkpoints_v2/` (holdout), `checkpoints_cv/` (folds+all)
- `metrics.json`, `v7_summary.json`, `agreement.json`, `nested_latefuse_v2.json`
- `submission_v7_latefuse.csv`

## Reproduce

```powershell
cd D:\CUHK-X\Small-Model-Track\baselines\v7_stgcn
.\.venv\Scripts\Activate.ps1
python train.py --mode holdout --model stgcn_fuse --epochs 55 --batch-size 20 --ckpt-dir checkpoints_v2 --metrics-out metrics_holdout_v2.json
python train.py --mode cv --model stgcn_fuse --epochs 50 --batch-size 20 --ckpt-dir checkpoints_cv --metrics-out metrics_cv.json
python nested_latefuse_submit.py
```
