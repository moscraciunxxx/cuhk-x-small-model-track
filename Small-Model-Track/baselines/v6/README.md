# CUHK-X Small Model Track — baselines/v6

Quiet MSI run. Goal: beat MidFuse v2 holdout **0.537** by ≥+0.01 using confuse-pair / hierarchical specialists on Skeleton+IMU.

## Primary result (leak-free)

| Split | Acc | vs 0.537 |
|-------|-----:|---------:|
| MidFuse v2b holdout {8,9,24} | 0.5366 | −0.0004 |
| **v6 raw-pair specialists (nested-OOF policy)** | **0.5406** | **+0.0036** |
| Clear-win gate (≥+0.01) | — | **not met** |

- **ping_disk_saver**: `false`
- **submission.csv**: **not updated** (gain below clear-win / “clearly better” bar)
- Model size: MidFuse ~8.4MB + 15× RawPairNet ~80k params each (~0.3MB) ≪ 100MB

### Winning honest recipe (small gain)

1. Base: frozen MidFuse v2b `checkpoints_midfuse_v2b/best_holdout.pt`
2. Train tiny dual Conv1D binary specialists on raw skel+IMU for confuse pairs (error_analysis top confusions)
3. At inference: if MidFuse top-2 form a known pair, margin < 0.75 and specialist conf ≥ 0.7, defer to specialist
4. Nested selection used **MidFuse 5-fold OOF logits** on non-holdout users (not in-sample logits)

Selected pairs by OOF delta: `(29,32)`, `(12,13)`, `(8,9)` — holdout **selected** 0.5406; all-pairs 0.5386.

## What was tried

| Approach | Holdout (honest) | Notes |
|----------|-----------------:|-------|
| Feat MLP on MidFuse penultimate+logits | ~0.538–0.541 | Tiny; thr peeking up to 0.541 |
| Raw dual Conv pair specialists + nested OOF | **0.5406** | Primary |
| Cluster multi-class raw (kitchen/exercise/…) | 0.5386 | Nested thr; oracle peek 0.5485 |
| Leave-3-user policy + temp/bias calib | ≤0.5366 | Calib helped OOF, **hurt** holdout |
| Full second-stage 40-way on pen features | ~0.51–0.52 | Worse |
| **Leaky** raw + holdout early-stop + thr=1.01 | ~0.562 | Invalid; not used |

## Layout

- `specialists_v6.py` — MidFuse feature extractors + feat-MLP pair/multi specialists
- `refine_v6.py` — raw dual Conv specialists + gated deferral
- `final_v6.py` / `final_v6b.py` — nested selection + holdout eval (v6b = OOF-correct)
- `cluster_v6c.py` — cluster specialists (top1-in-group deferral)
- `calib_L3_v6d.py` — temperature/bias + leave-3-user selection
- `checkpoints/raw_specialists_holdout_train.pt` — specialists trained on non-holdout
- `v6_summary.json`, `holdout_probe_summary.json`, `nested_oof_selection.json`, …
- `logs/` — run logs

## Reuse

```powershell
cd D:\CUHK-X\Small-Model-Track\baselines\v6
D:\CUHK-X\Small-Model-Track\baselines\skeleton_imu_v2\.venv\Scripts\Activate.ps1
python final_v6b.py
```

Cache/code from `baselines\skeleton_imu_v2\` (no Radar/Thermal).

## Decision

Holdout **+0.0036** vs 0.537 is real but below the preferred **+0.01** clear-win threshold. No full CV submit promotion; keep current `submission.csv` (v3 fold ensemble).
