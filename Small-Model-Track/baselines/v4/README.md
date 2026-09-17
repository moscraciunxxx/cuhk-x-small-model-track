# CUHK-X Small Model Track - v4

Quiet overnight on MSI. Goal: beat MidFuse v2 holdout **0.537** and/or improve OOF vs v3 **0.5015**.

## Headline

| Exp | CV / OOF | Holdout {8,9,24} | vs 0.537 |
|-----|----------|------------------|----------|
| v2 MidFuse (baseline) | CV 0.505 | **0.537** | - |
| v3 fold ensemble | OOF 0.5015 | - | - |
| **BoneMidFuse** | CV 0.499 / OOF 0.496 | 0.497 | FAIL |
| **Multi-seed MidFuse** (42/123/7) | - | mean 0.502 (best seed7=0.531) | FAIL |
| **Gated Thermal** + MidFuse | - | 0.529 (base 0.537) | FAIL (hurt) |

**Beats holdout 0.537? NO.** Best v4-observed holdout = **0.529** (thermal) / **0.531** (seed7 alone). Do **not** ping Disk Saver.

## What failed (all 3 high-ROI ideas)

1. **Bone/angle features** (168-d: unit bones, lengths, velocities, joint cosines) + IMU MidFuse-like Conv: CV 0.499 +/- 0.050, holdout 0.497, OOF 0.496. No lift vs raw joints.
2. **Multi-seed bag**: holdout probes 0.493 / 0.483 / 0.531 (mean 0.502). Seed variance high; none beat 0.537. Full-data 3-seed bag still useful for diversity.
3. **Gated Thermal** tiny CNN (8 frames @64px, ~111k params): holdout 0.529 vs MidFuse-only 0.537. Residual thermal did not help (Radar skipped as planned).

## Submissions

- **Keep primary submit:** `..\v3_ensemble\submission_v3_ensemble.csv` (OOF 0.5015)
- v4 diversity candidate (no proven holdout lift): `submission_v4_best.csv` (= `submission_v4_full_mix.csv`: 0.4 v2_folds + 0.3 seed_bag + 0.3 bone_all)
- Cleaner v4 without failing bone: `submission_v2folds_seeds.csv`
- Agreement: v4_best vs v3 = **0.807**; vs v2 single = **0.714**

Ensemble footprint if using v2 folds + 3 seeds only: ~8 models x ~8MB ≈ **64MB** (<100MB). Including bone_all still OK.

## Layout

- `bones.py`, `train_bone.py`, `checkpoints_bone/`
- `train_seeds.py`, `checkpoints_seeds/`, `checkpoints_seeds_holdout/`
- `train_thermal.py`, `checkpoints_thermal/`
- `ensemble_v4.py`, `metrics.json`, `README.md`
- `cache/`, `.venv` junctions to skeleton_imu_v2

## Reproduce

```powershell
cd D:\CUHK-X\Small-Model-Track\baselines\v4
.\.venv\Scripts\Activate.ps1
python build_bone_cache.py
python train_bone.py --mode cv --model bonemidfuse --epochs 50 --batch-size 32 --lr 8e-4 --patience 15
python train_seeds.py --mode holdout --seeds 42 123 7
python train_seeds.py --mode full --seeds 42 123 7
python train_thermal.py --epochs 20 --batch-size 12
python ensemble_v4.py
python summarize_v4.py
```
