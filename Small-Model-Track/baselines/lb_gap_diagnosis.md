# LB gap diagnosis (2026-09-07)

## Public vs local
- Public LB: **0.41293** (v19 submission.csv)
- Local nested OOF / holdout: **0.6653 / 0.6436**

## Format / protocol — NOT the bug
Verified against `sample_submission.csv` and `test.csv`:
- 405 rows; columns `path,prediction`
- Path order **exact match** (incl. trailing `/`)
- Predictions int64 in **[0, 39]**; 0 nulls; 0 duplicate paths; 37 unique classes used
- Schema matches competition sample

Conclusion: **not a CSV/id/format bug**.

## Why local >> public
1. **Weak modality for this LB**: skeleton+IMU KD stack. Public notebooks reaching **~0.71–0.80** use **Thermal** or **YOLO person-crop + R(2+1)D on Depth_Color**. Official data page: reference baseline uses **Depth_Color**.
2. **Optimistic local CV**: KD students distilled from ensemble soft labels; successive KD blends (v15–v19) can inflate OOF/holdout without transferring to true cross-subject public split (test users 10–11, 25–26). Holdout users 8/9/24 may correlate with train better than true test subjects.
3. Pred distribution: class 36 heavily over-predicted in v19 (53/405) — possible mode bias.

## Next actions
1. Train Depth_Color / Thermal temporal CNN (in flight) under 100MB.
2. Optional calibration: submit non-KD MidFuse `submission_skeleton_imu_v2.csv` via **kaggle CLI** (needs `kaggle.json`).
3. Ping Disk Saver on next public LB only.

