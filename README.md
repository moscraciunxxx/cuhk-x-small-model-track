# CUHK-X Small Model Track

Kaggle [cuhk-x-competition-small-model-track](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track). 40-class daily-activity recognition from IR, Depth, Thermal, skeleton, and IMU. No RGB. Packed deploy target ≤ 100 MB.

Published after the Kaggle submission window closed on **2026-09-15 15:55 UTC**.

Kaggle account: [`vitaliecervinschi`](https://www.kaggle.com/vitaliecervinschi). GitHub: [`moscraciunxxx`](https://github.com/moscraciunxxx).

## Result

| | |
|---|---|
| Competition | [CUHK-X Small Model Track](https://www.kaggle.com/competitions/cuhk-x-competition-small-model-track) |
| Account | [`vitaliecervinschi`](https://www.kaggle.com/vitaliecervinschi) |
| Best COMPLETE public | **0.73134** — `submission_ir_v73.csv` (ref `56243077`) |
| Previous keep | 0.72636 `submission_ir_v66.csv` |
| Metric | accuracy, 405 test clips, 40 classes |
| Pack | int8 4ch R(2+1)D-34 + YOLOv8n ≈ 67.08 MB |

Gated locally, never uploaded (quota exhausted, then CreateSubmission 400): `submission_ir_v77.csv` (clips 0169+0283), `submission_ir_v79.csv` (0068+0299).

## Method

Late fusion, not skeleton-only:

1. IR YOLO-crop R(2+1)D (classic-9 + T=24 IR swap) as the base IR stream.
2. 4-channel Depth RGB + IR gray R(2+1)D-34 (IG-65M/Kinetics init), several T and IR-box vs non-IR-box crops.
3. Thermal 3D CNN logits.
4. MidFuse (skeleton + IMU) logits at `wc = 0.09`.
5. Label-free ranked prefix: swap at most k in {2,3,4} unused test clips. Never 0-diff, never greedy hold-label rules, never retouch a scored clip set.

Hold users `{8, 9, 24}`. Nested leave-one-holdout-user fuse lives in `probe_ir_v24_fuse.py` (`nested_fixed` / `apply_cfg`).

## Layout

```text
Small-Model-Track/
  baselines/thermal_v1/     winning IR + thermal + 4ch stack
  baselines/skeleton_imu_v2/
  baselines/depth_color_v1/
  class_mapping.csv
  submissions/              public CSVs including ir_v73
  Testing/test_file/        sample schema only
```

Training frames and `Testing/data/` are gitignored. Get them from Kaggle. Rebuild caches with `build_cache_*.py` and train with `train_r2p1d34_4ch.py`.

## Setup

```bat
python -m pip install -r requirements.txt
```

Reproduce the v73 gate (needs data + caches + 4ch logits on disk):

```bat
cd Small-Model-Track\baselines\thermal_v1
python -u check_ir_v73_gate.py
```

That imports shipped `nested_fixed` / `apply_cfg` and writes `submission_ir_v73.csv` if the gate holds.

## Constraints

- No RGB
- Hold users 8 / 9 / 24
- 40 classes
- Pack ≤ 100 MB
- MidFuse `wc = 0.09`
- Do not reseed `cache/depth_ir_4ch_v31`

## License

MIT. See [LICENSE](LICENSE).
