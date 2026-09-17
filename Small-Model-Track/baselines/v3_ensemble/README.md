# CUHK-X Small Model Track — v3 ensemble

Quiet overnight run on MSI. Primary deliverable: **5-fold MidFuse softmax ensemble**.

## Headline vs v2

| | v2 single MidFuse | v3 fold ensemble (v2 ckpts) | longer 80ep | MixUp holdout |
|--|--|--|--|--|
| GroupKFold mean±std | **0.505±0.058** | OOF **0.5015** | 0.501±0.056 | — |
| Holdout {8,9,24} | **0.537** | — | 0.533 | 0.511 |
| Test agreement w/ single | 1.0 | **0.679** | 0.637 | — |
| Submission | submission_skeleton_imu_v2.csv | **submission_v3_ensemble.csv** | submission_v3_long_ensemble.csv | — |

**Recommendation:** use `submission_v3_ensemble.csv` (diversity vs single; OOF≈CV). Longer train / MixUp did **not** beat v2.

## Reproduce ensemble

```powershell
cd D:\CUHK-X\Small-Model-Track\baselines\v3_ensemble
.\.venv\Scripts\Activate.ps1
python oof_and_ensemble.py --ckpt-dir ..\skeleton_imu_v2\checkpoints --out submission_v3_ensemble.csv
```

## Files
- `oof_and_ensemble.py`, `metrics_oof_ensemble.json`, `metrics.json`
- `checkpoints/` — longer-train fold ckpts (did not beat v2)
- `cache/`, `.venv` — junctions to v2 / skeleton_v1
- MidFuseWide registered in `model.py` (unused after negative probes)
