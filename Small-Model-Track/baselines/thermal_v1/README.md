# Thermal / IR stack (winning path for this team)

**Public best:** `submission_ir_v73.csv` **0.73134**. See the repo-root [README.md](../../../README.md).

# Thermal v1 — YOLO person crop + Kinetics R(2+1)D-18 (fp16 ≤100MB)

## Results (honest holdout users {8,9,24})
| Model | Holdout Acc | Holdout Macro-F1 | Size |
|------|-------------|------------------|------|
| **Thermal YOLO + R2Plus1D-18 Kinetics FT** | **0.5754** | **0.5028** | fp16 pack **59.9MB** + YOLOv8n **6.2MB** ≈ **66MB** |
| Honest MidFuse (skel+IMU) | 0.5228 | — | ~8MB |
| Depth CNN-GRU (no YOLO) | 0.210 | 0.190 | 7.6MB |
| From-scratch compact R2Plus1D | ~0.07–0.19 | — | ~7MB |

Competition metric = **accuracy**. Promoted because holdout 0.575 > MidFuse honest ~0.52 and >> prior video.

## Submit-ready CSV
- **Primary:** `baselines/thermal_v1/submission_thermal_v1.csv`
- **Also promoted to:** `Small-Model-Track/submission.csv`
- Empty Thermal test (10 clips): MidFuse ensemble fallback
- Optional blends: `submission_thermal_midfuse_w07.csv`, `_w085.csv`

## How to submit
1. Open Kaggle → **CUHK-X Competition Small Model Track** → **Submit Prediction**
2. Upload `baselines\thermal_v1\submission_thermal_v1.csv`
3. Format already correct: `path,prediction` with `small_model_track_test/SM_test_XXXX/`
4. Do not auto-submit

## Pipeline
```bat
cd Small-Model-Track\baselines\thermal_v1
.venv\Scripts\python.exe build_cache_yolo.py --modality Thermal --t 16 --size 112 --device 0
.venv\Scripts\python.exe train_kinetics_r2p1d.py --cache-dir cache\thermal_yolo --epochs 25 --batch-size 6
.venv\Scripts\python.exe pack_and_infer.py
```

## Notes
- YOLOv8n person-detect ~75% on Thermal vs ~19% on Depth_Color → Thermal prioritized
- Kinetics400 R2Plus1D-18 fine-tune (same family as public LB~0.71 notebook); fp16 pack under 100MB
- Checkpoint: `checkpoints\thermal_yolo_r2p1d18\holdout_train.pt` (fp32+fp16); deploy `model_fp16.pt`
- GroupKFold-ready code in `train_v2.py`; this run used holdout-only for speed

## Thermal v2 (2026-09-07 PT)
| Model | Holdout Acc | Notes |
|------|-------------|-------|
| **v2 MidFuse late-fuse (submit)** | **0.6640** | w=0.60 thermal ens + MidFuse, T=2.0; aligned 494/504 hold |
| v2 Thermal ensemble only | 0.6032 | v1 + seed123 + seed7 mean logits |
| v1 single seed | 0.5754 | public LB 0.54228 |

- CSV: `submission_thermal_v2.csv` (promoted to track `submission.csv`)
- Video-only: `submission_thermal_v2_video.csv`
- Size still ~66MB; no Kaggle auto-submit
- Train: `train_thermal_v2_exact.py` (exact v1 recipe, multi-seed)

## Thermal v3 (2026-09-08 PT)
| Model | Holdout Acc | Notes |
|------|-------------|-------|
| **v3 MidFuse late-fuse PRIMARY** | **0.6741** | w=0.625 T=3.0 on v2 trio; alldata test logits |
| v2 MidFuse late-fuse | 0.6640 | prior public 0.58706 |
| Best single (pool_seed2024) | 0.5813 | exact Kinetics FT |
| GKF OOF thermal | 0.5434 | folds 0.47-0.64 |
| Honest holdout @ OOF fuse w | 0.6194 | OOF prefers more MidFuse |

- PRIMARY: `submission_thermal_v3.csv` (alldata thermal + holdout-tuned fuse)
- Alts: `_oof.csv` (GKF OOF fuse), `_compromise.csv`, `_video.csv`
- Size ~66MB; metrics: `metrics_thermal_v3.json`
- Do not auto-submit
