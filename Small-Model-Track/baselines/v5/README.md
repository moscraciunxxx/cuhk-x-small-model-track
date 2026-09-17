# CUHK-X Small Model Track — v5

Quiet MSI run. Tasks: error analysis, promote submission, Depth/IR probe, optional MidFuse focal tweak.

## Headline

| Item | Result |
|------|--------|
| MidFuse v2b holdout (baseline) | **0.537** |
| Beat 0.537 clearly? | **No** (no Disk Saver ping) |
| Primary upload | `D:\CUHK-X\Small-Model-Track\submission.csv` (= v3 fold ensemble) |
| Focal MidFuse holdout | **0.457** (−0.079) — dead end |
| Depth+IR TinyTempCNN alone | **0.145** holdout |
| Best late-fuse (0.7 MidFuse + 0.3 vision) | **0.541** (+0.004) — **fails +0.5pp gate**; no full CV |

## Artifacts

- `error_analysis.md` / `error_analysis.json` — holdout + OOF confusion
- `modality_coverage.json` — Depth/IR empty rates = **0** train & test
- Repo-root `submission.csv` + `submissions\submission_v3_ensemble.csv` + `submission_README.txt`
- `tiny_vision.py` (~0.75 MB / 197k params), `probe_depth_ir.py`, `probe_focal.py`
- `cache/vision_*.npz` — K=8 frames, 48×64, depth-gray + IR
- `v5_summary.json` — machine-readable outcome for parent

## Error analysis (summary)

- Semantic confusions dominate: Pour↔Stir, Peel→Stir, Sweep↔Mop, Turn_pages↔Read, Squats→Stand_up
- Imbalance contributes (support–recall corr ≈ 0.41 holdout / 0.44 OOF) but does not fully explain errors
- Worst holdout recalls: Comb_hair, Write, Watch_TV, jumping_jacks, Peel_fruits, Play_games, Squats

## Depth/IR verdict

Full coverage (not a missing-data issue). Tiny temporal CNN ≪10MB trains fine but is weak alone (~0.15). Late fusion with MidFuse gives only a noisy +0.4pp on holdout — **do not** invest in full GroupKFold CV.

## Skip

Radar / Thermal (known bad from v4).
