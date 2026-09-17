# Small-Model-Track v9 — seed MidFuse + method blend

## Summary vs v8

| | v8 `conf_weighted_3` | **v9 selected** (`blend_ABD`) |
|---|---|---|
| Nested OOF | 0.5383 | **0.5486** (+0.0103) |
| Holdout | 0.5564 | **0.5663** (+0.0099) |
| Branches | MidFuse + ST-GCN + GRU | MidFuse×2 seeds + ST-GCN + GRU + DeepConv |
| Fusion | conf-weighted 3-way | equal blend of 3 nested fuse methods |

Win criteria met: nested OOF ≥ 0.548 **and** holdout ≥ 0.566.  
`submission.csv` overwritten. `ping_disk_saver`: **true**.

Selection used **nested GroupKFold on non-holdout OOF only**; holdout evaluated last.

## What changed vs v8

1. **Second MidFuse seed** (`checkpoints_midfuse_s123`, seed=123) — same recipe as v2b.
2. **DeepConv** branch (existing fold ckpts) for CNN diversity.
3. **Method-level blend** (no leaky stacker): equal average of three nested-honest fuses:
   - **A** `pow_all5`: power-mean over (MF, MF2, ST-GCN, GRU, DeepConv)
   - **B** `eq_mfavg_st_gru`: equal softmax of (MF seed-avg, ST-GCN, GRU)
   - **D** `pow_mf_mf2_st_gru`: power-mean over (MF, MF2, ST-GCN, GRU)
4. LR-flip / temporal TTA **hurt** (~−2–3pp single-model OOF) — not used.
5. Avoided sklearn stackers (failed in v8).

## Protocol

1. GroupKFold OOF logits per branch (cached `oof_logits_v9b.npz`).
2. Nested OOF probs for each base fuse (hyperparams fit inside each GroupKFold split).
3. Equal blend of method OOF probs → nested score for `blend_ABD` (no extra fit).
4. Holdout: fit each base fuse on full non-holdout; apply; equal-blend once.
5. Test: 5-fold avg softmax per branch → same fuse → `submission_v9.csv`.

## Key ablations (nested → holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| conf_v8_3 (replay) | 0.5375 | 0.5584 |
| pow_all5 | 0.5478 | 0.5604 |
| eq_mfavg_st_gru | 0.5470 | 0.5723 |
| conf_mfavg_st_gru | 0.5478 | 0.5743 |
| pow_mf_mf2_st_gru | 0.5474 | 0.5663 |
| **blend_ABD (selected)** | **0.5486** | **0.5663** |

## Files

- `v9c_method_blend.py` — selected pipeline
- `v9b_seed_fuse.py` / `v9_fuse_tta.py` — prior ablations
- `oof_logits_v9b.npz` — branch OOF logits
- `metrics.json` — full comparison
- `submission_v9.csv` / `submission_v9_probs.npz`
- MidFuse seed2 ckpts: `../skeleton_imu_v2/checkpoints_midfuse_s123/`

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v9\v9c_method_blend.py
```

(Requires MidFuse seed123 CV ckpts already trained.)
