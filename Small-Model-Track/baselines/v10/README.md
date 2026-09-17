# Small-Model-Track v10 - MidFuse x3 + GRU x2 + compact_fuse blend

## Summary vs v9

| | v9 blend_ABD | **v10 selected** |
|---|---|---|
| Nested OOF | 0.5486 | **0.5585** (+0.0099) |
| Holdout | 0.5663 | 0.5624 (-0.0040) |
| Branches | MF x2 + ST-GCN + GRU + DeepConv | MF x3 + ST-GCN + GRU x2 + DeepConv + compact_fuse |
| Fusion | equal blend of 3 nested fuses | equal blend of 2 nested fuses |

**Clear-win gate met:** nested OOF >= **0.558** (holdout gate 0.576 not met).
submission.csv overwritten. ping_disk_saver: **true**.

Selection used **nested GroupKFold on non-holdout OOF only**; holdout evaluated last (not used for selection).

## Selected method

eqblend_pow_mfavg_st_gru_cf_eq_mfavg_st_gruavg = equal average of:

1. **pow_mfavg_st_gru_cf** - nested power-mean over (mf_avg3, ST-GCN, GRU, compact_fuse)
2. **eq_mfavg_st_gruavg** - equal softmax avg of (mf_avg3, ST-GCN, gru_avg2)

Where:
- mf_avg3 = logit-avg of MidFuse seeds **42 / 123 / 7**
- gru_avg2 = logit-avg of GRU seeds **42 / 7**
- compact_fuse = new dual-stream compact model (seed 42)

## What changed vs v9

1. **Third MidFuse seed** (checkpoints_midfuse_s7, seed=7)
2. **Second GRU seed** (checkpoints_gru_s7, seed=7)
3. **New compact_fuse branch** (checkpoints_compact_fuse) - architectural diversity
4. Expanded nested method search: power/conf/logit-mean/rank, subset search, method blends
5. Class-prior / hard-example calib explored (small or no gain alone)
6. **sklearn stackers** tried again -> still overfit; **not used**
7. **TTA skipped** (hurt in v9)
8. **Radar fill skipped** (not a quick OOF/holdout win)

## Key ablations (nested -> holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| blend_ABD v9 replay | 0.5486 | 0.5663 |
| pow_mfs_st_gru (MF x3) | 0.5491 | 0.5584 |
| pow_all_grus | 0.5503 | 0.5604 |
| pow_mfavg_st_gru_cf | 0.5528 | 0.5545 |
| **eqblend pow_cf + eq_gruavg (selected)** | **0.5585** | **0.5624** |
| peek best holdout (NOT used) | ~0.549 | 0.5802 conf_mfavg_st_gru_cf |

## Protocol

1. GroupKFold OOF logits per branch -> oof_logits_v10.npz
2. Nested OOF for each base fuse (hyperparams fit inside each GroupKFold split)
3. Equal method-blend of nested OOF probs -> honest nested score
4. Select max nested; evaluate holdout last
5. Test: 5-fold avg softmax per branch -> same fuse -> submission

## Files

- 10b_advanced.py - selected pipeline
- 10_fuse.py - earlier MF x3 search
- oof_logits_v10.npz - branch OOF logits
- metrics.json - full comparison
- submission_v10.csv / submission_v10b.csv / probs npz
- New ckpts: ../skeleton_imu_v2/checkpoints_midfuse_s7/, checkpoints_gru_s7/, checkpoints_compact_fuse/

## Reproduce

`	ext
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v10\v10b_advanced.py
`

## Size / constraints

- No large pretrained weights; MidFuse ~2.1M, compact_fuse ~1.1M, DeepConv ~2.6M
- GroupKFold / nested OOF discipline maintained
