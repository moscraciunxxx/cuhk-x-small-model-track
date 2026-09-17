# Small-Model-Track v11 - ABG reweight + v9/v10 blends

## Summary vs v9 / v10

| | v9 blend_ABD | v10 selected | **v11 selected** (`wABG_4_2_3`) |
|---|---|---|---|
| Nested OOF | 0.5486 | 0.5585 | **0.5614** |
| Holdout | 0.5663 | 0.5624 | **0.5663** |

Win criteria (AND): nested OOF >= **0.5585** AND holdout >= **0.5663**.  
clear_win_and=True. overwrite_submission=True. ping_disk_saver=True.

Selection used **nested GroupKFold on non-holdout OOF only**; holdout evaluated last.

## Selected method

`wABG_4_2_3` params=`{"wa": 4, "wb": 2, "wg": 3, "members": ["A", "B", "G"]}`

Weighted blend of three nested fuses:

1. **A** `pow_mfavg_st_gru_cf` - nested power-mean over (mf_avg3, ST-GCN, GRU, compact_fuse)
2. **B** `eq_mfavg_st_gruavg` - equal softmax of (mf_avg3, ST-GCN, gru_avg2)
3. **G** `pow_mfavg_st_gruavg` - nested power-mean over (mf_avg3, ST-GCN, gru_avg2)

v10 was equal(A,B). v11 searches integer weights (wa,wb,wg) on nested OOF only.

## Protocol

1. Reuse `baselines/v10/oof_logits_v10.npz` branch OOF logits
2. Nested OOF for base fuses A/B/G (+ others); ABG weight grid; v9/v10 blends; margin gates
3. Select max nested OOF
4. Holdout last (AND gate for overwrite)
5. Test: 5-fold avg softmax -> same recipe -> submission

## Key ablations (nested -> holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| v9_blend_ABD | 0.5486 | 0.5663 |
| v10_selected | 0.5585 | 0.5624 |
| **wABG_4_2_3 (selected)** | **0.5614** | **0.5663** |
| peek best hold (NOT used) | 0.5528 | 0.5861 |

Both-gate hits in search: 8.

## Files

- `v11_abg_blend.py` - pipeline
- `holdout_logits.npz` - cached holdout branch logits
- `metrics.json` - full comparison
- `submission_v11.csv` / `submission_v11_candidate.csv` / `submission_v11_probs.npz`

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v11\v11_abg_blend.py
```

No TTA. No sklearn stacker. <=100MB (logits/probs only).
