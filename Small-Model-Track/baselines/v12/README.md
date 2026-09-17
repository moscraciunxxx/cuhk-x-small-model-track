# Small-Model-Track v12 - expand ABG + temp-scale + deepconv-diverse

## Summary vs v11

| | v9 | v10 | v11 `wABG_4_2_3` | **v12 selected** (`v11ref`) |
|---|---|---|---|---|
| Nested OOF | 0.5486 | 0.5585 | **0.5614** | **0.5614** |
| Holdout | 0.5663 | 0.5624 | **0.5663** | **0.5663** |

Clear-win overwrite/ping: holdout >= **0.576** OR >= **0.5713** (0.5663+0.005), AND nested OOF >= **0.558**.  
Prefer: holdout > 0.5663 with OOF >= 0.5614.  
clear_win=False. prefer_hit=False. overwrite_submission=False. ping_disk_saver=False.

Selection = **max nested OOF only**; holdout evaluated last. **Track `submission.csv` left as v11** (no clear win).

## Ideas tried

1. Expanded integer-weight search around A/B/G and both-gate neighborhood; ABG+extra; ABG{H,E,N} 4-way (raw + `ts_` substrates). n_methods=4586.
2. Diverse OOF-complete deepconv recipes (H/J/K/N/O/P) - no new training (deepconv already in `oof_logits_v10.npz`).
3. OOF-only per-branch temperature scaling (nested GroupKFold), then re-blend. Temp scaling did **not** help nested OOF (T* often 0.5; `ts_v11ref` OOF 0.5561 < raw 0.5614).
4. No TTA, no sklearn stackers, no holdout peeking for selection.

## Honest table

| Method | Nested OOF | Holdout |
|---|---|---|
| v9 | 0.5486 | 0.5663 |
| v10 | 0.5585 | 0.5624 |
| v11 wABG_4_2_3 | 0.5614 | 0.5663 |
| v12 selected (`v11ref`) | **0.5614** | **0.5663** |
| best hold w/ OOF>=0.558 (NOT selected) | 0.5589 | 0.5703 |
| peek best hold (NOT used) | 0.5528 | 0.5861 |

- prefer_hits (OOF>=0.5614 & hold>0.5663): **0**
- clear_hits (hold>=0.5713/0.576 & OOF>=0.558): **0**
- OOF>=0.558 & hold>v11: 27 methods exist, but max hold among them is **0.5703** (delta =+0.0040 < 0.005 gate)

## Selected

`v11ref` params=`{"members": ["A", "B", "G"], "w": [4, 2, 3], "substrate": "raw"}`

Equivalent to v11 `wABG_4_2_3` (A:B:G = 4:2:3). Expanded search tied this max nested OOF; no recipe beat both metrics under the clear-win gate.

## Notes on deepconv-diverse

| Base | Nested OOF | Holdout |
|---|---|---|
| H pow(mf_avg,stgcn,gru_avg,deepconv) | 0.5437 | 0.5723 |
| N pow(mf_avg,stgcn,gru_avg,deepconv,cfuse) | 0.5474 | 0.5762 |
| J eq(mf_avg,stgcn,deepconv,cfuse) | 0.5462 | 0.5683 |

High holdout on H/N alone but nested OOF too low to select; ABG+H mixes reach hold0.5703 only near the OOF floor (0.558), still short of clear-win.

## Files

- `v12_expand_blend.py` - pipeline
- `holdout_logits.npz` - cached holdout branch logits
- `metrics.json` - full comparison (4586 methods)
- `submission_v12.csv` / `submission_v12_candidate.csv` / `submission_v12_probs.npz` (same preds as v11; not promoted)

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v12\v12_expand_blend.py
```

100MB (logits/probs only). Nested GroupKFold discipline. Track submission remains v11.
