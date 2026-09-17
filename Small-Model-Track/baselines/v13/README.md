# Small-Model-Track v13 — ST-GCN two-stream base + nested late-fuse

## Summary vs v11

| | v11 `wABG_4_2_3` | **v13 selected** |
|---|---|---|
| Nested OOF | 0.5614 | **0.5631** |
| Holdout | 0.5663 | **0.5703** |
| New base | — | **stgcn_2s** (joint+bone+IMU) |

Clear-win overwrite gate: holdout **> 0.5663** AND nested OOF **≥ 0.558**.  
clear_win=True. overwrite_submission=True. **ping_disk_saver=true**.

Selection = max nested OOF only; holdout evaluated last.

## Selected method

`v11style211_plus_eq_mf_st_s2s_x1.0`

- Classic ABG-style mix with weights (wa,wb,wg)=(2,1,1) over bases A/B/G (mf_avg + ST-GCN + GRU/cfuse recipes)
- Plus equal-weight `eq_mf_st_s2s` = equal softmax of (mf_avg, stgcn, **s2s**)
- Extra weight w=1.0

## New base: ST-GCN two-stream (`stgcn_2s`)

Trained with GroupKFold via `baselines/v7_stgcn/train.py --model stgcn_2s`:

| Metric | Value |
|--------|------:|
| Params | 1,148,080 (~4.7 MB ckpt) |
| CV mean±std | **0.4766 ± 0.055** |
| Per-fold | 0.408 / 0.500 / 0.532 / 0.530 / 0.414 |
| Holdout alone | 0.4515 |
| OOF non-holdout | 0.4815 |
| vs v7 ST-GCN CV | 0.457 → **+0.020** |

Complementary to MidFuse (agreement moderate); alone weaker than MidFuse but lifts nested fuse.

## Ideas tried / not promoted

1. **MidFuseWide** (wider skel + MS IMU + SE, 6.3M): severe overfit (fold0 ~0.35 vs v2b 0.45). Killed. No MixUp.
2. **MidFusePlus** smoke (skel velocity + MS IMU): still lagged v2b learning curve at epoch 12; deferred after s2s clear-win.
3. **MS-STGCN multi-scale temporal** (`model_ms_stgcn.py`): implemented, not trained this round (GPU time used by s2s).

## Protocol

1. Train `stgcn_2s` 5-fold GroupKFold + holdout (`checkpoints_stgcn2s/`)
2. Dump OOF/holdout logits (`dump_oof.py`)
3. Nested fuse with v10/v11 branch logits + s2s (`v13_fuse.py`)
4. Select max nested OOF; evaluate holdout last
5. Test: 5-fold avg softmax → same recipe → submission

## Key ablations (nested → holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| v11ref / wABG_4_2_3 | 0.5614 | 0.5663 |
| **v13 selected** | **0.5631** | **0.5703** |
| best hold w/ OOF≥0.558 (NOT selected) | 0.5585 | 0.5743 |
| peek best hold (NOT used) | 0.5540 | 0.5842 |

## Files

- `model.py` — MidFusePlus lite (deferred)
- `model_ms_stgcn.py` — MS temporal 2s ST-GCN (ready, not trained)
- `train_mfw.py` / `train_ms2s.py` — trainers for deferred bases
- `dump_oof.py` — OOF/holdout logit dump
- `v13_fuse.py` — nested selection + submit
- `checkpoints_stgcn2s/` — fold + holdout ckpts
- `oof_stgcn2s.npz`, `holdout_stgcn2s.npz`
- `metrics_stgcn2s.json`, `metrics.json`
- `submission_v13.csv` / `submission_v13_candidate.csv` / `submission_v13_probs.npz`

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v7_stgcn\train.py --mode cv --model stgcn_2s --epochs 50 --batch-size 16 --lr 8e-4 --patience 12 --seed 42 --no-balanced-sampler --label-smoothing 0.05 --ckpt-dir baselines\v13\checkpoints_stgcn2s --metrics-out baselines\v13\metrics_stgcn2s.json --cache-dir baselines\skeleton_imu_v2\cache

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v13\dump_oof.py --kind stgcn_2s --ckpt-dir baselines\v13\checkpoints_stgcn2s --oof-out baselines\v13\oof_stgcn2s.npz --holdout-out baselines\v13\holdout_stgcn2s.npz

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v13\v13_fuse.py
```

No TTA. No sklearn stacker. ≤100MB (s2s ~4.7MB + logits). Nested GroupKFold discipline.
