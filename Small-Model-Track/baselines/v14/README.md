# Small-Model-Track v14 — MS-STGCN branch + nested late-fuse

## vs v13

| | v13 selected | **v14 selected (max nested OOF)** | soft-prefer (hold↑, OOF≥0.560) |
|---|---:|---:|---:|
| Nested OOF | **0.5631** | **0.5635** | 0.5606 |
| Holdout | **0.5703** | 0.5663 | **0.5762** |
| Clear-win overwrite | yes (v13 gate) | **no** | no |

**Track `submission.csv` left as v13.** No overwrite / no `ping_disk_saver`.

### Clear-win gate (v14)

Overwrite + ping only if `holdout ≥ 0.580` **OR** (`holdout ≥ 0.575` **and** nested OOF ≥ `0.568`).
Prefer holdout `> 0.5703` with OOF not below `0.560` (soft) — recorded but not overwritten.

Selection protocol unchanged: **max nested OOF**; holdout last.

## Selected method

`v11style523_plus_eq_mf_st_s2s_ms_x3.0`

- Classic ABG (5,2,3) + extra `eq_mf_st_s2s_ms` (equal softmax of mf_avg, stgcn, s2s, **ms2s**) at weight 3.0
- Nested OOF 0.5635 (+0.0004 vs v13) but holdout 0.5663 (−0.004 vs v13)

## Soft-prefer (not selected; not overwritten)

`v13sel_plus_eq_mf_st_ms_x0.5` — nested OOF 0.5606, holdout **0.5762**
(v13 selected recipe + light MidScale `eq_mf_st_ms`). Meets soft preference but fails clear-win (needs OOF≥0.568 with hold≥0.575, or hold≥0.580).

## New base: MS-STGCN two-stream (`ms_stgcn_2s`)

Multi-scale temporal (kernels 3/5/9) joint+bone ST-GCN + IMU gate. Trained GroupKFold via `train_ms2s.py`.

| Metric | ms2s | s2s (v13) |
|--------|-----:|----------:|
| Params | 2,037,704 (~8.3 MB ckpt) | 1,148,080 |
| CV mean±std | 0.4616 ± 0.066 | 0.4766 ± 0.055 |
| Per-fold | 0.406 / 0.545 / 0.486 / 0.507 / 0.365 | 0.408 / 0.500 / 0.532 / 0.530 / 0.414 |
| Holdout alone | 0.4535 | 0.4515 |
| OOF non-holdout | 0.4662 | 0.4815 |

Diversity vs s2s (non-holdout): agreement **0.559**; ms-only-correct **0.095** when s2s wrong. Complementary enough to enter nested search; alone weaker than MidFuse.

## Ideas tried / not promoted

1. **MS-STGCN** — trained full CV+holdout; fused; no clear-win.
2. **MidFusePlus** — smoke fold0 best ~0.30 vs v2b ~0.45; **not finished** (do not repeat MidFuseWide overfit path).
3. **Nested fuse v13-selected + ms2s** — soft-prefer hold 0.5762 / OOF 0.5606; gate miss.
4. Fine weight grid around soft/gate region — **0 gate hits**.
5. Extra MidFuse / compact_fuse seeds — skipped (mf_avg already 3 seeds; s99 incomplete; agreement check does not justify GPU).
6. **No TTA**, **no sklearn/leaky stackers**.

## Protocol

1. Train `ms_stgcn_2s` 5-fold GroupKFold + holdout (`checkpoints_ms2s/`)
2. Nested fuse with v10/v11 branch logits + v13 s2s + ms2s (`v14_fuse.py`)
3. Select max nested OOF; evaluate holdout last
4. Test: 5-fold avg softmax → same recipe → `submission_v14*.csv` (candidate only)

## Files

- `model_ms_stgcn.py`, `train_ms2s.py`, `v14_fuse.py`
- `checkpoints_ms2s/` (fold0–4 + holdout)
- `oof_ms2s.npz`, `holdout_ms2s.npz`, `metrics_ms2s.json`
- `oof_stgcn2s.npz`, `holdout_stgcn2s.npz` (copies from v13 for fuse)
- `metrics.json`, `run_fuse.log`, `train_ms2s.log`
- `submission_v14.csv` / `_candidate.csv` / `_probs.npz` (selected max-OOF; **not** track)

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v14\train_ms2s.py --epochs 50 --batch-size 16 --lr 8e-4 --patience 12 --seed 42 --label-smoothing 0.05 --ckpt-dir baselines\v14\checkpoints_ms2s --metrics-out baselines\v14\metrics_ms2s.json --oof-out baselines\v14\oof_ms2s.npz --holdout-logits-out baselines\v14\holdout_ms2s.npz --cache-dir baselines\skeleton_imu_v2\cache

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v14\v14_fuse.py
```

≤100MB (ms2s ckpts ~50MB + logits). Nested GroupKFold discipline.
