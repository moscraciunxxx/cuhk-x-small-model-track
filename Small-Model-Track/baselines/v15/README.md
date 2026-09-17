# Small-Model-Track v15 — MidFusePlus fix + KD student + nested late-fuse

## vs v13 (track best before)

| | v13 selected | **v15 selected** |
|---|---:|---:|
| Nested OOF | 0.5631 | **0.5791** |
| Holdout | 0.5703 | **0.5881** |
| Clear-win overwrite | yes | **yes** |

Clear-win gate (v15): `holdout > 0.5703` **and** nested OOF `≥ 0.560` (or holdout `≥ 0.580`).  
`clear_win=True`. `overwrite_submission=True`. **`ping_disk_saver=true`**.

Selection = **max nested OOF**; holdout evaluated last. No TTA. No leaky stackers.

## Selected method

`v11style211_plus_eq_kd_s2s_x3.0`

- Classic ABG (wa,wb,wg)=(2,1,1) over A/B/G
- Plus `eq_kd_s2s` = equal softmax of (**KD compact student**, **stgcn_2s**) at weight **3.0**

## What worked

### 1. Knowledge distill (clear-win driver)

CompactMidFuse student (~1.13M) trained with GroupKFold on **OOF-honest** soft labels from a v13-style ensemble mix (mf_avg / stgcn / gru / cfuse / s2s), loss = 0.5·CE + 0.5·T²·KL (T=2).

| Metric | KD compact |
|--------|----------:|
| Params | 1,134,152 |
| CV mean±std | **0.6234 ± 0.044** |
| OOF non-holdout | **0.6257** |
| Holdout alone | **0.5881** |

Smoke fold0 best **0.559** before full CV.

### 2. MidFusePlus fix (no clear-win alone)

v13/v14 MidFusePlus (~3.2M MultiScale IMU) smoke fold0 ~0.30 vs MidFuse v2b ~0.45 — broken.

**Root causes fixed:**
- Dropped MultiScale IMU; keep exact v2b `TemporalConvBranch` IMU
- Add skeleton **velocity** channels only (51→102); params **2.10M** (v2b 2.07M)
- Match v2b training: `balanced_sampler=False`, lr=1e-3, bs=48, epochs=50, patience=12

| Metric | MidFusePlus v15 | MidFuse v2b |
|--------|----------------:|------------:|
| Params | 2,096,552 | 2,072,072 |
| CV mean±std | 0.4874 ± 0.060 | 0.5051 |
| OOF non-holdout | 0.4847 | — |
| Holdout alone | 0.5089 | 0.5366 |

Smoke fold0 best **0.439** (near v2b 0.455). Useful diversity but nested fuse with MFP alone did **not** beat v13 clear-win (same soft-prefer as v14: hold 0.5762 / OOF 0.5606).

## Ideas tried / not promoted as selected

1. MidFusePlus velocity — trained full CV; fused as `mfw`; no selected clear-win without KD.
2. Blind re-blend of ms2s/s2s (v14 repeat) — no clear-win.
3. MidFuseWide path — not repeated (overfit).

## Protocol

1. Fix+smoke MidFusePlus → full GroupKFold + holdout (`checkpoints_mfp/`)
2. KD compact student full GroupKFold + holdout (`checkpoints_kd/`)
3. Nested fuse with v10/v11 branches + s2s + ms2s + mfp + **kd** (`v15_fuse.py`)
4. Select max nested OOF; evaluate holdout last
5. Test: 5-fold avg softmax → same recipe → submission

## Key ablations (nested → holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| v13 selected | 0.5631 | 0.5703 |
| v14 max-OOF (no overwrite) | 0.5635 | 0.5663 |
| soft-prefer ms (not selected) | 0.5606 | 0.5762 |
| **v15 selected (KD)** | **0.5791** | **0.5881** |

## Files

- `model.py` — MidFusePlus (velocity + v2b branches)
- `train_mfp.py`, `checkpoints_mfp/`, `oof_mfp.npz`, `holdout_mfp.npz`, `metrics_mfp.json`
- `train_kd.py`, `checkpoints_kd/`, `oof_kd.npz`, `holdout_kd.npz`, `metrics_kd.json`
- `v15_fuse.py`, `metrics.json`, `run_fuse_kd.log`
- `submission_v15.csv` / `_candidate.csv` / `_probs.npz` (also overwrote track `submission.csv`)

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v15\train_mfp.py --epochs 50 --patience 12 --no-balanced-sampler --ckpt-dir baselines\v15\checkpoints_mfp --metrics-out baselines\v15\metrics_mfp.json --oof-out baselines\v15\oof_mfp.npz --holdout-logits-out baselines\v15\holdout_mfp.npz

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v15\train_kd.py --student compact --epochs 45 --patience 12 --ckpt-dir baselines\v15\checkpoints_kd --metrics-out baselines\v15\metrics_kd.json --oof-out baselines\v15\oof_kd.npz --holdout-logits-out baselines\v15\holdout_kd.npz

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v15\v15_fuse.py
```

No TTA. Nested GroupKFold discipline. KD teacher soft labels from OOF logits only.
