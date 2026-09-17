# Small-Model-Track v17 — KD multi-blend + diversity

## vs v16 (previous track best)

| | v16 selected | **v17 selected** |
|---|---:|---:|
| Nested OOF | 0.6521 | **0.6636** |
| Holdout | 0.6139 | **0.6257** |
| Clear-win overwrite | yes | **yes** |

Clear-win gate (v17): `holdout >= 0.624` **or** (`holdout >= 0.619` **and** nested OOF `>= 0.660`).  
`clear_win=True`. `overwrite_submission=True`. **`ping_disk_saver=true`**.

Selection = **max nested OOF among multi-branch blends** (solos kept for ablation/peek); holdout evaluated last. No TTA. No leaky stackers.

## Selected method

`eq_kd_kd_c_kd_a02` — equal softmax of:

1. **kd** — MidFuseNet (~2.07M), teacher = nested v15-selected soft labels, T=2, α=0.3, seed=42 (from v16)
2. **kd_c** — CompactMidFuse (~1.13M), teacher = nested v13-style, T=2, α=0.3, seed=42 (from v16)
3. **kd_a02** — CompactMidFuse, teacher = nested v13-style, T=2, **α=0.2**, seed=42 (**new v17**)

Fold checkpoints for selected recipe ≈ **83.5MB** (≤100MB).

## What worked

### Exhaustive KD multi-blends (v16 gap)

v16 trained `kd` / `kd_c` / `kd2` / `kd_alt` but only fused `eq_kd_kd2`. Expanding to all equal / power / conf-weighted combinations of the KD pool recovers clear-win territory (e.g. `eq_kd_kd_c_kd_alt` nested≈0.6616 hold≈0.6277).

### α=0.2 distill student

`kd_a02` alone: nested OOF **0.6694** / hold 0.6020 (large OOF–hold gap → not selected as solo).  
Equal blend with `kd`+`kd_c` keeps nested **0.6636** and lifts holdout to **0.6257** (clears hold-alone gate).

### Other diversity (ablations)

| Branch | Student | Teacher | T | α | seed | Nested OOF | Holdout |
|--------|---------|---------|--:|--:|-----:|----------:|--------:|
| kd3 | compact | v13 | 2 | 0.3 | 7 | 0.6546 | 0.6040 |
| kd_a02 | compact | v13 | 2 | 0.2 | 42 | **0.6694** | 0.6020 |
| kd_mf_v13 | midfuse | v13 | 2 | 0.3 | 42 | 0.6608 | 0.6158 |
| kd (v16) | midfuse | v15sel | 2 | 0.3 | 42 | 0.6484 | 0.6119 |
| kd_c (v16) | compact | v13 | 2 | 0.3 | 42 | 0.6517 | 0.6178 |

Limited triples `eq_kd_kd_c_{kd3,a02,mf_v13}` entered the blend search; `eq_kd_kd_c_kd_a02` won nested among blends.

Non-KD fusion with s2s/mfp still hurts nested OOF (agreement ~0.55–0.60 but weak absolute accuracy) — not in selected recipe.

Peek-best hold among OOF-eligible: `pow_kd_kd_c_kd_alt` (hold 0.6317, nested 0.6612) — not selected (lower nested than winner).

## Protocol

1. Reuse v16 KD OOFs/checkpoints; train seed3 / α=0.2 / MidFuse×v13
2. Nested fuse: exhaustive multi-blends on `{kd,kd2,kd_c,kd_alt}` + conf-weighted + limited new triples (`v17_fuse.py`)
3. Select max nested OOF among **blends**; evaluate holdout last
4. Test: 5-fold avg softmax → equal blend → submission
5. Overwrite track `submission.csv` + ping only on clear-win gate

## Files

- `train_kd.py`, `v17_fuse.py`, `model.py`
- `checkpoints_kd3/`, `checkpoints_kd_a02/`, `checkpoints_kd_mf_v13/` (+ junctions to v16 kd/kd2/kd_c/kd_alt)
- `oof_*.npz` / `holdout_*.npz` / `metrics_kd*.json`
- `metrics.json`, `run_fuse.log`, `quick_eval.json`
- `submission_v17.csv` / `_candidate.csv` / `_probs.npz` (also overwrote track `submission.csv`)

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v17\train_kd.py --teacher v13 --student compact --T 2 --alpha 0.3 --seed 7 --ckpt-dir baselines\v17\checkpoints_kd3 --metrics-out baselines\v17\metrics_kd3.json --oof-out baselines\v17\oof_kd3.npz --holdout-logits-out baselines\v17\holdout_kd3.npz --oof-key kd3

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v17\train_kd.py --teacher v13 --student compact --T 2 --alpha 0.2 --seed 42 --ckpt-dir baselines\v17\checkpoints_kd_a02 --metrics-out baselines\v17\metrics_kd_a02.json --oof-out baselines\v17\oof_kd_a02.npz --holdout-logits-out baselines\v17\holdout_kd_a02.npz --oof-key kd_a02

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v17\train_kd.py --teacher v13 --student midfuse --T 2 --alpha 0.3 --seed 42 --ckpt-dir baselines\v17\checkpoints_kd_mf_v13 --metrics-out baselines\v17\metrics_kd_mf_v13.json --oof-out baselines\v17\oof_kd_mf_v13.npz --holdout-logits-out baselines\v17\holdout_kd_mf_v13.npz --oof-key kd_mf_v13

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v17\v17_fuse.py
```

No TTA. Nested GroupKFold discipline. KD teachers from OOF logits only.
