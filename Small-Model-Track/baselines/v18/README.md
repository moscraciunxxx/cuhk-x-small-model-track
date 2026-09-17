# Small-Model-Track v18 - KD alpha/seed/teacher diversity

## vs v17 (previous track best)

| | v17 selected | **v18 selected** |
|---|---:|---:|
| Nested OOF | 0.6636 | **0.6727** |
| Holdout | 0.6257 | **0.6317** |
| Fold ckpt size | ~83.5MB | **~83.5MB** |
| Clear-win overwrite | yes | **yes** |

Clear-win gate (v18): `holdout >= 0.636` **or** (`holdout >= 0.631` **and** nested OOF `>= 0.670`).  
`clear_win=True`. `overwrite_submission=True`. **`ping_disk_saver=true`**.

Selection = **max nested OOF among multi-branch KD blends with fold-ckpt sum <=100MB**; holdout evaluated last. No TTA. No leaky stackers.

## Selected method

`pow_kd_mf_v13_kd_T4_a02_kd_eq_a02` - nested power-mean (p=0.0 = geometric) of:

1. **kd_mf_v13** - MidFuseNet (~2.07M), teacher=v13, T=2, α=0.3, seed=42 (from v17)
2. **kd_T4_a02** - CompactMidFuse (~1.13M), teacher=v13, **T=4**, α=0.2, seed=42 (**new v18**)
3. **kd_eq_a02** - CompactMidFuse, teacher=**eq_best**, T=2, α=0.2, seed=42 (**new v18**)

Fold checkpoints for selected recipe ≈ **83.5MB** (≤100MB).

### Size note

Unconstrained max-nested blend was `eq_kd_c_kd_a02_kd_mf_v13_kd_mf_a02_kd_eq_a02` (nested **0.6764** / hold **0.6317**) but fold ckpts ≈**145MB**. Re-selected under ≤100MB budget; both clear the soft gate.

## What worked

### New KD diversity (v18 trained)

| Branch | Student | Teacher | T | α | seed | Nested OOF | Holdout |
|--------|---------|---------|--:|--:|-----:|----------:|--------:|
| kd_mf_a02 | midfuse | v13 | 2 | 0.2 | 42 | **0.6719** | **0.6277** |
| kd_a01 | compact | v13 | 2 | 0.1 | 42 | 0.6686 | 0.5921 |
| kd_a015 | compact | v13 | 2 | 0.15 | 42 | 0.6682 | 0.6099 |
| kd_T1_a02 | compact | v13 | 1 | 0.2 | 42 | 0.6678 | 0.5941 |
| kd_T4_a02 | compact | v13 | 4 | 0.2 | 42 | 0.6653 | **0.6198** |
| kd_a025 | compact | v13 | 2 | 0.25 | 42 | 0.6645 | 0.6020 |
| kd_a02s123 | compact | v13 | 2 | 0.2 | 123 | 0.6616 | 0.6020 |
| kd_a02s7 | compact | v13 | 2 | 0.2 | 7 | 0.6612 | 0.5921 |
| kd_eq_a02 | compact | eq_best | 2 | 0.2 | 42 | 0.6583 | **0.6257** |
| kd_c_v15a02 | compact | v15sel | 2 | 0.2 | 42 | 0.6575 | 0.6139 |
| kd_rich_a02 | compact | eq_kd_rich | 2 | 0.2 | 42 | 0.6397 | 0.6000 |

`kd_mf_a02` alone already exceeds the nested soft threshold (0.6719) with strong hold (0.6277).  
`kd_T4_a02` / `kd_eq_a02` add holdout-friendly diversity for blends.

### KD-only exhaustive blends

Lean fuse (`v18_fuse_kd.py`): eq / pow / conf over top-9 KD pool (size≤5 combos). Non-KD still hurts nested.  
Best hold among nested≥0.670: `pow_kd_mf_v13_kd_mf_a02_kd_eq_a02` hold **0.6416** nested 0.6715 (over size budget).  
Peek-best hold overall: `conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02` hold **0.6436** nested 0.6653 (below OOF soft).

## Protocol

1. Reuse v17 KD OOFs/checkpoints via junctions; train α/seed/T/teacher diversity
2. Nested KD-only fuse; select max nested among ≤100MB blends; holdout last
3. Test: 5-fold avg softmax → power-mean (p fit on full non-holdout) → submission
4. Overwrite track `submission.csv` + ping only on clear-win gate

## Files

- `train_kd.py`, `train_v18_batch.py`, `v18_fuse_kd.py`, `model.py`
- New: `checkpoints_kd_{a01,a015,a025,a02s7,a02s123,mf_a02,c_v15a02,eq_a02,T1_a02,T4_a02,rich_a02}/`
- Junctions to v17/v16 KD / s2s / mfp checkpoints + hardlinked OOFs
- `metrics.json`, `run_fuse_kd.log`, `submission_v18.csv` / `_candidate.csv` / `_probs.npz`
- Track `submission.csv` overwritten; `submission_README.txt` ping

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v18\train_v18_batch.py
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v18\v18_fuse_kd.py
# then size-constrained rebuild if unconstrained exceeds 100MB:
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v18\_rebuild_size.py
```

No TTA. Nested GroupKFold discipline. KD teachers from OOF logits only.
