# Small-Model-Track v16 — stronger KD + nested late-fuse

## vs v15 (previous track best)

| | v15 selected | **v16 selected** |
|---|---:|---:|
| Nested OOF | 0.5791 | **0.6521** |
| Holdout | 0.5881 | **0.6139** |
| Clear-win overwrite | yes | **yes** |

Clear-win gate (v16): `holdout >= 0.598` **or** (`holdout >= 0.593` **and** nested OOF `>= 0.585`).  
`clear_win=True`. `overwrite_submission=True`. **`ping_disk_saver=true`**.

Selection = **max nested OOF**; holdout evaluated last. No TTA. No leaky stackers.

## Selected method

`eq_kd_kd2` — equal softmax of:

1. **kd** — MidFuseNet student (~2.07M), teacher = nested **v15-selected** soft labels, T=2, α=0.3, seed=42  
2. **kd2** — CompactMidFuse (~1.13M), teacher = nested **v13-style** soft labels, T=2, α=0.3, seed=123  

## What worked

### Stronger KD (clear-win driver)

v15 KD used flat v13 soft labels, T=2, α=0.5 → OOF 0.6257 / hold 0.5881.

v16 changes:
- **Nested OOF-honest teachers** (GroupKFold power/eq fits; no holdout labels in teacher)
- **α=0.3** (more distill / less hard CE) — large smoke lift on fold0
- **Better teachers**: v15-selected blend and nested v13-style
- **Larger student** (MidFuseNet) + **second seed** compact for diversity
- Temperature grid smoke: T∈{2,3,4}, α∈{0.3,0.5}; winners promoted to full CV

| Branch | Student | Teacher | T | α | seed | OOF nh | Holdout |
|--------|---------|---------|--:|--:|-----:|-------:|--------:|
| kd | midfuse | v15sel | 2 | 0.3 | 42 | 0.6484 | 0.6119 |
| kd_c | compact | v13 nested | 2 | 0.3 | 42 | **0.6517** | **0.6178** |
| kd2 | compact | v13 nested | 2 | 0.3 | 123 | 0.6505 | 0.6059 |
| kd_alt | compact | v15sel | 3 | 0.3 | 42 | 0.6418 | 0.6059 |
| kdv15 (ref) | compact | v13 flat | 2 | 0.5 | 42 | 0.6257 | 0.5881 |

Peek-best hold among OOF-eligible: `solo_kd_c` (hold 0.6178) — not selected (lower nested OOF than `eq_kd_kd2`).

### MidFusePlus

Reused v15 fixed MFP (CV ~0.487). Useful diversity in the search grid; not in the selected pair.

## Protocol

1. Fold0 smoke grid over teacher × student × T × α  
2. Full GroupKFold + holdout for top configs (+ second seed)  
3. Nested fuse with v10/v11 + s2s/ms2s/mfp + all KD variants (`v16_fuse.py`)  
4. Select max nested OOF; evaluate holdout last  
5. Test: 5-fold avg softmax → recipe → submission  

## Key ablations (nested → holdout)

| Method | Nested OOF | Holdout |
|---|---|---|
| v15 selected | 0.5791 | 0.5881 |
| solo_kd (midfuse) | 0.6484 | 0.6119 |
| solo_kd_c | 0.6517 | 0.6178 |
| **v16 selected eq_kd_kd2** | **0.6521** | **0.6139** |

## Files

- `train_kd.py` — multi-teacher / T / α / seed KD trainer  
- `checkpoints_kd/` (midfuse), `checkpoints_kd2/` (compact s123), `checkpoints_kd_c/`, `checkpoints_kd_alt/`  
- `oof_*.npz` / `holdout_*.npz` / `metrics_kd*.json`  
- `v16_fuse.py`, `metrics.json`, `run_fuse.log`  
- `submission_v16.csv` / `_candidate.csv` / `_probs.npz` (also overwrote track `submission.csv`)  
- Junctions: `checkpoints_mfp` → v15, `checkpoints_stgcn2s` → v13, `checkpoints_ms2s` → v14  

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v16\train_kd.py --teacher v15sel --student midfuse --T 2 --alpha 0.3 --seed 42 --ckpt-dir baselines\v16\checkpoints_kd --metrics-out baselines\v16\metrics_kd.json --oof-out baselines\v16\oof_kd.npz --holdout-logits-out baselines\v16\holdout_kd.npz --oof-key kd

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v16\train_kd.py --teacher v13 --student compact --T 2 --alpha 0.3 --seed 123 --ckpt-dir baselines\v16\checkpoints_kd2 --metrics-out baselines\v16\metrics_kd2.json --oof-out baselines\v16\oof_kd2.npz --holdout-logits-out baselines\v16\holdout_kd2.npz --oof-key kd2

baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v16\v16_fuse.py
```

No TTA. Nested GroupKFold discipline. KD teacher soft labels from OOF logits only.
