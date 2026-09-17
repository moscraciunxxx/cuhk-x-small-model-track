# Small-Model-Track v8 — refined late-fuse (+ GRU branch)

## Summary vs v7

| | v7 late-fuse (MF+ST α=0.5) | **v8 selected** (`conf_weighted_3`) |
|---|---|---|
| OOF (non-holdout) | 0.5235 | **0.5383** (+0.0148) |
| Holdout | 0.5426 | **0.5564** (+0.0138) |
| Branches | MidFuse + ST-GCN | MidFuse + ST-GCN + **GRU-attn** |
| Fusion | global α on softmax | confidence-weighted 3-way (temp=4) |

Selection used **nested GroupKFold on non-holdout OOF only**; holdout evaluated last (no α/temp peek).

`submission.csv` **overwritten** (OOF ≥ v7+0.005 and holdout ≥ 0.55).  
`ping_disk_saver`: **true** (holdout 0.556 > 0.543 and OOF 0.538 > 0.53 with new CSV).

## Branches (unchanged checkpoints)

- **MidFuse v2b** (`skeleton_imu_v2/checkpoints_midfuse_v2b`): OOF 0.5033 / holdout 0.5366
- **ST-GCN fuse** (`v7_stgcn/checkpoints_cv` + holdout ckpt): OOF 0.4744 / holdout 0.4792
- **GRU-attn** (`skeleton_imu_v2/checkpoints_gru`): OOF 0.5008 / holdout 0.5030 — diverse (agree MF≈0.55, ST≈0.48)

No new large pretrained weights; stacker/fuse is sklearn on OOF probs only (≤100MB track).

## Method ablations (nested OOF → holdout)

| Method | Nested OOF | Holdout | Notes |
|---|---|---|---|
| alpha2 MF+ST (v7-style) | 0.5194 | 0.5426 | in-sample α pick 0.5235 |
| alpha2 MF+GRU | 0.5210 | 0.5584 | |
| **w3 equal MF+ST+GRU** | 0.5313 | **0.5624** | peek-best holdout; not selected |
| per-class α MF+ST | 0.5194 | 0.5446 | in-sample 0.5495 overfit |
| conf-gate MF+ST | 0.5243 | 0.5465 | |
| **conf_weighted_3 (selected)** | **0.5383** | **0.5564** | temp=4 on max-prob weights |
| stack logreg 2/3 | 0.46 / 0.45 | 0.49 / 0.47 | underperformed |
| stack MLP 2/3 | 0.49 / 0.51 | 0.53 / 0.53 | underperformed |

## Protocol

1. GroupKFold OOF logits for all three branches (cached `oof_logits.npz`).
2. Fit/select fusion on **non-holdout** only; nested CV for hyperparams.
3. Holdout logits from holdout-trained ckpts; apply selected fuse **once**.
4. Test: average 5-fold softmaxes per branch, then apply selected fuse → `submission_v8.csv`.

## Files

- `v8_refine_fuse.py` — pipeline
- `metrics.json` — full comparison
- `oof_logits.npz` — cached OOF
- `submission_v8.csv` / `submission_v8_probs.npz`
- Track root `submission.csv` updated to v8

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v8\v8_refine_fuse.py
```
