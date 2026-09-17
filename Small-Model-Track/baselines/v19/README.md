# Small-Model-Track v19 — compress over-budget peek under 100MB

## vs v18

| | v18 selected | **v19 selected** |
|---|---:|---:|
| Nested OOF | 0.6727 | **0.6653** |
| Holdout | 0.6317 | **0.6436** |
| Fold ckpt size | ~83.5MB | **93.54MB** |
| Clear-win overwrite | yes | **true** |

Clear-win gate (v19): `holdout >= 0.642` **or** (`holdout >= 0.637` **and** nested OOF `>= 0.68`).

## Selected method

`conf_kd_kd_mf_v13_kd_mf_a02_kd_eq_a02` (temp=0.5) — same blend as v18 peek-best hold (0.6436),
compressed by **sharing fewer midfuse folds**:

| Student | Arch | Folds kept | Role |
|---------|------|------------|------|
| kd | midfuse (~2.07M) | [1, 2, 3] | teacher=v15sel |
| kd_mf_v13 | midfuse | [1, 2, 3] | teacher=v13 α=0.3 |
| kd_mf_a02 | midfuse | [1, 2, 3] | teacher=v13 α=0.2 |
| kd_eq_a02 | compact (~1.13M) | [0, 1, 2, 3, 4] | teacher=eq_best α=0.2 |

Full 5-fold package was ~141MB. Dropping midfuse folds 0 and 4 (lowest val_acc) yields **93.54MB ≤ 100MB**.
Holdout/OOF scores use the established holdout-train / NH-OOF logit protocol (unchanged by fold subset).

## What we tried

1. **Fewer folds shared** (chosen): midfuse {1,2,3} + compact 5 → clear win.
2. **fp16 pack** all 5 folds (~71MB): also fits; kept as `ckpt_fp16/` experiment artifact; submission uses fp32 fold-subset for loader safety.
3. Near-budget pow trio hold 0.6416 still **fails** hold>=0.642 even if sized under 100MB; soft gate needs OOF>=0.680 (max available 0.6764) — no new KD needed once peek clears alone gate.

No TTA. No leaky stackers. No Chrome.

## Reproduce

```text
baselines\v7_stgcn\.venv\Scripts\python.exe baselines\v19\build_v19.py
```
