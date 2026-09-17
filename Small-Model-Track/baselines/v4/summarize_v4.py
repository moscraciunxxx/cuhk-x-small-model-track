"""Write final metrics.json + README for v4."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))

def load(p):
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

bone = load(ROOT / "metrics_bone.json") or {}
seeds_h = load(ROOT / "metrics_seeds_holdout.json") or {}
seeds = load(ROOT / "metrics_seeds.json") or {}
thermal = load(ROOT / "metrics_thermal.json") or {}
ens = load(ROOT / "metrics_ensemble.json") or {}
triple = load(ROOT / "metrics_triple_holdout.json") or {}

v2_hold = 0.5366336633663367
v3_oof = 0.5015

bone_hold = (bone.get("holdout") or {}).get("best_val_acc")
bone_cv = bone.get("mean_val_acc")
bone_oof = (ens.get("bone_metrics") or {}).get("oof_accuracy")
th_hold = thermal.get("holdout_gated_thermal")
th_base = thermal.get("holdout_midfuse_baseline")
seed_hold_mean = seeds_h.get("mean_monitor_val")

beats = []
fails = []
if bone_hold is not None:
    (beats if bone_hold > v2_hold else fails).append(f"bone holdout {bone_hold:.4f} vs {v2_hold:.4f}")
if bone_cv is not None:
    (beats if bone_cv > 0.505 else fails).append(f"bone CV {bone_cv:.4f} vs v2 CV 0.505")
if bone_oof is not None:
    (beats if bone_oof > v3_oof else fails).append(f"bone OOF {bone_oof:.4f} vs v3 OOF {v3_oof:.4f}")
if th_hold is not None and th_base is not None:
    (beats if th_hold > th_base else fails).append(f"thermal {th_hold:.4f} vs midfuse {th_base:.4f}")
if seed_hold_mean is not None:
    (beats if seed_hold_mean > v2_hold else fails).append(f"seed-bag holdout-mean {seed_hold_mean:.4f}")

holdout_best = max([x for x in [bone_hold, th_hold, seed_hold_mean,
                                (triple.get("holdout") or triple.get("mean_val_acc"))] if x is not None] or [0])

primary_sub = "submission_v4_best.csv"
if not (ROOT / primary_sub).exists():
    for alt in ["submission_v4_full_mix.csv", "submission_v2folds_seeds.csv", "submission_bone_folds.csv"]:
        if (ROOT / alt).exists():
            primary_sub = alt
            break

metrics = {
    "track": "CUHK-X Small Model Track v4",
    "finished_at_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
    "baselines": {"v2_holdout": v2_hold, "v2_cv": 0.5050853925069481, "v3_oof": v3_oof},
    "bone_midfuse": {
        "cv_mean": bone_cv,
        "cv_std": bone.get("std_val_acc"),
        "holdout": bone_hold,
        "oof": bone_oof,
        "n_params": (bone.get("all_train") or bone.get("holdout") or {}).get("n_params"),
    },
    "triple_holdout": triple.get("mean_val_acc") or (triple.get("holdout") or {}).get("best_val_acc"),
    "seed_bag": {
        "holdout_probe_mean": seed_hold_mean,
        "holdout_per_seed": [r.get("best_val_acc") for r in (seeds_h.get("results") or [])],
        "full_monitor_mean": seeds.get("mean_monitor_val"),
        "seeds": seeds.get("seeds") or seeds_h.get("seeds"),
    },
    "gated_thermal": thermal,
    "ensemble": ens,
    "holdout_best_observed": holdout_best,
    "beats_v2_holdout_0537": bool(holdout_best > v2_hold),
    "beats": beats,
    "fails": fails,
    "primary_submission": str(ROOT / primary_sub),
    "recommendation": (
        "Ping Disk Saver / submit if holdout_best > 0.537; else keep v3_ensemble as primary submit."
        if holdout_best > v2_hold else
        "No clear holdout beat of 0.537; keep submission_v3_ensemble.csv; v4 may still diversify."
    ),
}
(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

readme = f"""# CUHK-X Small Model Track — v4

Quiet overnight on MSI. Goal: beat MidFuse v2 holdout **0.537** and/or improve OOF vs v3 **0.5015**.

## Experiments

| Exp | Metric | Result | vs baseline |
|-----|--------|--------|-------------|
| BoneMidFuse (GroupKFold) | CV mean | {bone_cv} | v2 CV 0.505 |
| BoneMidFuse | Holdout {{8,9,24}} | {bone_hold} | v2 **0.537** |
| BoneMidFuse | OOF | {bone_oof} | v3 OOF 0.5015 |
| Multi-seed MidFuse (3 seeds) holdout probe mean | | {seed_hold_mean} | 0.537 |
| Gated Thermal holdout | | {th_hold} (base {th_base}) | — |

**Beats holdout 0.537?** `{bool(holdout_best > v2_hold)}` (best observed={holdout_best})

## What worked / failed
- Beats: {beats if beats else "none"}
- Fails: {fails if fails else "none"}

## Primary submission
`{primary_sub}`

Also see `submission_*.csv` variants and `metrics_ensemble.json`.

## Reproduce
```powershell
cd D:\\CUHK-X\\Small-Model-Track\\baselines\\v4
.\\.venv\\Scripts\\Activate.ps1
python build_bone_cache.py
python train_bone.py --mode cv --model bonemidfuse --epochs 50 --batch-size 32 --no-balanced-sampler
python train_seeds.py --mode full --seeds 42 123 7
python train_thermal.py
python ensemble_v4.py
python summarize_v4.py
```

## Notes
- CNN only; BoneMidFuse ~same size as MidFuse; TinyThermal ≪10MB.
- Radar skipped. Thermal gated; empty Thermal → MidFuse logits.
- GPU shared overnight with other jobs; batch sizes kept modest.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print(json.dumps(metrics, indent=2))
