"""Batch-train v18 KD diversity students sequentially."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT.parent / "v7_stgcn" / ".venv" / "Scripts" / "python.exe"
TRAIN = ROOT / "train_kd.py"

# (oof_key, student, teacher, T, alpha, seed)
JOBS = [
    ("kd_a01", "compact", "v13", 2.0, 0.1, 42),
    ("kd_a015", "compact", "v13", 2.0, 0.15, 42),
    ("kd_a025", "compact", "v13", 2.0, 0.25, 42),
    ("kd_a02s7", "compact", "v13", 2.0, 0.2, 7),
    ("kd_a02s123", "compact", "v13", 2.0, 0.2, 123),
    ("kd_mf_a02", "midfuse", "v13", 2.0, 0.2, 42),
    ("kd_c_v15a02", "compact", "v15sel", 2.0, 0.2, 42),
    ("kd_eq_a02", "compact", "eq_best", 2.0, 0.2, 42),
    ("kd_T1_a02", "compact", "v13", 1.0, 0.2, 42),
    ("kd_T4_a02", "compact", "v13", 4.0, 0.2, 42),
    ("kd_rich_a02", "compact", "eq_kd_rich", 2.0, 0.2, 42),
]

def main():
    track = ROOT.parent.parent
    for key, student, teacher, T, alpha, seed in JOBS:
        oof = ROOT / f"oof_{key}.npz"
        hold = ROOT / f"holdout_{key}.npz"
        metrics = ROOT / f"metrics_{key}.json"
        ckpt = ROOT / f"checkpoints_{key}"
        log = ROOT / f"train_{key}.log"
        if oof.exists() and hold.exists() and metrics.exists() and (ckpt / "best_fold0.pt").exists():
            print(f"SKIP {key} (already trained)", flush=True)
            continue
        cmd = [
            str(PY), str(TRAIN),
            "--student", student,
            "--teacher", teacher,
            "--T", str(T),
            "--alpha", str(alpha),
            "--seed", str(seed),
            "--ckpt-dir", str(ckpt),
            "--metrics-out", str(metrics),
            "--oof-out", str(oof),
            "--holdout-logits-out", str(hold),
            "--oof-key", key,
        ]
        print(f"=== TRAIN {key} ===", " ".join(cmd), flush=True)
        with open(log, "w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, cwd=str(track), stdout=lf, stderr=subprocess.STDOUT)
        print(f"=== DONE {key} rc={p.returncode} log={log} ===", flush=True)
        if p.returncode != 0:
            print(f"FAIL {key}", flush=True)
            # continue to next; fuse will skip missing
    print("ALL JOBS FINISHED", flush=True)

if __name__ == "__main__":
    main()
