"""Quiet overnight v4 orchestrator. Log only; no user messaging."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "overnight_v4.log"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, cwd=None):
    log(f"RUN: {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd or str(ROOT), capture_output=False)
    log(f"EXIT {p.returncode}: {' '.join(cmd)}")
    return p.returncode


def main():
    log("=== v4 overnight start ===")
    log(f"python={PY}")

    run([PY, "build_bone_cache.py"], cwd=str(ROOT))

    rc = run([
        PY, "train_bone.py",
        "--mode", "cv", "--model", "bonemidfuse",
        "--epochs", "50", "--batch-size", "32", "--lr", "8e-4",
        "--patience", "15", "--seed", "42",
        "--ckpt-dir", str(ROOT / "checkpoints_bone"),
        "--metrics-out", str(ROOT / "metrics_bone.json"),
    ])
    log(f"bone CV rc={rc}")

    bone_m = ROOT / "metrics_bone.json"
    try_triple = False
    if bone_m.exists():
        m = json.loads(bone_m.read_text(encoding="utf-8"))
        hold = (m.get("holdout") or {}).get("best_val_acc") or 0
        cv = m.get("mean_val_acc") or 0
        log(f"bone results cv={cv} holdout={hold}")
        if hold >= 0.52 or cv >= 0.50:
            try_triple = True
    if try_triple:
        Path(ROOT / "checkpoints_triple").mkdir(parents=True, exist_ok=True)
        run([
            PY, "train_bone.py",
            "--mode", "holdout", "--model", "triple",
            "--epochs", "50", "--batch-size", "24", "--lr", "8e-4",
            "--patience", "15", "--seed", "42",
            "--ckpt-dir", str(ROOT / "checkpoints_triple"),
            "--metrics-out", str(ROOT / "metrics_triple_holdout.json"),
        ])

    Path(ROOT / "checkpoints_seeds_holdout").mkdir(parents=True, exist_ok=True)
    run([
        PY, "train_seeds.py",
        "--mode", "holdout",
        "--seeds", "42", "123", "7",
        "--epochs", "50", "--batch-size", "32", "--lr", "8e-4",
        "--patience", "15",
        "--ckpt-dir", str(ROOT / "checkpoints_seeds_holdout"),
        "--metrics-out", str(ROOT / "metrics_seeds_holdout.json"),
    ])
    run([
        PY, "train_seeds.py",
        "--mode", "full",
        "--seeds", "42", "123", "7",
        "--epochs", "50", "--batch-size", "32", "--lr", "8e-4",
        "--patience", "15",
        "--ckpt-dir", str(ROOT / "checkpoints_seeds"),
        "--metrics-out", str(ROOT / "metrics_seeds.json"),
    ])

    run([
        PY, "train_thermal.py",
        "--epochs", "20", "--batch-size", "12", "--patience", "7",
    ])

    run([PY, "ensemble_v4.py"])
    run([PY, "summarize_v4.py"])
    log("=== v4 overnight done ===")


if __name__ == "__main__":
    main()
