import subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
py = str(ROOT / ".venv" / "Scripts" / "python.exe")
log = ROOT / "logs" / "train_cv.log"
err = ROOT / "logs" / "train_cv.err"

def run(cmd, logpath, errpath):
    print(f"RUN {cmd}", flush=True)
    with open(logpath, "a", encoding="utf-8") as lo, open(errpath, "a", encoding="utf-8") as er:
        lo.write(f"\n==== {time.ctime()} {' '.join(cmd)} ====\n")
        er.write(f"\n==== {time.ctime()} {' '.join(cmd)} ====\n")
        p = subprocess.run(cmd, stdout=lo, stderr=er, cwd=str(ROOT))
    print(f"exit={p.returncode}", flush=True)
    return p.returncode

# Full GroupKFold CV + holdout + all_train
rc = run([
    py, "train.py",
    "--mode", "cv",
    "--cv-splits", "5",
    "--epochs", "40",
    "--model", "midfuse",
    "--batch-size", "40",
    "--seed", "42",
    "--patience", "12",
    "--lr", "1e-3",
], log, err)

# Infer if train ok
if rc == 0:
    run([
        py, "infer.py",
        "--ckpt", str(ROOT / "checkpoints" / "best.pt"),
        "--out", str(ROOT / "submission_skeleton_imu_v2.csv"),
    ], ROOT / "logs" / "infer.log", ROOT / "logs" / "infer.err")
else:
    # fallback: skeleton-only deepconv
    print("midfuse CV failed — trying deepconv skeleton-only", flush=True)
    rc2 = run([
        py, "train.py",
        "--mode", "cv",
        "--cv-splits", "5",
        "--epochs", "40",
        "--model", "deepconv",
        "--skeleton-only",
        "--batch-size", "48",
        "--seed", "42",
        "--patience", "12",
        "--metrics-out", str(ROOT / "metrics_skel_only.json"),
    ], ROOT / "logs" / "train_skel.log", ROOT / "logs" / "train_skel.err")
    if rc2 == 0:
        run([
            py, "infer.py",
            "--ckpt", str(ROOT / "checkpoints" / "best.pt"),
            "--out", str(ROOT / "submission_skeleton_v2.csv"),
        ], ROOT / "logs" / "infer_skel.log", ROOT / "logs" / "infer_skel.err")

# summarize
import json
mp = ROOT / "metrics.json"
if mp.exists():
    m = json.load(open(mp, encoding="utf-8"))
    print("SUMMARY mean", m.get("mean_val_acc"), "std", m.get("std_val_acc"),
          "holdout", (m.get("holdout") or {}).get("best_val_acc"), flush=True)
print("DONE", time.ctime(), flush=True)
