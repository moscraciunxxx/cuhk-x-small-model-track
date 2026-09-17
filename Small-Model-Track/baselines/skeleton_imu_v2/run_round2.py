"""Round-2: stronger honest CV. No balanced sampler (was hurting). Try deepconv + midfuse + gru."""
import subprocess, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
py = str(ROOT / ".venv" / "Scripts" / "python.exe")

def run(cmd, logname):
    log = ROOT / "logs" / logname
    err = ROOT / "logs" / (logname + ".err")
    print(f"RUN {cmd}", flush=True)
    with open(log, "w", encoding="utf-8") as lo, open(err, "w", encoding="utf-8") as er:
        lo.write(f"==== {time.ctime()} ====\n")
        p = subprocess.run(cmd, stdout=lo, stderr=er, cwd=str(ROOT))
    print(f"exit={p.returncode} log={log}", flush=True)
    return p.returncode

# 1) deepconv skeleton-only, no balanced sampler, longer patience
rc1 = run([
    py, "train.py", "--mode", "cv", "--cv-splits", "5", "--epochs", "50",
    "--model", "deepconv", "--skeleton-only", "--batch-size", "48",
    "--seed", "42", "--patience", "15", "--no-balanced-sampler",
    "--label-smoothing", "0.05", "--lr", "8e-4",
    "--metrics-out", str(ROOT / "metrics_deepconv.json"),
    "--ckpt-dir", str(ROOT / "checkpoints_deepconv"),
], "train_deepconv.log")

# 2) midfuse without balanced sampler
rc2 = run([
    py, "train.py", "--mode", "cv", "--cv-splits", "5", "--epochs", "50",
    "--model", "midfuse", "--batch-size", "40",
    "--seed", "42", "--patience", "15", "--no-balanced-sampler",
    "--label-smoothing", "0.05", "--lr", "8e-4",
    "--metrics-out", str(ROOT / "metrics_midfuse_v2b.json"),
    "--ckpt-dir", str(ROOT / "checkpoints_midfuse_v2b"),
], "train_midfuse_v2b.log")

# 3) gru_attn skeleton-only
rc3 = run([
    py, "train.py", "--mode", "cv", "--cv-splits", "5", "--epochs", "50",
    "--model", "gru_attn", "--skeleton-only", "--batch-size", "48",
    "--seed", "42", "--patience", "15", "--no-balanced-sampler",
    "--label-smoothing", "0.05", "--lr", "1e-3",
    "--metrics-out", str(ROOT / "metrics_gru.json"),
    "--ckpt-dir", str(ROOT / "checkpoints_gru"),
], "train_gru.log")

# Pick best by mean_val_acc (and prefer holdout as tie-break)
cands = []
for name, path, ckpt_dir, dual_name in [
    ("midfuse_v1", ROOT/"metrics.json", ROOT/"checkpoints", "submission_skeleton_imu_v2.csv"),
    ("deepconv", ROOT/"metrics_deepconv.json", ROOT/"checkpoints_deepconv", "submission_skeleton_v2.csv"),
    ("midfuse_v2b", ROOT/"metrics_midfuse_v2b.json", ROOT/"checkpoints_midfuse_v2b", "submission_skeleton_imu_v2b.csv"),
    ("gru", ROOT/"metrics_gru.json", ROOT/"checkpoints_gru", "submission_skeleton_v2_gru.csv"),
]:
    if path.exists():
        m = json.load(open(path, encoding="utf-8"))
        hold = (m.get("holdout") or {}).get("best_val_acc", 0)
        cands.append((m.get("mean_val_acc", 0), hold, name, path, ckpt_dir, dual_name, m))

cands.sort(reverse=True)
print("CANDIDATES:", [(c[2], c[0], c[1]) for c in cands], flush=True)
best = cands[0]
mean_acc, hold, name, path, ckpt_dir, out_name, m = best
print(f"BEST={name} cv={mean_acc:.4f} holdout={hold:.4f}", flush=True)

# copy best ckpt to main checkpoints/best.pt and metrics.json
import shutil
src = Path(m.get("primary_ckpt") or (ckpt_dir / "best.pt"))
dst = ROOT / "checkpoints" / "best.pt"
dst.parent.mkdir(exist_ok=True)
if src.exists():
    shutil.copy2(src, dst)
shutil.copy2(path, ROOT / "metrics.json")

# infer with best
out_csv = ROOT / ("submission_skeleton_imu_v2.csv" if "midfuse" in name else "submission_skeleton_v2.csv")
run([
    py, "infer.py", "--ckpt", str(dst), "--out", str(out_csv),
], "infer_best.log")

# also write comparison summary
summary = {
    "candidates": [
        {"name": c[2], "mean_val_acc": c[0], "holdout": c[1]} for c in cands
    ],
    "selected": name,
    "mean_val_acc": mean_acc,
    "holdout": hold,
    "submission": str(out_csv),
    "vs_v1_holdout_0.479": {
        "cv_delta": mean_acc - 0.479,
        "holdout_delta": hold - 0.479,
    },
}
json.dump(summary, open(ROOT / "comparison.json", "w"), indent=2)
print("SUMMARY", summary, flush=True)
print("DONE", time.ctime(), flush=True)
