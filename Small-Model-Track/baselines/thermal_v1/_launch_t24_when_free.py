"""Safe T24 retrain launcher: preserve best partial, only then start CUDA train.
Run ONLY when HELIOS gone and GPU idle.
"""
from __future__ import annotations
import json, shutil, subprocess, sys, time
from pathlib import Path
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    def now_pt():
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
except Exception:
    def now_pt():
        return time.strftime("%Y-%m-%d %H:%M:%S")

ROOT = Path(__file__).resolve().parent
CK = ROOT / "checkpoints" / "ir_yolo_r2p1d18_t24_v24"
PARTIAL = CK / "pool_seed42_partial_ep003_val06455.pt"
ACTIVE = CK / "pool_seed42.pt"
ASIDE = CK / "pool_seed42_aside_before_full_retrain.pt"
STATUS = ROOT / "metrics_ir_v24_status.json"
LOG = ROOT / "logs" / "train_ir_t24_v24_seed42_full.log"
ERR = ROOT / "logs" / "train_ir_t24_v24_seed42_full.err.log"

def gpu_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().splitlines()[0]
        return int(float(out))
    except Exception:
        return -1

def helios_train_running():
    """True only for actual HELIOS/hotmoe train, not _watch_helios_*."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % CommandLine"],
            text=True, timeout=20, stderr=subprocess.DEVNULL)
    except Exception:
        return True  # fail closed
    for ln in out.splitlines():
        low = ln.lower()
        if "_watch_helios" in low or "watch_helios" in low:
            continue
        if "run_helios" in low or "helios_val" in low or "hotmoe" in low:
            return True
        if "helios" in low and ("--execute" in low or "train" in low):
            return True
    return False

def main():
    used = gpu_used()
    hel = helios_train_running()
    print(f"precheck gpu_used_mib={used} helios_train={hel} at={now_pt()}", flush=True)
    if hel or not (0 <= used < 400):
        print("ABORT: HELIOS train still present or GPU not idle — stay off CUDA", flush=True)
        return 2
    CK.mkdir(parents=True, exist_ok=True)
    if not PARTIAL.exists() and ACTIVE.exists():
        shutil.copy2(ACTIVE, PARTIAL)
        print(f"copied active -> {PARTIAL.name}", flush=True)
    if ACTIVE.exists():
        shutil.copy2(ACTIVE, ASIDE)
        ACTIVE.unlink()
        print(f"moved {ACTIVE.name} aside -> {ASIDE.name} (train skips if exists)", flush=True)
    s = json.loads(STATUS.read_text(encoding="utf-8-sig"))
    s["outcome"] = "TRAINING_T24_SEED42"
    s["angles"]["3_more_frames_t24"]["result"] = "TRAINING_FULL"
    s["gpu_handoff"] = {"status": "TRAINING", "at": now_pt(), "note": "HELIOS full done; safe launcher full retrain seed42"}
    s["updated_at"] = now_pt()
    s["waiting_for"] = {"helios_gone": False, "gpu_free": True}
    STATUS.write_text(json.dumps(s, indent=2), encoding="utf-8")
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    cmd = [str(py), "-u", "train_ir_r2p1d18_tmore_v24.py", "--t", "24", "--seeds", "42"]
    print("launch", cmd, flush=True)
    with open(LOG, "w", encoding="utf-8") as lo, open(ERR, "w", encoding="utf-8") as le:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lo, stderr=le)
    print(f"PID={p.pid} log={LOG}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
