"""Wait for HELIOS/ViPT gone + GPU free, then train T24 focal_ft seeds 2024 888 (reuse 42 in ens)."""
from __future__ import annotations
import json, subprocess, sys, time
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
LOG = ROOT / "logs" / "train_focal_ft_t24_s2024_888.log"
ERR = ROOT / "logs" / "train_focal_ft_t24_s2024_888.err"
PIDF = ROOT / "logs" / "train_focal_ft_t24_s2024_888.pid"
STATUS = ROOT / "metrics_ir_v25_status.json"
CK = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
CACHE = ROOT / "cache" / "ir_yolo_ft_v24_t24"

def gpu_used():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().splitlines()[0]
        return int(float(out))
    except Exception:
        return -1

def blocking_procs():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % { $_.ProcessId.ToString() + '|' + $_.CommandLine }"],
            text=True, timeout=25, stderr=subprocess.DEVNULL)
    except Exception as e:
        return [f"check_failed:{e}"]
    bad = []
    for ln in out.splitlines():
        low = ln.lower()
        if "_watch_helios" in low or "watch_helios" in low:
            continue
        if "_wait_gpu_launch" in low or "wait_gpu_launch_t24" in low:
            continue
        if "run_helios" in low or "helios_val" in low or "hotmoe" in low or "vipt" in low:
            bad.append(ln.strip()[:200])
            continue
        if "helios" in low and ("--execute" in low or "train" in low):
            bad.append(ln.strip()[:200])
    return bad

def main():
    print(f"wait_gpu_launch start at {now_pt()}", flush=True)
    deadline = time.time() + 240 * 60  # up to 4h for HELIOS 75 seqs
    while time.time() < deadline:
        used = gpu_used()
        bad = blocking_procs()
        print(f"{now_pt()} mem={used}MiB blockers={len(bad)}", flush=True)
        if bad:
            for b in bad[:2]:
                print("  block:", b, flush=True)
        if not bad and 0 <= used < 100:
            break
        time.sleep(20)
    else:
        print("TIMEOUT waiting for GPU", flush=True)
        return 2

    time.sleep(5)
    used = gpu_used()
    bad = blocking_procs()
    if bad or not (0 <= used < 100):
        print(f"ABORT after recheck mem={used} bad={bad}", flush=True)
        return 3

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    CK.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    try:
        s = json.loads(STATUS.read_text(encoding="utf-8-sig")) if STATUS.exists() else {}
    except Exception:
        s = {}
    s["outcome"] = "TRAINING_T24_SEEDS_42_2024_888"
    s["gpu"] = "TRAINING"
    s["updated_at"] = now_pt()
    STATUS.write_text(json.dumps(s, indent=2), encoding="utf-8")

    # Include 42 so hold_logits_strong.npz ens includes s42+2024+888
    cmd = [str(py), "-u", "train_ir_focal_ft_t24_v24.py",
           "--t", "24",
           "--cache-dir", str(CACHE),
           "--ckpt-dir", str(CK),
           "--seeds", "42", "2024", "888"]
    print("launch", cmd, flush=True)
    with open(LOG, "w", encoding="utf-8") as lo, open(ERR, "w", encoding="utf-8") as le:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lo, stderr=le)
    PIDF.write_text(str(p.pid), encoding="utf-8")
    print(f"PID={p.pid} log={LOG} at={now_pt()}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
