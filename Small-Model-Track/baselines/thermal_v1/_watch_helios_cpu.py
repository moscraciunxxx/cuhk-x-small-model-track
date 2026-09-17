"""CPU-only: watch for HELIOS done markers; do NOT auto-start CUDA (parent must clear)."""
from __future__ import annotations
import time, json, subprocess
from pathlib import Path
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    def now_pt():
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
except Exception:
    def now_pt():
        return time.strftime("%Y-%m-%d %H:%M:%S")

SUB = Path(r"D:\Coding Compete\Hyperspectral Object Tracking Challenge 2026\submissions")
MARKERS = [
    SUB / "helios_val_full.csv",
    SUB / "helios_execute_done.flag",
    Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\logs\helios_finished.flag"),
]
PAUSE = Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\logs\_paused_for_helios.txt")
FLAG = Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\logs\gpu_free_after_helios.json")
HANDOFF = Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\logs\gpu_handoff_ir_v24.txt")

def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()[0]
        return int(float(out))
    except Exception:
        return -1

def helios_python_running():
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % CommandLine"],
            text=True, timeout=15, stderr=subprocess.DEVNULL)
        return any("helios" in ln.lower() for ln in out.splitlines())
    except Exception:
        return False

def main():
    print("watching HELIOS markers", [str(m) for m in MARKERS], flush=True)
    while True:
        used = gpu_used_mib()
        hits = [str(m) for m in MARKERS if m.exists()]
        hel = helios_python_running()
        print(f"markers={hits} helios_proc={hel} gpu_used_mib={used} at={now_pt()}", flush=True)
        # Only write ready flag when marker exists AND gpu idle AND no helios proc.
        # Still: parent must explicitly clear before CUDA resume.
        if hits and (not hel) and 0 <= used < 400:
            payload = {
                "markers": hits,
                "gpu_used_mib": used,
                "ready_candidate": True,
                "parent_clear_required": True,
                "resume_cmd": "python -u train_ir_r2p1d18_tmore_v24.py --t 24 --seeds 42  (partial ep003 saved aside; will retrain full)",
                "at": now_pt(),
            }
            FLAG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            HANDOFF.write_text(
                "HELIOS appears done / GPU idle — WAIT for parent clear before CUDA\n" + json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            print("READY_CANDIDATE (parent clear still required)", payload, flush=True)
            return
        time.sleep(60)

if __name__ == "__main__":
    main()
