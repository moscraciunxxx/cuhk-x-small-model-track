"""CPU watcher: when HotMoE finishes hotmoe_val.csv and GPU idle, write gpu_free_flag."""
from __future__ import annotations
import time, json, subprocess
from pathlib import Path
HOT = Path(r"D:\Coding Compete\Hyperspectral Object Tracking Challenge 2026\submissions\hotmoe_val.csv")
FLAG = Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\logs\gpu_free_after_hotmoe.json")
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

def main():
    print("watching", HOT, flush=True)
    while True:
        used = gpu_used_mib()
        exists = HOT.exists()
        print(f"hotmoe_csv={exists} gpu_used_mib={used}", flush=True)
        if exists and 0 <= used < 400:
            payload = {"hotmoe_val_csv": str(HOT), "gpu_used_mib": used, "ready": True,
                       "next": "resume build_cache_ir_tmore_v24 / train_ir_r2p1d18_tmore_v24"}
            FLAG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            HANDOFF.write_text(
                "GPU FREE after HotMoE — " + time.strftime("%Y-%m-%d %H:%M") +
                "\nResume: finish T24 cache -> train_ir_r2p1d18_tmore_v24 -> probe_ir_v24_fuse\n" +
                json.dumps(payload), encoding="utf-8")
            print("READY", payload, flush=True)
            return
        time.sleep(60)

if __name__ == "__main__":
    main()
