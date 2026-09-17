"""Rebuild IR YOLO crop cache at T=24/32 using same yolov8n recipe as v4 (Angle 3).
Tag: ir_yolo_v4_t24 / t32. Reuses build_cache_yolo_v4 logic.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=24, choices=[24, 32])
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default=str(ROOT / "yolov8n.pt"))
    args = ap.parse_args()
    tag = f"v4_t{args.t}"
    cmd = [
        sys.executable, str(ROOT / "build_cache_yolo_v4.py"),
        "--modality", "IR",
        "--t", str(args.t),
        "--size", str(args.size),
        "--weights", args.weights,
        "--device", args.device,
        "--tag", tag,
        "--force",
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
