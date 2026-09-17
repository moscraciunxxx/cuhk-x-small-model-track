import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
PT = timezone(timedelta(hours=-7))
now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
ROOT = Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1")
status = {
  "tag": "ir_v24_status",
  "outcome": "IN_PROGRESS_PAUSED_GPU_YIELD_HOTMOE",
  "keep_ir_v7": True,
  "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": 0.7530364372469636},
  "wrote_csv": False,
  "csv": None,
  "gate": {"hold_min": 0.758, "nested_min": 0.758, "min_disagree": 20},
  "angles": {
    "1_x3d_m_112": {"result": "MISS", "solo_hold": 0.5703, "fuse_best_honest_nested": 0.7359,
      "note": "no r2plus1d_34 in tv/pv; X3D-M ~6MB fp16; no fuse lift"},
    "1_r2plus1d_r50": {"result": "MISS_DEAD_BRANCH", "best_val_hold": 0.5921,
      "note": "plateau ~0.59 << classic R2P1D-18 ~0.67"},
    "1_r2plus1d_34": {"result": "UNAVAILABLE", "note": "no arch+weights (403/404 IG65M)"},
    "3_more_frames_t24": {"result": "PAUSED_CACHE_INCOMPLETE", "cache": "cache/ir_yolo_v4_t24",
      "note": "only train_x npy present; resume AFTER HotMoE frees GPU then train_ir_r2p1d18_tmore_v24 + fuse"},
    "2_yolo_ir_ft": {"result": "SCRIPTED_NOT_RUN", "script": "train_yolo_ir_ft_v24.py"},
    "4_focal_ce": {"result": "CPU_PREP", "weak_cls": [25, 38, 26, 37, 18, 24],
      "analysis": "logs/ir_v7_class_weak.json"},
    "5_fuse": {"result": "WAITING_STRONG_IR", "script": "probe_ir_v24_fuse.py"}
  },
  "next_roi": [
    "GPU free: finish ir_yolo_v4_t24 cache then train_ir_r2p1d18_tmore_v24.py",
    "If T24 weak: YOLO-FT then light rebuild + R2P1D-18",
    "Focal/class-balanced CE on classic T16 weak classes",
    "Hold submission_ir_v7.csv — no weak submit"
  ],
  "gpu_handoff": {
    "machine": "MosCraciunXXX",
    "machineId": "4ff6e647-6f8c-4d02-88f0-5dda6684ae36",
    "status": "YIELDED_TO_HOTMOE",
    "at": now,
    "note": "All CUHK-X cuda stopped until hotmoe_val.csv / parent clears"
  },
  "updated_at": now
}
(ROOT / "metrics_ir_v24_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print("wrote", ROOT / "metrics_ir_v24_status.json", now)
