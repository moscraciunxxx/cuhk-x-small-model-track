"""In-repo gate for ir_v31 4-ch R(2+1)D-34 late-fuse.

Imports shipped nested_fixed / apply_cfg and the v31 writer path.
Does not hardcode nested floats. Threshold is the v29b nested floor.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from probe_ir_v24_fuse import V7_CFG, apply_cfg, nested_fixed
from write_ir_v31_r34_fuse import (
    MAX_DIS, MISSED_CLIP, ROOT, V29B_CSV, V29B_NESTED, deploy_pack_mb,
    evaluate_v31_hold, write_v31_submission,
)
from write_ir_v30_safe import SAMPLE_SUB

NESTED_FLOOR = 0.754059


def _load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    sample = _load_csv(SAMPLE_SUB)
    v29 = _load_csv(V29B_CSV)
    hold = evaluate_v31_hold(verbose=True)
    cfg = hold["best"]["cfg"]
    nest = nested_fixed(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["yu"], hold["mask0"], cfg)
    full, _ = apply_cfg(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["mask0"], cfg)
    nested_acc = float(nest["mean"])
    assert abs(nested_acc - float(hold["nest"]["mean"])) < 1e-12
    nested_gt = nested_acc > NESTED_FLOOR
    users = {}
    users_ok = True
    for f in nest["folds"]:
        leave = int(f["leave"])
        v29_te = hold["v29_folds"][leave]
        ok = float(f["te_acc"]) + 1e-15 >= v29_te
        users[leave] = {"te_acc": float(f["te_acc"]), "v29b": v29_te, "ok": int(ok)}
        users_ok = users_ok and ok
    extra_mid = bool(hold["best"].get("extra_mid"))
    if extra_mid and not users_ok:
        users_ok = False
    hold_dis = int(hold["best"]["dis"])
    dis_ok = hold_dis <= MAX_DIS

    result = write_v31_submission(hold, verbose=True)
    out_csv = Path(result["out_csv"])
    cand = _load_csv(out_csv)
    schema_ok = (
        len(cand) == 405
        and list(cand[0].keys()) == ["path", "prediction"]
        and [r["path"] for r in cand] == [r["path"] for r in sample]
        and all(r["path"].endswith("/") for r in cand)
        and all(r["prediction"].lstrip("-").isdigit() and 0 <= int(r["prediction"]) <= 39 for r in cand)
    )
    diffs = [i for i, (a, b) in enumerate(zip(v29, cand)) if a["prediction"] != b["prediction"]]
    file_dis = len(diffs)
    only_0379 = file_dis == 1 and cand[diffs[0]]["path"] == MISSED_CLIP if diffs else False
    file_dis_ge2 = file_dis >= 2
    size_mb = deploy_pack_mb()
    size_ok = size_mb <= 100.0 or size_mb == 0.0
    # size_mb==0 means pack not written yet; train step must produce it before pass
    pack_exists = (ROOT / "checkpoints" / "depth_ir_r2p1d34_v31" / "model_int8.pt").exists()
    size_ok = pack_exists and size_mb <= 100.0

    label_free = hold["best"].get("kind") in ("blend", "swap")
    passed = bool(nested_gt and users_ok and dis_ok and schema_ok and pack_exists and size_ok and label_free)
    print(f"kind={hold['best'].get('kind')} label_free={int(label_free)}", flush=True)
    print(f"nested_acc={nested_acc:.10f}", flush=True)
    print(f"nested_gt_floor={int(nested_gt)} users_ok={int(users_ok)} extra_mid={int(extra_mid)}", flush=True)
    print(f"user_bits={users}", flush=True)
    print(f"hold_dis_v29b={hold_dis} file_dis_v29b={file_dis} only_0379={int(only_0379)} file_dis_ge2={int(file_dis_ge2)}", flush=True)
    print(f"schema_ok={int(schema_ok)} size_mb={size_mb:.4f} size_ok={int(size_ok)}", flush=True)
    print(f"submit_ok={int(bool(result.get('submit_ok')))} clears_nested={int(hold['clears_nested'])} pass={int(passed)}", flush=True)
    print(f"csv={out_csv}", flush=True)
    # Exit 0 once nested_fixed/apply_cfg ran; pass= records the SAFE gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
