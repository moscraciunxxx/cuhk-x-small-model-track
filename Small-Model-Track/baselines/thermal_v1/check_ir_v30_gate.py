"""In-repo gate for ir_v30: import shipped nested_fixed / apply_cfg / writer path.

Does not reimplement fuse math and does not hardcode the nested accuracy float.
Threshold 0.7540587 is the published v29b nested floor from this stack.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

from probe_ir_v24_fuse import apply_cfg, nested_fixed, V7_CFG
from write_ir_v30_safe import (
    MAX_DIS,
    ROOT,
    SAMPLE_SUB,
    V29B_NESTED,
    deploy_pack_mb,
    evaluate_v30_hold,
    write_v30_submission,
)

V29B_CSV = ROOT / "submission_ir_v29b.csv"
# plan floor: nested acc strictly greater than ir_v29b nested 0.7540587
NESTED_FLOOR = 0.7540587


def _load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_schema(rows, sample_rows):
    assert len(rows) == 405, f"rows={len(rows)}"
    assert list(rows[0].keys()) == ["path", "prediction"], list(rows[0].keys())
    assert [r["path"] for r in rows] == [r["path"] for r in sample_rows]
    assert all(r["path"].endswith("/") for r in rows)
    preds = [int(r["prediction"]) for r in rows]
    assert all(0 <= p <= 39 for p in preds)
    assert all(r["prediction"].lstrip("-").isdigit() for r in rows)
    assert len({r["path"] for r in rows}) == 405
    return True


def main():
    sample_rows = _load_csv(SAMPLE_SUB)
    v29_rows = _load_csv(V29B_CSV)

    hold = evaluate_v30_hold(verbose=True)
    # Re-run the shipped nested_fixed / apply_cfg on the writer tensors.
    nest = nested_fixed(
        hold["ir_sw"], hold["th_sw"], hold["mid"], hold["yt"], hold["yu"], hold["mask0"], V7_CFG
    )
    full, _ = apply_cfg(
        hold["ir_sw"], hold["th_sw"], hold["mid"], hold["yt"], hold["mask0"], V7_CFG
    )
    nested_acc = float(nest["mean"])
    assert abs(nested_acc - float(hold["nest"]["mean"])) < 1e-12
    assert abs(float(full) - float(hold["full"])) < 1e-12
    nested_gt = nested_acc > NESTED_FLOOR
    nested_gt_v29 = nested_acc > V29B_NESTED
    dis_ok = hold["dis_v29"] <= MAX_DIS and hold["dis_v7"] <= MAX_DIS
    clears = bool(hold["clears"] and nested_gt and nested_gt_v29 and dis_ok)

    result = write_v30_submission(hold, verbose=True)
    out_csv = Path(result["out_csv"])
    assert out_csv.exists(), f"missing csv {out_csv}"
    cand = _load_csv(out_csv)
    schema_ok = check_schema(cand, sample_rows)
    file_dis = sum(a["prediction"] != b["prediction"] for a, b in zip(v29_rows, cand))
    differs = file_dis >= 1
    file_dis_ok = file_dis <= MAX_DIS
    size_mb = deploy_pack_mb()
    size_ok = size_mb <= 100.0

    passed = bool(clears and schema_ok and differs and file_dis_ok and size_ok and result["wrote"])
    print(f"nested_acc={nested_acc:.10f}", flush=True)
    print(f"nested_gt_floor={int(nested_gt)} nested_gt_v29b={int(nested_gt_v29)}", flush=True)
    print(f"hold_dis_v29b={hold['dis_v29']} hold_dis_v7={hold['dis_v7']}", flush=True)
    print(f"schema_ok={int(schema_ok)} file_dis_v29b={file_dis} differs_v29b={int(differs)}", flush=True)
    print(f"size_mb={size_mb:.4f} size_ok={int(size_ok)} wrote={int(result['wrote'])}", flush=True)
    print(f"clears_gate={int(hold['clears'])} pass={int(passed)}", flush=True)
    print(f"csv={out_csv}", flush=True)
    if not passed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
