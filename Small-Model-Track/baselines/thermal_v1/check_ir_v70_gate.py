"""Gate ir_v70. Imports shipped nested_fixed/apply_cfg."""
from __future__ import annotations

import csv
from pathlib import Path

from probe_ir_v24_fuse import apply_cfg, nested_fixed
from write_ir_v30_safe import SAMPLE_SUB
from write_ir_v70_v69v44_on_v66 import (
    BANNED, CFG, KEEP, NEST_MIN, ROOT, SPEC, U24_MIN, U8_MIN, V29B_CSV, V66_CSV,
    deploy_pack_mb, evaluate_v70_hold, write_v70_submission,
)


def _load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    sample, v29, v66 = _load(SAMPLE_SUB), _load(V29B_CSV), _load(V66_CSV)
    hold = evaluate_v70_hold(verbose=True)
    nest = nested_fixed(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["yu"], hold["mask0"], CFG)
    apply_cfg(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["mask0"], CFG)
    nested_acc = float(nest["mean"])
    nested_gt = nested_acc + 1e-12 >= NEST_MIN
    users, users_ok = {}, True
    for f in nest["folds"]:
        leave, te = int(f["leave"]), float(f["te_acc"])
        if leave == 8:
            floor, ok = U8_MIN, te + 1e-5 >= U8_MIN
        elif leave == 24:
            floor, ok = U24_MIN, te + 1e-5 >= U24_MIN
        else:
            floor = hold["v29_folds"][leave]
            ok = te + 1e-15 >= floor
        users[leave] = {"te_acc": te, "floor": floor, "ok": int(ok)}
        users_ok = users_ok and ok
    result = write_v70_submission(hold, verbose=True)
    if not result.get("submit_ok") or result.get("out_csv") is None:
        print(f"kind=avg_v69_v44_on_v66 spec={SPEC} nested_acc={nested_acc:.10f}", flush=True)
        print(f"submit_ok=0 pass=0 test={result['test']}", flush=True)
        raise SystemExit(1)
    cand = _load(Path(result["out_csv"]))
    schema_ok = len(cand) == 405 and [r["path"] for r in cand] == [r["path"] for r in sample]
    diffs = [i for i, (a, b) in enumerate(zip(v66, cand)) if a["prediction"] != b["prediction"]]
    paths = [cand[i]["path"] for i in diffs]
    banned = [p for p in paths if p in BANNED]
    keep_ok = all(a["prediction"] == b["prediction"] for a, b in zip(v66, cand) if a["path"] in KEEP)
    fd = len(diffs)
    size_mb = deploy_pack_mb()
    ranked = hold["best"]["kind"] == "avg_v69_v44_on_v66" and abs(CFG["wc"] - 0.09) < 1e-9
    expected = {
        "small_model_track_test/SM_test_0051/",
        "small_model_track_test/SM_test_0077/",
        "small_model_track_test/SM_test_0337/",
    }
    clips_ok = set(paths) == expected
    passed = (
        nested_gt and users_ok and schema_ok and 2 <= fd <= 4 and not banned
        and keep_ok and 0 < size_mb <= 100 and ranked and clips_ok
        and bool(result.get("submit_ok"))
    )
    print(f"kind=avg_v69_v44_on_v66 spec={SPEC} nested_acc={nested_acc:.10f}", flush=True)
    print(f"nested_gt_floor={int(nested_gt)} users_ok={int(users_ok)} user_bits={users}", flush=True)
    print(f"file_dis_v66={fd} paths={paths} banned={banned} keep_public={int(keep_ok)} clips_ok={int(clips_ok)}", flush=True)
    print(f"size_mb={size_mb:.4f} submit_ok={int(bool(result.get('submit_ok')))} pass={int(passed)}", flush=True)
    print(f"csv={result['out_csv']}", flush=True)
    if not passed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
