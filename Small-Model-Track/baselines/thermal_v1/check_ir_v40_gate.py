"""In-repo gate for ir_v40 avg-on-v39 ranked prefix. Imports shipped nested_fixed/apply_cfg."""
from __future__ import annotations

import csv
from pathlib import Path

from probe_ir_v24_fuse import apply_cfg, nested_fixed
from write_ir_v30_safe import SAMPLE_SUB
from write_ir_v40_avg_on_v39 import (
    AVG_SPEC, BANNED, CFG, KEEP, NEST_MIN, ROOT, U24_MIN, U8_MIN,
    V29B_CSV, V36_CSV, V39_CSV, deploy_pack_mb, evaluate_v40_hold, write_v40_submission,
)


def _load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    sample = _load_csv(SAMPLE_SUB)
    v29 = _load_csv(V29B_CSV)
    v36 = _load_csv(V36_CSV)
    v39 = _load_csv(V39_CSV)
    hold = evaluate_v40_hold(verbose=True)
    cfg = hold["best"]["cfg"]
    assert abs(float(cfg["wc"]) - 0.09) < 1e-9
    assert hold["best"]["kind"] == "avg_irbox_t24_on_v39"
    assert AVG_SPEC != (0.30, 0.50, 4)
    nest = nested_fixed(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["yu"], hold["mask0"], CFG)
    apply_cfg(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["mask0"], CFG)
    nested_acc = float(nest["mean"])
    assert abs(nested_acc - float(hold["nest"]["mean"])) < 1e-12
    nested_gt = nested_acc + 1e-12 >= NEST_MIN
    users = {}
    users_ok = True
    for f in nest["folds"]:
        leave = int(f["leave"])
        te = float(f["te_acc"])
        if leave == 8:
            floor, ok = U8_MIN, te + 1e-5 >= U8_MIN
        elif leave == 24:
            floor, ok = U24_MIN, te + 1e-5 >= U24_MIN
        else:
            floor = hold["v29_folds"][leave]
            ok = te + 1e-15 >= floor
        users[leave] = {"te_acc": te, "floor": floor, "ok": int(ok)}
        users_ok = users_ok and ok
    result = write_v40_submission(hold, verbose=True)
    out_csv = Path(result["out_csv"])
    cand = _load_csv(out_csv)
    schema_ok = (
        len(cand) == 405
        and list(cand[0].keys()) == ["path", "prediction"]
        and [r["path"] for r in cand] == [r["path"] for r in sample]
        and all(r["path"].endswith("/") for r in cand)
        and all(r["prediction"].lstrip("-").isdigit() and 0 <= int(r["prediction"]) <= 39 for r in cand)
    )
    diffs39 = [i for i, (a, b) in enumerate(zip(v39, cand)) if a["prediction"] != b["prediction"]]
    diffs29 = [i for i, (a, b) in enumerate(zip(v29, cand)) if a["prediction"] != b["prediction"]]
    paths39 = [cand[i]["path"] for i in diffs39]
    banned_hit = [p for p in paths39 if p in BANNED]
    keep_ok = all(a["prediction"] == b["prediction"] for a, b in zip(v39, cand) if a["path"] in KEEP)
    file_dis_v39 = len(diffs39)
    tight = 2 <= file_dis_v39 <= 4
    size_mb = deploy_pack_mb()
    pack_ok = 0 < size_mb <= 100.0
    passed = bool(
        nested_gt and users_ok and schema_ok and pack_ok and tight
        and not banned_hit and keep_ok
        and abs(cfg["wc"] - 0.09) < 1e-9
        and hold["best"]["n_avg"] >= 1
    )
    print(f"kind=avg_irbox_t24_on_v39 spec={AVG_SPEC} wc={cfg['wc']}", flush=True)
    print(f"nested_acc={nested_acc:.10f}", flush=True)
    print(f"nested_gt_floor={int(nested_gt)} users_ok={int(users_ok)}", flush=True)
    print(f"user_bits={users}", flush=True)
    print(
        f"hold_dis_v29b={hold['best']['dis']} file_dis_v29b={len(diffs29)} "
        f"file_dis_v39={file_dis_v39} paths={paths39} banned={banned_hit} keep_public={int(keep_ok)}",
        flush=True,
    )
    print(f"schema_ok={int(schema_ok)} size_mb={size_mb:.4f} size_ok={int(pack_ok)}", flush=True)
    print(f"submit_ok={int(bool(result.get('submit_ok')))} pass={int(passed)}", flush=True)
    print(f"csv={out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
