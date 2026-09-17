"""In-repo gate for ir_v38 T24-on-v36 ranked prefix. Imports shipped nested_fixed/apply_cfg."""
from __future__ import annotations

import csv
from pathlib import Path

from probe_ir_v24_fuse import apply_cfg, nested_fixed
from write_ir_v30_safe import SAMPLE_SUB
from write_ir_v38_t24_on_v36 import (
    CFG, MAX_DIS, MISSED_CLIP, ROOT, T24_SPEC, U24_MIN, U8_MIN, V29B_CSV, V29B_NESTED, V36_CSV,
    deploy_pack_mb, evaluate_v38_hold, write_v38_submission,
)


def _load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    sample = _load_csv(SAMPLE_SUB)
    v29 = _load_csv(V29B_CSV)
    v36 = _load_csv(V36_CSV)
    hold = evaluate_v38_hold(verbose=True)
    cfg = hold["best"]["cfg"]
    assert abs(float(cfg["wc"]) - 0.09) < 1e-9
    assert T24_SPEC[2] != 4
    nest = nested_fixed(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["yu"], hold["mask0"], CFG)
    apply_cfg(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["mask0"], CFG)
    nested_acc = float(nest["mean"])
    assert abs(nested_acc - float(hold["nest"]["mean"])) < 1e-12
    nested_gt = nested_acc > V29B_NESTED
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
    result = write_v38_submission(hold, verbose=True)
    out_csv = Path(result["out_csv"])
    cand = _load_csv(out_csv)
    schema_ok = (
        len(cand) == 405
        and list(cand[0].keys()) == ["path", "prediction"]
        and [r["path"] for r in cand] == [r["path"] for r in sample]
        and all(r["path"].endswith("/") for r in cand)
        and all(r["prediction"].lstrip("-").isdigit() and 0 <= int(r["prediction"]) <= 39 for r in cand)
    )
    diffs36 = [i for i, (a, b) in enumerate(zip(v36, cand)) if a["prediction"] != b["prediction"]]
    diffs29 = [i for i, (a, b) in enumerate(zip(v29, cand)) if a["prediction"] != b["prediction"]]
    file_dis_v36 = len(diffs36)
    file_dis_v29 = len(diffs29)
    only_0379 = file_dis_v36 == 1 and cand[diffs36[0]]["path"] == MISSED_CLIP if diffs36 else False
    size_mb = deploy_pack_mb()
    pack_ok = 0 < size_mb <= 100.0
    tight = 2 <= file_dis_v36 <= 4
    passed = bool(
        nested_gt and users_ok and schema_ok and pack_ok and tight
        and abs(cfg["wc"] - 0.09) < 1e-9 and T24_SPEC[2] != 4
    )
    print(f"kind=t24_on_v36 t24_spec={T24_SPEC} wc={cfg['wc']}", flush=True)
    print(f"nested_acc={nested_acc:.10f}", flush=True)
    print(f"nested_gt_floor={int(nested_gt)} users_ok={int(users_ok)}", flush=True)
    print(f"user_bits={users}", flush=True)
    print(
        f"hold_dis_v29b={hold['best']['dis']} file_dis_v29b={file_dis_v29} "
        f"file_dis_v36={file_dis_v36} only_0379={int(only_0379)} tight_2_4={int(tight)}",
        flush=True,
    )
    print(f"schema_ok={int(schema_ok)} size_mb={size_mb:.4f} size_ok={int(pack_ok)}", flush=True)
    print(f"submit_ok={int(bool(result.get('submit_ok')))} pass={int(passed)}", flush=True)
    print(f"csv={out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
