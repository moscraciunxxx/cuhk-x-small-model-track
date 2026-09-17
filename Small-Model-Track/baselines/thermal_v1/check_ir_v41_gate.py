"""In-repo gate for ir_v41 s123-on-v40. Imports shipped nested_fixed/apply_cfg."""
from __future__ import annotations
import csv
from pathlib import Path
from probe_ir_v24_fuse import apply_cfg, nested_fixed
from write_ir_v30_safe import SAMPLE_SUB
from write_ir_v41_s123_on_v40 import (
    BANNED, CFG, KEEP, NEST_MIN, ROOT, S123_SPEC, U24_MIN, U8_MIN, V29B_CSV, V40_CSV,
    deploy_pack_mb, evaluate_v41_hold, write_v41_submission,
)

def _load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    sample, v29, v40 = _load_csv(SAMPLE_SUB), _load_csv(V29B_CSV), _load_csv(V40_CSV)
    hold = evaluate_v41_hold(verbose=True)
    cfg = hold["best"]["cfg"]
    assert abs(float(cfg["wc"]) - 0.09) < 1e-9
    assert hold["best"]["kind"] == "s123_on_v40"
    nest = nested_fixed(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["yu"], hold["mask0"], CFG)
    apply_cfg(hold["ir_b"], hold["th"], hold["mid"], hold["yt"], hold["mask0"], CFG)
    nested_acc = float(nest["mean"])
    assert abs(nested_acc - float(hold["nest"]["mean"])) < 1e-12
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
    result = write_v41_submission(hold, verbose=True)
    cand = _load_csv(Path(result["out_csv"]))
    schema_ok = (
        len(cand) == 405 and list(cand[0].keys()) == ["path", "prediction"]
        and [r["path"] for r in cand] == [r["path"] for r in sample]
        and all(r["path"].endswith("/") for r in cand)
        and all(r["prediction"].lstrip("-").isdigit() and 0 <= int(r["prediction"]) <= 39 for r in cand)
    )
    diffs40 = [i for i, (a, b) in enumerate(zip(v40, cand)) if a["prediction"] != b["prediction"]]
    paths = [cand[i]["path"] for i in diffs40]
    banned_hit = [p for p in paths if p in BANNED]
    keep_ok = all(a["prediction"] == b["prediction"] for a, b in zip(v40, cand) if a["path"] in KEEP)
    fd = len(diffs40)
    size_mb = deploy_pack_mb()
    passed = bool(
        nested_gt and users_ok and schema_ok and 0 < size_mb <= 100 and 2 <= fd <= 4
        and not banned_hit and keep_ok and hold["best"]["n_s123"] >= 1
    )
    print(f"kind=s123_on_v40 spec={S123_SPEC} wc={cfg['wc']}", flush=True)
    print(f"nested_acc={nested_acc:.10f}", flush=True)
    print(f"nested_gt_floor={int(nested_gt)} users_ok={int(users_ok)}", flush=True)
    print(f"user_bits={users}", flush=True)
    print(f"file_dis_v40={fd} paths={paths} banned={banned_hit} keep_public={int(keep_ok)}", flush=True)
    print(f"schema_ok={int(schema_ok)} size_mb={size_mb:.4f} submit_ok={int(bool(result.get('submit_ok')))} pass={int(passed)}", flush=True)
    print(f"csv={result['out_csv']}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
