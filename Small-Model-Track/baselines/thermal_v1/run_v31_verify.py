"""After train_r2p1d34_4ch writes logits: inventory, gate x2, csv diff, submit or no_submit."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CK34 = ROOT / "checkpoints" / "depth_ir_r2p1d34_v31"
SCRATCH = ROOT / "_tmp_verify"
PT = timezone(timedelta(hours=-7))
MISSED = "small_model_track_test/SM_test_0379/"
KAG = Path(r"C:\Users\moscr\anaconda3\Scripts\kaggle.exe")
COMP = "cuhk-x-competition-small-model-track"
PY = sys.executable


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def inventory():
    import numpy as np
    from model_r2p1d34 import assert_r2p1d34_4ch, build_r2p1d34_4ch
    m = build_r2p1d34_4ch(pretrained=False, progress=False)
    ident = assert_r2p1d34_4ch(m)
    hl, tl = CK34 / "hold_logits.npy", CK34 / "test_logits.npy"
    hold = np.load(hl) if hl.exists() else None
    test = np.load(tl) if tl.exists() else None
    cache = json.loads((ROOT / "cache" / "depth_ir_4ch_v31" / "cache_meta.json").read_text(encoding="utf-8"))
    lines = [
        f"timestamp={datetime.now(PT).strftime('%Y-%m-%d %H:%M:%S PT')}",
        f"arch={ident['arch']}",
        f"layers={ident['layers']}",
        f"in_ch={ident['in_ch']}",
        f"nparams={ident['nparams']}",
        "input=Depth_Color RGB + IR gray",
        "not_r2plus1d_18=true",
        f"cache_train_shape={cache['train_shape']}",
        f"cache_test_shape={cache['test_shape']}",
        f"hold_logits={hl} exists={hl.exists()} shape={None if hold is None else list(hold.shape)}",
        f"test_logits={tl} exists={tl.exists()} shape={None if test is None else list(test.shape)}",
        "late_fuse_loads=IR v29b + Thermal v3/v6 + MidFuse classic via evaluate_v30_hold",
    ]
    if hold is not None:
        lines.append(f"hold_n={hold.shape[0]} hold_c={hold.shape[1]}")
        lines.append(f"hold_md5={md5(hl)}")
    if test is not None:
        lines.append(f"test_n={test.shape[0]} test_c={test.shape[1]}")
        lines.append(f"test_md5={md5(tl)}")
        if test.shape[0] != 405:
            lines.append("ERROR test_n != 405")
    (SCRATCH / "r2p1d34_member.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if hold is None or test is None:
        raise SystemExit("logits missing")
    if hold.shape[1] != 40 or test.shape != (405, 40):
        raise SystemExit(f"bad logit shapes hold={hold.shape} test={None if test is None else test.shape}")


def run_gate(tag: str):
    log = SCRATCH / f"local_gate_{tag}.txt"
    p = subprocess.run([PY, "-u", str(ROOT / "check_ir_v31_gate.py")], cwd=str(ROOT),
                       capture_output=True, text=True)
    text = (p.stdout or "") + (p.stderr or "")
    log.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return p.returncode, text


def csv_diff():
    v29 = list(csv.DictReader(open(ROOT / "submission_ir_v29b.csv", encoding="utf-8")))
    cand_p = ROOT / "submission_ir_v31.csv"
    if not cand_p.exists():
        (SCRATCH / "csv_diff.txt").write_text("missing submission_ir_v31.csv\n", encoding="utf-8")
        return None
    cand = list(csv.DictReader(open(cand_p, encoding="utf-8")))
    diffs = []
    for i, (a, b) in enumerate(zip(v29, cand)):
        if a["prediction"] != b["prediction"]:
            diffs.append(f"{i} {b['path']} v29b={a['prediction']} v31={b['prediction']}")
    only = len(diffs) == 1 and MISSED in diffs[0]
    text = [
        f"file_dis={len(diffs)}",
        f"only_SM_test_0379={int(only)}",
        f"ge2={int(len(diffs) >= 2)}",
        *diffs,
    ]
    (SCRATCH / "csv_diff.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    return len(diffs), only


def model_size():
    from write_ir_v31_r34_fuse import deploy_pack_mb
    from write_ir_v31_r34_fuse import CK34 as C
    pack = C / "model_int8.pt"
    yolo = ROOT / "yolov8n.pt"
    mb = deploy_pack_mb()
    lines = [
        f"total_mb={mb:.4f}",
        f"int8_bytes={pack.stat().st_size if pack.exists() else 0}",
        f"yolo_bytes={yolo.stat().st_size if yolo.exists() else 0}",
        f"int8={pack}",
        f"yolo={yolo}",
        f"assert_le_100={int(mb <= 100 and pack.exists())}",
    ]
    (SCRATCH / "model_size.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return mb, pack.exists() and mb <= 100


def maybe_submit(submit_ok: bool, file_dis, only_0379):
    if not submit_ok or file_dis is None or file_dis < 2 or only_0379:
        reason = []
        if not submit_ok:
            reason.append("submit_ok=false (nested/per-user/dis gate)")
        if file_dis is None:
            reason.append("no csv")
        elif file_dis < 2:
            reason.append(f"file_dis={file_dis} < 2")
        if only_0379:
            reason.append("only SM_test_0379 differs (1-flip, do not submit)")
        txt = "NO_SUBMIT\n" + "\n".join(reason) + "\nno 1-flip / thermal-conf-gate submit this run\n"
        (SCRATCH / "no_submit.txt").write_text(txt, encoding="utf-8")
        print(txt, flush=True)
        return False
    csv_path = ROOT / "submission_ir_v31.csv"
    msg = "ir_v31 4ch R2P1D-34 late-fuse vs v29b nested-gated dis<=15 file_dis>=2"
    p = subprocess.run(
        [str(KAG), "competitions", "submit", "-c", COMP, "-f", str(csv_path), "-m", msg],
        capture_output=True, text=True,
    )
    (SCRATCH / "kaggle_submit.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
    print(p.stdout, p.stderr, flush=True)
    if p.returncode != 0:
        (SCRATCH / "kaggle_submit.log").write_text(
            (p.stdout or "") + (p.stderr or "") + f"\nrc={p.returncode}\n", encoding="utf-8"
        )
        return False
    for i, name in enumerate(["kaggle_submissions.txt", "kaggle_submissions_2.txt"]):
        p2 = subprocess.run([str(KAG), "competitions", "submissions", "-c", COMP], capture_output=True, text=True)
        (SCRATCH / name).write_text((p2.stdout or "") + (p2.stderr or ""), encoding="utf-8")
        if i == 0:
            (SCRATCH / "kaggle_submissions.txt").write_text((p2.stdout or "") + (p2.stderr or ""), encoding="utf-8")
    return True


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    inventory()
    rc1, t1 = run_gate("1")
    rc2, t2 = run_gate("2")
    file_dis, only = csv_diff() if True else (None, None)
    mb, size_ok = model_size()
    submit_ok = "submit_ok=1" in t1 and "submit_ok=1" in t2
    nested_ok = "nested_gt_floor=1" in t1
    print(f"gate_rc={rc1},{rc2} size_ok={size_ok} submit_ok={submit_ok} nested_ok={nested_ok}", flush=True)
    maybe_submit(submit_ok and rc1 == 0 and rc2 == 0 and size_ok, file_dis, only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
