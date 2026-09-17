"""Write submission_ir_v26.csv for soft_c9_pa_T1.5_a0.6 clear win.
Verify gate across mid variants that have matching test logits.
"""
from __future__ import annotations
import csv, json, shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, preds_full, nested_fixed, nested_retune,
    GATE, MIN_DISAGREE, V7_CFG,
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
CK = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
PT = timezone(timedelta(hours=-7))


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def soft_mix(a, b, T=1.5, alpha=0.6):
    pa, pb = softmax_np(a, T), softmax_np(b, T)
    p = (1 - alpha) * pa + alpha * pb
    return np.log(np.clip(p, 1e-8, 1.0)).astype(np.float32)


def write_sub(path, meta, preds, empty, fb):
    nfb = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i]))
                nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb


def load_classic9_test():
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    new_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
    z = np.load(old_ckpt / "hold_logits_v6.npz", allow_pickle=True)
    tags_old = [str(t) for t in z["tags"]]
    hz = np.load(new_dir / "hold_logits_new_seeds.npz", allow_pickle=True)
    member_acc = {}
    tag_to_test = {}
    for t in tags_old:
        seed = t.replace("pool_seed", "")
        p = old_ckpt / f"test_logits_{t}.npy"
        if not p.exists():
            p = old_ckpt / f"test_logits_seed{seed}.npy"
        tag_to_test[t] = np.load(p)
        idx = list(tags_old).index(t)
        member_acc[t] = float((z["base"][idx].argmax(1) == z["y"]).mean())
    for t in hz["tags"]:
        t = str(t)
        seed = t.replace("pool_seed", "")
        tag_to_test[t] = np.load(new_dir / f"test_logits_seed{seed}.npy")
        member_acc[t] = float((hz[t].argmax(1) == z["y"]).mean())
    order = sorted(member_acc.keys(), key=lambda k: -member_acc[k])[:9]
    return np.mean([tag_to_test[t] for t in order], 0).astype(np.float32), order


def main():
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    phaseA = np.mean([
        np.load(CK / "hold_logits_seed42.npy"),
        np.load(CK / "hold_logits_seed888.npy"),
        np.load(CK / "hold_logits_seed2024.npy"),
    ], 0).astype(np.float32)
    ir = soft_mix(classic9, phaseA, T=1.5, alpha=0.6)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]

    mid_hold = {
        "ens4_bonetcn": np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy")[hold_idx].astype(np.float32),
        "plain": np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")[hold_idx].astype(np.float32),
        "ens3": np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3.npy")[hold_idx].astype(np.float32),
        "ens3_soft": np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3_softT25.npy")[hold_idx].astype(np.float32),
    }
    mid_test_map = {
        "plain": TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy",
        "ens3": TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits_ens3.npy",
        "ens3_soft": TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits_ens3_softT25.npy",
    }
    Ts_fine = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    Ts_med = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    rows = []
    for name, mid in mid_hold.items():
        mask = th.any(1) & mid.any(1)
        v7p = preds_full(classic9, th, mid, mask, V7_CFG)
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask, Ts_fine, ngrid=25)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask, Ts_med, ngrid=17)
        wa = float(np.mean([f["cfg"]["wa"] for f in nest_rt["folds"]]))
        wb = float(np.mean([f["cfg"]["wb"] for f in nest_rt["folds"]]))
        wc = float(np.mean([f["cfg"]["wc"] for f in nest_rt["folds"]]))
        T = float(np.mean([f["cfg"]["T"] for f in nest_rt["folds"]]))
        s = max(wa + wb + wc, 1e-9)
        scfg = {"wa": wa / s, "wb": wb / s, "wc": wc / s, "T": T, "mode": "nested_stacked",
                "acc": b_acc, "n": int(mask.sum())}
        nest_f = nested_fixed(ir, th, mid, yt, yu, mask, scfg)
        honest = min(float(nest_rt["mean"]), float(nest_f["mean"]))
        pred = preds_full(ir, th, mid, mask, scfg)
        dis = int(((pred >= 0) & (v7p >= 0) & (pred != v7p)).sum())
        clears = bool(b_acc >= GATE and honest >= GATE and dis >= MIN_DISAGREE)
        has_test = name in mid_test_map and mid_test_map[name].exists()
        row = {"mid": name, "full": b_acc, "honest": honest, "nestR": float(nest_rt["mean"]),
               "nestF": float(nest_f["mean"]), "dis": dis, "clears": clears, "has_test": has_test,
               "scfg": scfg, "bcfg": bcfg}
        rows.append(row)
        print(f"{name}: full={b_acc:.4f} honest={honest:.5f} dis={dis} clear={clears} has_test={has_test} scfg={scfg}", flush=True)

    # Prefer clear + has_test; else clear ens4 reported but no CSV if no matching test
    ranked = sorted(rows, key=lambda r: (r["clears"] and r["has_test"], r["clears"], r["honest"], r["dis"]), reverse=True)
    chosen = None
    for r in ranked:
        if r["clears"] and r["has_test"]:
            chosen = r
            break
    if chosen is None:
        print("NO_CSV: no mid with both clear-gate and matching test logits", flush=True)
        status = {
            "tag": "ir_v26_soft_csv_blocked",
            "outcome": "WIN_METRICS_NO_CSV",
            "keep_ir_v7": True,
            "wrote_csv": False,
            "reason": "winning soft_c9_pa may need ens4_bonetcn mid test logits (missing)",
            "rows": rows,
            "updated_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        }
        (ROOT / "metrics_ir_v26_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        (ROOT / "metrics_ir_v25_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return 2

    ir_test_c9, order = load_classic9_test()
    phaseA_test = np.mean([
        np.load(CK / "test_logits_seed42.npy"),
        np.load(CK / "test_logits_seed888.npy"),
        np.load(CK / "test_logits_seed2024.npy"),
    ], 0).astype(np.float32)
    ir_test = soft_mix(ir_test_c9, phaseA_test, T=1.5, alpha=0.6)
    th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy").astype(np.float32)
    mid_test = np.load(mid_test_map[chosen["mid"]]).astype(np.float32)
    cfg = chosen["scfg"]
    T = cfg["T"]
    preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T) + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)

    cache = ROOT / "cache" / "ir_yolo_v4"
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    out = ROOT / "submission_ir_v26.csv"
    nfb = write_sub(out, meta, preds, empty, fb)
    v7 = []
    with open(ROOT / "submission_ir_v7.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v7.append(int(row["prediction"]))
    dis_test = int(sum(int(a) != int(b) for a, b in zip(preds, v7)))
    shutil.copy2(out, TRACK / "submission.csv")
    print(f"WROTE {out} nfb={nfb} mid={chosen['mid']} honest={chosen['honest']:.5f} "
          f"full={chosen['full']:.4f} hold_dis={chosen['dis']} test_dis={dis_test}", flush=True)

    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    kaggle_note = (
        f"Clear win vs gate 0.758: honest_nested={chosen['honest']:.5f} full={chosen['full']:.4f} "
        f"disagree_test_vs_v7={dis_test}. Submit submission_ir_v26.csv "
        f"(soft classic9+phaseA T1.5 a0.6, mid={chosen['mid']}). Beat public ir_v7 0.69154."
    )
    status = {
        "tag": "ir_v26_soft_c9_pa_csv",
        "outcome": "WIN",
        "keep_ir_v7": False,
        "wrote_csv": True,
        "csv": "submission_ir_v26.csv",
        "promoted_track": str(TRACK / "submission.csv"),
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154},
        "method": f"soft_mix(classic9, phaseA_42_888_2024, T=1.5, a=0.6) + th_v6 + mid_{chosen['mid']}; nested_stacked cfg",
        "metrics": {
            "full": chosen["full"],
            "honest_nested": chosen["honest"],
            "disagree_hold_vs_v7": chosen["dis"],
            "disagree_test_vs_v7": dis_test,
            "cfg": cfg,
            "mid": chosen["mid"],
            "ir_order9": order,
            "empty_fallback": nfb,
        },
        "all_mid_rows": rows,
        "kaggle_note": kaggle_note,
        "next_roi": ["Kaggle submit ir_v26", "Strong T24 s42 still training for further lift"],
        "updated_at": now,
    }
    (ROOT / "metrics_ir_v26_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v25_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "HOTC_GPU_HANDOFF.md").write_text(
        "# CUHK-X Small Model Track - status\n\n"
        f"**WIN** nest-honest **{chosen['honest']:.4f}** soft_c9_pa_T1.5_a0.6 mid={chosen['mid']} "
        f"full={chosen['full']:.4f} test_dis_vs_v7={dis_test}.\n"
        "CSV: **submission_ir_v26.csv** (promoted track submission.csv). Prior public ir_v7=0.69154.\n\n"
        f"## Kaggle\n{kaggle_note}\n\n"
        f"## Updated {now}\n",
        encoding="utf-8",
    )
    print("STATUS updated", kaggle_note, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
