"""ir_v19: Mid bone-concat attempt + IR MC3 diversity fuse vs ir_v7 gate.
Gate: hold+nested-honest sameT >= ~0.763 AND >=20 disagrees vs ir_v7.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT, write_sub
from probe_ir_v18 import (
    apply_cfg, preds_full, nested_retune, load_ir_pools, build_thermal_variants, honest_of, V7_CFG, V7_HOLD, GATE, MIN_DISAGREE,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
MC3_DIR = ROOT / "checkpoints" / "ir_yolo_mc3_18_v19"


def main():
    t0 = time.time()
    print("ir_v19 Mid bone + IR MC3 diversity + classic9/thermal fuse", flush=True)
    members, yt, yu = load_members()
    ir_meta_all = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "train_meta.json", encoding="utf-8"))
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    ir_meta = [ir_meta_all[i] for i in hold_idx]
    assert len(ir_meta) == len(yt)

    mid_alts = {
        "mid_ir_v4": np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")[hold_idx].astype(np.float32),
    }
    for name, path in [
        ("mid_ens3", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3.npy"),
        ("mid_ens3_softT25", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3_softT25.npy"),
        ("mid_ens4_bonetcn", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy"),
    ]:
        if path.exists():
            m = np.load(path)
            if len(m) == len(tu):
                mid_alts[name] = m[hold_idx].astype(np.float32)
            elif len(m) == len(hold_idx):
                mid_alts[name] = m.astype(np.float32)
            print(f"  mid {name} solo={(mid_alts[name].argmax(1)==yt)[mid_alts[name].any(1)].mean():.4f}", flush=True)

    pools, c9 = load_ir_pools(members, yt, yu)
    focus_irs = ["classic9_base", "classic_top6_base", "classic_acc_w_base"]

    # MC3 diversity member
    mc3_npz = MC3_DIR / "hold_logits_strong.npz"
    mc3_solo = None
    mc3_acc = None
    if mc3_npz.exists():
        z = np.load(mc3_npz, allow_pickle=True)
        # ens over seeds; shape (n_seeds, n_hold, 40) or ens (n_hold,40)
        if "ens" in z.files:
            mc3_solo = z["ens"].astype(np.float32)
        else:
            mc3_solo = np.mean(z["logits"], 0).astype(np.float32)
        # align length
        if len(mc3_solo) != len(yt):
            print(f"WARN mc3 hold len {len(mc3_solo)} != {len(yt)}; skip mc3 pools", flush=True)
            mc3_solo = None
        else:
            mc3_acc = float((mc3_solo.argmax(1) == yt).mean())
            pools["mc3_s42"] = mc3_solo
            # blends with classic bases
            for base_name in ["classic9_base", "classic_top6_base", "classic_acc_w_base"]:
                b = pools[base_name]
                for w_mc3 in [0.15, 0.25, 0.35, 0.5]:
                    key = f"{base_name}+mc3_w{w_mc3:.2f}"
                    pools[key] = ((1 - w_mc3) * b + w_mc3 * mc3_solo).astype(np.float32)
                    focus_irs.append(key)
            focus_irs.append("mc3_s42")
            print(f"MC3 solo hold={mc3_acc:.4f}", flush=True)
    else:
        print("MC3 hold npz missing — fuse without new IR member", flush=True)

    print("IR focus:", {k: float((pools[k].argmax(1) == yt).mean()) for k in focus_irs if k in pools}, flush=True)

    th_vars = build_thermal_variants(yt, ir_meta)
    th_focus = []
    for k in ["th_v6_v2trio", "th_v6_soft", "th_v2_ens"]:
        if k in th_vars:
            th_focus.append(k)
    for k in th_vars:
        if k not in th_focus and ("th_v5" in k or "th_v2" in k or "th_v6" in k):
            th_focus.append(k)
    print("th_focus", th_focus, flush=True)

    mid0 = mid_alts.get("mid_ens3_softT25", mid_alts.get("mid_ens3", mid_alts["mid_ir_v4"]))
    ir0 = pools["classic9_base"]
    th0 = th_vars.get("th_v6_soft", th_vars["th_v6_v2trio"])
    mask0 = th0.any(1) & mid_alts["mid_ir_v4"].any(1)
    v7_full, _ = apply_cfg(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], yt, mask0, V7_CFG)
    v7_nest = nested_fixed(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    Ts = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    results = []
    for ir_name in focus_irs:
        if ir_name not in pools:
            continue
        ir = pools[ir_name]
        for th_name in th_focus:
            th = th_vars[th_name]
            for mid_name, md in mid_alts.items():
                mask = th.any(1) & md.any(1)
                if mask.sum() < 400:
                    continue
                b_acc, cfg = fuse3_sameT(ir, th, md, yt, mask, Ts, ngrid=21)
                cfg = dict(cfg); cfg["mode"] = "sameT"
                nest_f = nested_fixed(ir, th, md, yt, yu, mask, cfg)
                nest_r = nested_retune(ir, th, md, yt, yu, mask, Ts, ngrid=21)
                preds = preds_full(ir, th, md, mask, cfg)
                disagree = int(((preds != v7_preds) & (preds >= 0) & (v7_preds >= 0)).sum())
                row = {
                    "ir": ir_name, "th": th_name, "mid": mid_name, "mode": "sameT",
                    "full": float(b_acc),
                    "cfg": {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in cfg.items()},
                    "nested_fixed": float(nest_f["mean"]),
                    "nested_retune": float(nest_r["mean"]),
                    "disagree_vs_v7": disagree,
                    "th_solo": float((th.argmax(1) == yt)[th.any(1)].mean()),
                    "mid_solo": float((md.argmax(1) == yt)[md.any(1)].mean()),
                    "ir_solo": float((ir.argmax(1) == yt).mean()),
                }
                row["honest_nested"] = honest_of(row)
                results.append(row)
                print(
                    f"{ir_name}|{th_name}|{mid_name} sameT full={b_acc:.4f} "
                    f"nestF={row['nested_fixed']:.4f} nestR={row['nested_retune']:.4f} "
                    f"honest={row['honest_nested']:.4f} dis={disagree}",
                    flush=True,
                )

    same = [r for r in results if r["mode"] == "sameT"]
    same.sort(key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best_same = same[0] if same else None

    clears = False
    if best_same is not None:
        clears = (
            best_same["full"] >= GATE
            and best_same["honest_nested"] >= GATE
            and best_same["disagree_vs_v7"] >= MIN_DISAGREE
        )

    wrote_csv = False
    csv_path = None
    # No CSV unless clear (and we skip test write complexity unless clear)
    if clears:
        print("GATE CLEAR — CSV path not fully wired for MC3 blends; mark win for manual pack", flush=True)

    # Mid bone metrics
    mid_bone = {
        "bone_midfuse_v1_heavy": 0.4257,
        "bone_midfuse_v2_lite": None,
        "bone_tcn": None,
        "ens3": 0.5545,
        "ens3_softT25": 0.5604,
        "ens4_bonetcn_ir_aligned": float((mid_alts["mid_ens4_bonetcn"].argmax(1)==yt).mean()) if "mid_ens4_bonetcn" in mid_alts else None,
        "ens5_soft_native": 0.5624,
    }
    try:
        mid_bone["bone_midfuse_v2_lite"] = json.load(open(TRACK/"baselines"/"skeleton_imu_v2"/"metrics_bone_midfuse_v2.json"))["mean_val_acc"]
    except Exception:
        pass
    try:
        mid_bone["bone_tcn"] = json.load(open(TRACK/"baselines"/"skeleton_imu_v2"/"metrics_bone_tcn.json"))["mean_val_acc"]
    except Exception:
        pass

    status = {
        "tag": "ir_v19_mid_bone_ir_mc3",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "mid_bone": mid_bone,
        "mc3": {
            "dir": str(MC3_DIR),
            "solo_hold": mc3_acc,
            "npz": str(mc3_npz) if mc3_npz.exists() else None,
            "pack_note": "MC3 fp16 ~23MB + yolov8n ~6.5MB << 100MB",
        },
        "best": best_same,
        "best_honest_sameT": best_same,
        "top10_sameT": same[:10],
        "wrote_csv": wrote_csv,
        "csv": csv_path,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "delta_vs_gate": {
            "best_full": (best_same["full"] - GATE) if best_same else None,
            "honest_nested": (best_same["honest_nested"] - GATE) if best_same else None,
        },
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "summary": {
            "mid_ens3_softT25": mid_bone.get("ens3_softT25"),
            "mid_bone_tcn": mid_bone.get("bone_tcn"),
            "mid_ens5_soft": mid_bone.get("ens5_soft_native"),
            "mc3_solo": mc3_acc,
            "best_fuse_full": best_same["full"] if best_same else None,
            "best_fuse_honest_nested": best_same["honest_nested"] if best_same else None,
            "gate": GATE,
            "disagree": best_same["disagree_vs_v7"] if best_same else None,
            "v7_hold": V7_HOLD,
            "public_keep": "submission_ir_v7.csv @ 0.69154",
        },
        "gpu_status_at_finish": "idle",
    }

    # next_roi
    if clears:
        status["next_roi"].append("WIN: package MC3+YOLO fp16 if IR member is MC3; else classic pack; monitor public vs 0.69154")
    else:
        h = best_same["honest_nested"] if best_same else 0.0
        status["next_roi"].append(
            f"MISS gate: honest_sameT={h:.4f} full={best_same['full'] if best_same else 0:.4f} dis={best_same['disagree_vs_v7'] if best_same else 0} vs gate {GATE:.3f}"
        )
        if mid_bone.get("bone_tcn") and mid_bone["bone_tcn"] < 0.58:
            status["next_roi"].append(
                f"Mid bone-concat/TCN peaked ~{max(x for x in [mid_bone.get('bone_tcn'), mid_bone.get('bone_midfuse_v2_lite'), mid_bone.get('ens5_soft_native')] if x):.3f} <0.58 target; stop more Mid arch churn unless new modality features"
            )
        if mc3_acc is None:
            status["next_roi"].append("MC3 not available at fuse time")
        elif mc3_acc < 0.62:
            status["next_roi"].append(
                f"MC3 solo={mc3_acc:.3f} weak vs classic IR~0.70; try S3D or 2D-CNN+GRU on same crops, or longer MC3 recipe"
            )
        else:
            status["next_roi"].append(f"MC3 solo={mc3_acc:.3f} competitive — if fuse still misses, try 4-way fuse IR-classic+MC3+Th+Mid or stacking")
        status["next_roi"].append("Keep ir_v7; no weaker CSV; GPU idle for LMT or next job")

    from datetime import datetime, timezone, timedelta
    pt = timezone(timedelta(hours=-7))
    status["finished_at"] = datetime.now(pt).strftime("%Y-%m-%d %H:%M:%S PT")

    out = ROOT / "metrics_ir_v19_status.json"
    out.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("outcome", "best_honest_sameT", "delta_vs_gate", "next_roi", "mc3", "mid_bone")}, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
