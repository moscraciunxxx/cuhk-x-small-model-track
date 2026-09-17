"""ir_v20: S3D (or CNN+GRU) IR diversity + classic9 BASE + Thermal + mid_ens fuse vs ir_v7.
Gate: hold+nested-honest sameT >= ~0.763 AND >=20 disagrees vs ir_v7.
CPU-only fuse once hold logits exist.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import load_members, nested_fixed, fuse3_sameT
from probe_ir_v18 import (
    apply_cfg, preds_full, nested_retune, load_ir_pools, build_thermal_variants, honest_of, V7_CFG, V7_HOLD, GATE, MIN_DISAGREE,
)

ROOT = Path(__file__).resolve().parent
S3D_DIR = ROOT / "checkpoints" / "ir_yolo_s3d_v20"
CNN_DIR = ROOT / "checkpoints" / "ir_yolo_cnn_gru_v20"
MC3_DIR = ROOT / "checkpoints" / "ir_yolo_mc3_18_v19"
MC3_REF = 0.6138613861386139  # abort if new IR << this and no complementarity


def load_new_ir(yt):
    """Prefer S3D; fall back to CNN+GRU. Return (name, logits, acc, dir) or (None,...)."""
    for name, d in [("s3d", S3D_DIR), ("cnn_gru", CNN_DIR)]:
        npz = d / "hold_logits_strong.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        solo = z["ens"].astype(np.float32) if "ens" in z.files else np.mean(z["logits"], 0).astype(np.float32)
        if len(solo) != len(yt):
            print(f"WARN {name} hold len {len(solo)} != {len(yt)}; skip", flush=True)
            continue
        acc = float((solo.argmax(1) == yt).mean())
        return name, solo, acc, d
    return None, None, None, None


def disagree_count(a, b, y=None):
    return int((a.argmax(1) != b.argmax(1)).sum())


def main():
    t0 = time.time()
    print("ir_v20 S3D/CNN-GRU IR diversity fuse (CPU)", flush=True)
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

    ir_name_new, ir_solo, ir_acc, ir_dir = load_new_ir(yt)
    abort_note = None
    if ir_solo is None:
        abort_note = "No S3D/CNN-GRU hold npz yet — train first"
        print(abort_note, flush=True)
    else:
        pools[f"{ir_name_new}_solo"] = ir_solo
        focus_irs.append(f"{ir_name_new}_solo")
        dis_c9 = disagree_count(ir_solo, pools["classic9_base"])
        print(f"{ir_name_new} solo hold={ir_acc:.4f} disagree_vs_classic9={dis_c9}", flush=True)
        # quick complementarity probe: blend with classic9
        blend_hits = []
        for w_new in [0.15, 0.25, 0.35, 0.5]:
            key = f"classic9_base+{ir_name_new}_w{w_new:.2f}"
            blended = ((1 - w_new) * pools["classic9_base"] + w_new * ir_solo).astype(np.float32)
            pools[key] = blended
            focus_irs.append(key)
            bacc = float((blended.argmax(1) == yt).mean())
            blend_hits.append((w_new, bacc))
            print(f"  blend classic9+{ir_name_new} w={w_new:.2f} solo={bacc:.4f}", flush=True)
        best_blend = max(blend_hits, key=lambda t: t[1])[1]
        c9_acc = float((pools["classic9_base"].argmax(1) == yt).mean())
        # Abort early signal (recorded; fuse still runs for status)
        if ir_acc < MC3_REF - 0.02 and best_blend <= c9_acc + 0.001:
            abort_note = (
                f"ABORT_SIGNAL: {ir_name_new} solo={ir_acc:.4f} << MC3={MC3_REF:.4f} "
                f"and no complementarity (best_blend={best_blend:.4f} vs classic9={c9_acc:.4f})"
            )
            print(abort_note, flush=True)
        elif ir_acc >= 0.65 or best_blend > c9_acc + 0.002:
            print(f"PROMISING: solo={ir_acc:.4f} best_blend={best_blend:.4f} — multi-seed OK if GPU free", flush=True)

    # optional include MC3 for comparison pools
    mc3_npz = MC3_DIR / "hold_logits_strong.npz"
    mc3_acc = None
    if mc3_npz.exists():
        z = np.load(mc3_npz, allow_pickle=True)
        mc3 = z["ens"].astype(np.float32) if "ens" in z.files else np.mean(z["logits"], 0).astype(np.float32)
        if len(mc3) == len(yt):
            mc3_acc = float((mc3.argmax(1) == yt).mean())
            pools["mc3_s42"] = mc3
            print(f"MC3 ref solo={mc3_acc:.4f}", flush=True)

    print("IR focus:", {k: float((pools[k].argmax(1) == yt).mean()) for k in focus_irs if k in pools}, flush=True)

    th_vars = build_thermal_variants(yt, ir_meta)
    th_focus = [k for k in ["th_v6_v2trio", "th_v6_soft", "th_v2_ens"] if k in th_vars]
    for k in th_vars:
        if k not in th_focus and ("th_v5" in k or "th_v2" in k or "th_v6" in k):
            th_focus.append(k)
    print("th_focus", th_focus, flush=True)

    ir0 = pools["classic9_base"]
    mask0 = th_vars["th_v6_v2trio"].any(1) & mid_alts["mid_ir_v4"].any(1)
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

    from datetime import datetime, timezone, timedelta
    pt = timezone(timedelta(hours=-7))
    status = {
        "tag": "ir_v20_s3d_or_cnn_gru",
        "outcome": "WIN" if clears else ("PENDING_TRAIN" if ir_solo is None else "MISS"),
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "new_ir": {
            "arch": ir_name_new,
            "dir": str(ir_dir) if ir_dir else None,
            "solo_hold": ir_acc,
            "npz": str(ir_dir / "hold_logits_strong.npz") if ir_dir else None,
            "pack_note": "S3D fp16 ~16MB or CNN+GRU ~few MB + yolov8n ~6.5MB << 100MB",
            "abort_note": abort_note,
            "mc3_ref_solo": mc3_acc if mc3_acc is not None else MC3_REF,
        },
        "best": best_same,
        "best_honest_sameT": best_same,
        "top10_sameT": same[:10],
        "wrote_csv": False,
        "csv": None,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "delta_vs_gate": {
            "best_full": (best_same["full"] - GATE) if best_same else None,
            "honest_nested": (best_same["honest_nested"] - GATE) if best_same else None,
        },
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "summary": {
            "new_ir_arch": ir_name_new,
            "new_ir_solo": ir_acc,
            "mc3_solo": mc3_acc,
            "best_fuse_full": best_same["full"] if best_same else None,
            "best_fuse_honest_nested": best_same["honest_nested"] if best_same else None,
            "gate": GATE,
            "disagree": best_same["disagree_vs_v7"] if best_same else None,
            "v7_hold": V7_HOLD,
            "public_keep": "submission_ir_v7.csv @ 0.69154",
        },
        "gpu_status_at_finish": "unknown",
        "finished_at": datetime.now(pt).strftime("%Y-%m-%d %H:%M:%S PT"),
    }

    if clears:
        status["next_roi"].append("WIN: pack S3D/CNN-GRU fp16 + YOLO <=100MB; CSV only if parent authorizes")
    elif ir_solo is None:
        status["next_roi"].append("PENDING: GPU held by LMT — scripts ready; train S3D seed42 when 3060 free")
        status["next_roi"].append("Fallback ready: train_ir_cnn_gru_v20.py (depth_color TemporalCNNHAR) if S3D OOM")
        status["next_roi"].append("Keep ir_v7; no CSV; yield GPU to LMT")
    else:
        h = best_same["honest_nested"] if best_same else 0.0
        status["next_roi"].append(
            f"MISS gate: honest_sameT={h:.4f} full={best_same['full'] if best_same else 0:.4f} "
            f"dis={best_same['disagree_vs_v7'] if best_same else 0} vs gate {GATE:.3f}"
        )
        if abort_note:
            status["next_roi"].append(abort_note)
            status["next_roi"].append("If S3D weak: try CNN+GRU once; else stop IR arch churn, revisit Mid features or LOUO stack")
        elif ir_acc is not None and ir_acc < 0.65:
            status["next_roi"].append(f"{ir_name_new} solo={ir_acc:.3f} <0.65 — multi-seed only if clear complementarity in blends")
        status["next_roi"].append("Keep ir_v7; no weaker CSV; GPU handoff when done")

    out = ROOT / "metrics_ir_v20_status.json"
    out.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("outcome", "new_ir", "best_honest_sameT", "delta_vs_gate", "next_roi")}, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
