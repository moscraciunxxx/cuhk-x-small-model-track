"""Post-GPU probe: T24 multi-seed ens + T16 ff + classic9 nested-honest. CSV only if gate clears."""
from __future__ import annotations
import csv, json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune,
    GATE, MIN_DISAGREE, V7_CFG, V7_HOLD,
)
from fuse_ir_v9 import load_members, write_sub

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
PT = timezone(timedelta(hours=-7))
CK_T24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
PRIOR_BEST = 0.7542210500214152

def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]

def conf_gate_ir(a, b, T=1.5, thr=0.0):
    pa, pb = softmax_np(a, T), softmax_np(b, T)
    ca, cb = pa.max(1), pb.max(1)
    out = a.copy()
    use_b = (pb.argmax(1) != pa.argmax(1)) & (cb > ca + thr)
    out[use_b] = b[use_b]
    return out

def load_t24_hold():
    seeds = {}
    for p in sorted(CK_T24.glob("hold_logits_seed*.npy")):
        sid = p.stem.replace("hold_logits_seed", "")
        seeds[sid] = np.load(p).astype(np.float32)
    npz = CK_T24 / "hold_logits_strong.npz"
    if npz.exists():
        ens = np.load(npz)["ens"].astype(np.float32)
    elif seeds:
        ens = np.mean(list(seeds.values()), 0).astype(np.float32)
    else:
        raise FileNotFoundError("no t24 hold logits")
    return ens, seeds

def load_test_stack():
    """Return dict of named IR test logit arrays available for CSV."""
    out = {}
    # classic9 from prior caches
    for p in [
        ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy",
        ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "test_logits_classic9.npy",
    ]:
        if p.exists():
            out["classic9"] = np.load(p).astype(np.float32)
            break
    ff_ens = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "test_logits_ens.npy"
    if ff_ens.exists():
        out["ff"] = np.load(ff_ens).astype(np.float32)
    else:
        # mean available seed test logits
        parts = []
        for p in sorted((ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24").glob("test_logits_seed*.npy")):
            parts.append(np.load(p))
        if parts:
            out["ff"] = np.mean(parts, 0).astype(np.float32)
    t24_ens = CK_T24 / "test_logits_ens.npy"
    if t24_ens.exists():
        out["t24"] = np.load(t24_ens).astype(np.float32)
    else:
        parts = [np.load(p) for p in sorted(CK_T24.glob("test_logits_seed*.npy"))]
        if parts:
            out["t24"] = np.mean(parts, 0).astype(np.float32)
    th = None
    for p in [
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy",
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy",
    ]:
        if p.exists():
            th = np.load(p).astype(np.float32); break
    mid = None
    for p in [
        TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy",
    ]:
        if p.exists():
            mid = np.load(p).astype(np.float32); break
    return out, th, mid

def blend_test(name, stack, coeffs):
    """coeffs: list of (key, w) summing ~1 using keys in stack."""
    parts = []
    ws = []
    for k, w in coeffs:
        if k not in stack:
            return None
        parts.append(stack[k]); ws.append(w)
    ws = np.array(ws, dtype=np.float64); ws /= ws.sum()
    return sum(ws[i] * parts[i] for i in range(len(parts))).astype(np.float32)

def main():
    t0 = time.time()
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    if len(c9m) < 9:
        c9m = allc[:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)

    ft = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)
    t24, seed_logits = load_t24_hold()

    print("c9", float((classic9.argmax(1)==yt).mean()),
          "ff", float((ff.argmax(1)==yt).mean()),
          "t24_ens", float((t24.argmax(1)==yt).mean()),
          "n_seeds", len(seed_logits), flush=True)
    for sid, arr in seed_logits.items():
        print(f"  t24_s{sid}", float((arr.argmax(1)==yt).mean()), flush=True)

    pools = {
        "classic9_base": classic9,
        "t24_ens": t24,
        "ff_ens4": ff,
        "c9_0.5_t24_0.5": (0.5*classic9 + 0.5*t24).astype(np.float32),
        "c9_0.6_t24_0.4": (0.6*classic9 + 0.4*t24).astype(np.float32),
        "c9_0.4_ff_0.4_t24_0.2": (0.4*classic9 + 0.4*ff + 0.2*t24).astype(np.float32),
        "c9_0.4_ff_0.3_t24_0.3": (0.4*classic9 + 0.3*ff + 0.3*t24).astype(np.float32),
        "c9_0.35_ff_0.35_t24_0.3": (0.35*classic9 + 0.35*ff + 0.3*t24).astype(np.float32),
        "c9_0.3_ff_0.4_t24_0.3": (0.3*classic9 + 0.4*ff + 0.3*t24).astype(np.float32),
        "c9_0.45_ff_0.25_t24_0.3": (0.45*classic9 + 0.25*ff + 0.3*t24).astype(np.float32),
        "c9_0.5_ff_0.25_t24_0.25": (0.5*classic9 + 0.25*ff + 0.25*t24).astype(np.float32),
        "ff_0.5_t24_0.5": (0.5*ff + 0.5*t24).astype(np.float32),
        "c9_0.35_ff_0.3_t24_0.2_ft_0.15": (0.35*classic9 + 0.3*ff + 0.2*t24 + 0.15*ft).astype(np.float32),
        "c9_0.25_ff_0.35_t24_0.4": (0.25*classic9 + 0.35*ff + 0.4*t24).astype(np.float32),
        "c9_0.3_ff_0.25_t24_0.45": (0.3*classic9 + 0.25*ff + 0.45*t24).astype(np.float32),
        "c9_0.2_ff_0.3_t24_0.5": (0.2*classic9 + 0.3*ff + 0.5*t24).astype(np.float32),
    }
    for wa in np.linspace(0.20, 0.55, 8):
        for wb in np.linspace(0.15, 0.50, 8):
            wc = 1.0 - wa - wb
            if wc < 0.10 or wc > 0.50:
                continue
            pools[f"fg_c9_{wa:.2f}_ff_{wb:.2f}_t24_{wc:.2f}"] = (wa*classic9 + wb*ff + wc*t24).astype(np.float32)

    for T in (1.0, 1.5, 2.0, 2.5):
        for thr in (0.0, 0.05, 0.1, 0.15):
            pools[f"gate_c9_t24_T{T}_m{thr}"] = conf_gate_ir(classic9, t24, T=T, thr=thr)
            pools[f"gate_ff_t24_T{T}_m{thr}"] = conf_gate_ir(ff, t24, T=T, thr=thr)
            blend = (0.5*classic9 + 0.5*ff).astype(np.float32)
            pools[f"gate_c9ff_t24_T{T}_m{thr}"] = conf_gate_ir(blend, t24, T=T, thr=thr)

    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    Ts_fine = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0]
    Ts_med = [1.0,1.5,2.0,2.5,3.0,3.5]

    results = []
    for ir_name, ir in pools.items():
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask0, Ts_fine, ngrid=31)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask0, Ts_med, ngrid=17)
        pred = preds_full(ir, th, mid, mask0, bcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = min(float(nest_fixed["mean"]), float(nest_rt["mean"]))
        row = {
            "ir": ir_name, "full": b_acc, "cfg": bcfg,
            "nested_fixed": float(nest_fixed["mean"]), "nested_retune": float(nest_rt["mean"]),
            "honest_nested": honest, "disagree_vs_v7": disagree,
            "ir_solo": float((ir.argmax(1)==yt).mean()),
            "clears": bool(b_acc >= GATE and honest >= GATE and disagree >= MIN_DISAGREE),
        }
        results.append(row)
        print(f"{ir_name} full={b_acc:.4f} honest={honest:.4f} dis={disagree} solo={row['ir_solo']:.4f} clear={row['clears']}", flush=True)

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best = ranked[0]
    clears = [r for r in ranked if r["clears"]]
    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")

    wrote_csv = False
    csv_name = None
    csv_note = None
    if clears:
        top = clears[0]
        stack, th_te, mid_te = load_test_stack()
        # Parse blend weights from ir name when possible; else use c9/ff/t24 equal-ish from name
        ir_test = None
        name = top["ir"]
        if name.startswith("fg_c9_") or name.startswith("c9_"):
            # try parse c9_X_ff_Y_t24_Z
            import re
            m = re.search(r"c9_([0-9.]+)_ff_([0-9.]+)_t24_([0-9.]+)", name)
            if m and all(k in stack for k in ("classic9", "ff", "t24")):
                wa, wb, wc = float(m.group(1)), float(m.group(2)), float(m.group(3))
                ir_test = (wa*stack["classic9"] + wb*stack["ff"] + wc*stack["t24"]).astype(np.float32)
            elif name.startswith("c9_") and "_t24_" in name and "ff" not in name and all(k in stack for k in ("classic9","t24")):
                m2 = re.search(r"c9_([0-9.]+)_t24_([0-9.]+)", name)
                if m2:
                    wa, wc = float(m2.group(1)), float(m2.group(2))
                    ir_test = (wa*stack["classic9"] + wc*stack["t24"]).astype(np.float32)
        if ir_test is None and "t24" in stack and "ff" in stack and "classic9" in stack:
            # default to best known mix used on hold
            ir_test = (0.4*stack["classic9"] + 0.3*stack["ff"] + 0.3*stack["t24"]).astype(np.float32)
            csv_note = "fallback_ir_mix_0.4_0.3_0.3"
        if ir_test is not None and th_te is not None and mid_te is not None:
            cfg = top["cfg"]
            T = cfg["T"]
            preds = (cfg["wa"]*softmax_np(ir_test, T) + cfg["wb"]*softmax_np(th_te, T) + cfg["wc"]*softmax_np(mid_te, T)).argmax(1)
            meta = json.loads((ROOT / "cache" / "ir_yolo_ft_v24_t24" / "test_meta.json").read_text(encoding="utf-8"))
            empty = set(json.loads((ROOT / "cache" / "ir_yolo_ft_v24_t24" / "test_empty.json").read_text(encoding="utf-8")))
            # fallback map from skeleton
            fb = {}
            sk = TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv"
            if sk.exists():
                with sk.open(encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
            csv_name = "submission_ir_v25.csv"
            write_sub(ROOT / csv_name, meta, preds, empty, fb)
            # Also promote track submission only if stronger than keep rule - parent decides; we write local CSV
            wrote_csv = True
            print("WROTE", csv_name, "n", len(preds), "note", csv_note, flush=True)
        else:
            csv_note = f"missing_test_stack keys={list(stack.keys())} th={th_te is not None} mid={mid_te is not None}"
            print("GATE cleared but CSV skipped:", csv_note, flush=True)

    report = CK_T24 / "train_report.json"
    members_rep, ens_rep = {}, None
    if report.exists():
        try:
            tr = json.loads(report.read_text(encoding="utf-8-sig"))
            members_rep, ens_rep = tr.get("members", {}), tr.get("ens")
        except Exception:
            pass

    t24_solo = float((t24.argmax(1)==yt).mean())
    status = {
        "tag": "ir_v25_t24_multiseed_probe",
        "outcome": "WIN" if wrote_csv else ("CLEAR_NO_CSV" if clears else "MISS"),
        "keep_ir_v7": not wrote_csv,
        "wrote_csv": wrote_csv,
        "csv": csv_name if wrote_csv else None,
        "csv_note": csv_note,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce_full": v7_full,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "t24_train": {"members": members_rep, "ens": ens_rep, "n_seed_logits": len(seed_logits), "ens_solo_hold": t24_solo},
        "progress": {
            "focal_ft_ens4": float((ff.argmax(1)==yt).mean()),
            "focal_ft_t24_ens": t24_solo,
            "best_honest": best["honest_nested"],
            "best_full": best["full"],
            "best_ir": best["ir"],
            "best_disagree": best["disagree_vs_v7"],
            "n_clear": len(clears),
            "prior_best_honest": PRIOR_BEST,
            "delta_vs_gate": best["honest_nested"] - GATE,
            "improved_vs_prior": best["honest_nested"] - PRIOR_BEST,
        },
        "best": best,
        "top15": ranked[:15],
        "clears": clears[:5],
        "next_roi": (
            [f"PROMOTE/submit {csv_name}; beat ir_v7 0.69154"] if wrote_csv else
            ([
                "Gate cleared on hold but test logits incomplete - run infer_ir_t24_test.py / ensure ff test ens",
                "Keep ir_v7 until CSV writable",
            ] if clears else (
                [
                    "MISS vs gate 0.758; keep ir_v7; no weak CSV",
                    "Add more T24 seeds (7 123 2025) if ens_solo>=0.70 else densify blend CPU",
                    "Continue - do not stop",
                ] if t24_solo < 0.70 else [
                    "MISS but T24 ens strong; train more seeds 7 123 2025 and re-probe",
                    "Keep ir_v7 until clear",
                ]
            ))
        ),
        "elapsed_sec": round(time.time()-t0,1),
        "updated_at": now,
    }
    (ROOT / "metrics_ir_v25_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v25_t24_multiseed.json").write_text(
        json.dumps({"best": best, "top30": ranked[:30], "n_pools": len(pools), "clears": clears,
                    "seed_solos": {sid: float((a.argmax(1)==yt).mean()) for sid,a in seed_logits.items()},
                    "t24_ens_solo": t24_solo}, indent=2), encoding="utf-8")
    print("BEST", best["ir"], "honest", best["honest_nested"], "full", best["full"],
          "dis", best["disagree_vs_v7"], "clears", len(clears), "csv", wrote_csv)
    # signal for pipeline: exit 0 win, 2 miss-but-strong, 1 miss
    if wrote_csv:
        return 0
    if t24_solo >= 0.70 and not clears:
        return 2
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
