"""ir_v18: Thermal rethink (v5) + MidFuse ens3 upgrade + classic9 BASE fuse.
Gate: hold+nested-honest sameT >= ~0.763 AND >=20 disagrees vs ir_v7.
CSV only if clear. Nested-honest sameT (ir_v16 lesson).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import (
    load_members, softmax_np, nested_fixed, fuse3_sameT, fuse3_geom, fuse3_perT,
    write_sub,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7_HOLD = 0.7530364372469636
GATE = V7_HOLD + 0.01
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}


def apply_cfg(a, b, c, y, mask, cfg):
    mode = cfg.get("mode", "sameT")
    if mode == "perT":
        pa = softmax_np(a[mask], cfg["Ta"])
        pb = softmax_np(b[mask], cfg["Tb"])
        pc = softmax_np(c[mask], cfg["Tc"])
        p = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1)
    elif mode == "geom":
        eps = 1e-8
        T = cfg["T"]
        la = np.log(np.clip(softmax_np(a[mask], T), eps, 1))
        lb = np.log(np.clip(softmax_np(b[mask], T), eps, 1))
        lc = np.log(np.clip(softmax_np(c[mask], T), eps, 1))
        p = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        T = cfg["T"]
        p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
             + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean()), p


def preds_full(a, b, c, mask, cfg):
    out = np.full(len(a), -1, dtype=np.int64)
    mode = cfg.get("mode", "sameT")
    if mode == "perT":
        pa = softmax_np(a[mask], cfg["Ta"]); pb = softmax_np(b[mask], cfg["Tb"]); pc = softmax_np(c[mask], cfg["Tc"])
        out[mask] = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1)
    elif mode == "geom":
        eps = 1e-8; T = cfg["T"]
        la = np.log(np.clip(softmax_np(a[mask], T), eps, 1))
        lb = np.log(np.clip(softmax_np(b[mask], T), eps, 1))
        lc = np.log(np.clip(softmax_np(c[mask], T), eps, 1))
        out[mask] = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        T = cfg["T"]
        out[mask] = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
                     + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return out


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=21, leave_users=(8, 9, 24)):
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc, _ = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": acc, "n": int(te.sum()), "cfg": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cfg.items()}})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def clip_key(m):
    return (int(m["user_id"]), int(m["label"]), str(m.get("trial", "")), str(m.get("action_name", "")))


def align_thermal_npz_to_ir(npz_path, ir_meta, ir_len):
    z = np.load(npz_path, allow_pickle=True)
    th_logits = z["logits"].astype(np.float32)
    th_meta_path = ROOT / "cache" / "thermal_yolo" / "train_meta.json"
    th_users = np.load(ROOT / "cache" / "thermal_yolo" / "train_users.npy")
    th_meta = json.load(open(th_meta_path, encoding="utf-8"))
    hold_u = set(DEFAULT_HOLD_OUT_USERS)
    th_hold_idx = [i for i, u in enumerate(th_users) if int(u) in hold_u]
    assert len(th_hold_idx) == len(th_logits), (len(th_hold_idx), len(th_logits))
    key_to_logit = {}
    for j, i in enumerate(th_hold_idx):
        key_to_logit[clip_key(th_meta[i])] = th_logits[j]
    out = np.zeros((ir_len, th_logits.shape[1]), dtype=np.float32)
    matched = 0
    for i, m in enumerate(ir_meta):
        k = clip_key(m)
        if k in key_to_logit:
            out[i] = key_to_logit[k]
            matched += 1
    return out, matched, list(z["tags"]) if "tags" in z.files else [], z


def align_hold_logits_array(hold_logits, th_hold_idx, th_meta, ir_meta, ir_len):
    key_to_logit = {}
    for j, i in enumerate(th_hold_idx):
        key_to_logit[clip_key(th_meta[i])] = hold_logits[j]
    out = np.zeros((ir_len, hold_logits.shape[1]), dtype=np.float32)
    matched = 0
    for i, m in enumerate(ir_meta):
        k = clip_key(m)
        if k in key_to_logit:
            out[i] = key_to_logit[k]
            matched += 1
    return out, matched


def load_ir_pools(members, yt, yu):
    c9 = sorted([m for m in members if m["tag"] != "pool_seed55"], key=lambda d: -d.get("acc_base", d["acc"]))
    def base_of(m):
        return m["base"] if m.get("base") is not None else m["logits"]
    pools = {
        "classic9_base": np.mean([base_of(m) for m in c9], 0).astype(np.float32),
        "classic_top6_base": np.mean([base_of(m) for m in c9[:6]], 0).astype(np.float32),
    }
    w = np.array([max(m.get("acc_base", m["acc"]), 1e-3) for m in c9], dtype=np.float64); w /= w.sum()
    stack = np.stack([base_of(m) for m in c9], 0)
    pools["classic_acc_w_base"] = np.tensordot(w, stack, axes=(0, 0)).astype(np.float32)
    return pools, c9


def build_thermal_variants(yt, ir_meta):
    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy"
    th0 = np.load(old).astype(np.float32)
    variants = {"th_v6_v2trio": th0, "th_v6_soft": (th0 / 1.8).astype(np.float32)}

    th_users = np.load(ROOT / "cache" / "thermal_yolo" / "train_users.npy")
    th_meta = json.load(open(ROOT / "cache" / "thermal_yolo" / "train_meta.json", encoding="utf-8"))
    hold_u = set(DEFAULT_HOLD_OUT_USERS)
    th_hold_idx = [i for i, u in enumerate(th_users) if int(u) in hold_u]

    for label, path in [
        ("th_v2_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "holdout_ensemble_logits.npz"),
        ("th_v3_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "holdout_ensemble_logits.npz"),
        ("th_v4_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v4" / "holdout_ensemble_logits.npz"),
        ("th_v5_ens", ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink" / "holdout_ensemble_logits.npz"),
    ]:
        if path.exists():
            aligned, matched, tags, z = align_thermal_npz_to_ir(path, ir_meta, len(yt))
            variants[label] = aligned
            print(f"  {label}: matched={matched}/{len(yt)} tags={tags} solo={(aligned.argmax(1)==yt)[aligned.any(1)].mean():.4f}", flush=True)

    # Single v5 rethink ckpts
    v5 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink"
    if v5.exists():
        import torch
        for ck in sorted(v5.glob("pool_seed*.pt")):
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            hl = blob.get("hold_logits")
            if hl is None:
                continue
            hl = np.asarray(hl, dtype=np.float32)
            al, mt = align_hold_logits_array(hl, th_hold_idx, th_meta, ir_meta, len(yt))
            name = f"th_v5_{ck.stem}"
            variants[name] = al
            print(f"  {name}: matched={mt} val={float(blob.get('val_acc', -1)):.4f} solo={(al.argmax(1)==yt)[al.any(1)].mean():.4f}", flush=True)

    # Mix v5 with v2trio
    v5_keys = [k for k in variants if k.startswith("th_v5")]
    for vk in v5_keys:
        a, b = th0, variants[vk]
        both = a.any(1) & b.any(1)
        if both.sum() < 400:
            continue
        for wa, wb, tag in [(0.5, 0.5, "50_50"), (0.65, 0.35, "65_35"), (0.35, 0.65, "35_65")]:
            mix = a.copy()
            mix[both] = wa * a[both] + wb * b[both]
            variants[f"mix_v6_{tag}_{vk}"] = mix

    return variants


def honest_of(r):
    nf = r.get("nested_fixed") or 0.0
    nr = r.get("nested_retune")
    if r["mode"] == "sameT" and nr is not None:
        return float(min(nf, nr))
    if r["mode"] == "perT":
        return float(nf) - 0.01
    return float(nf)


def main():
    t0 = time.time()
    print("ir_v18 Thermal-rethink + Mid ens3 + classic9 BASE", flush=True)
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
    ]:
        if path.exists():
            m = np.load(path)
            if len(m) == len(tu):
                mid_alts[name] = m[hold_idx].astype(np.float32)
                print(f"  mid {name} solo={(mid_alts[name].argmax(1)==yt)[mid_alts[name].any(1)].mean():.4f}", flush=True)

    pools, c9 = load_ir_pools(members, yt, yu)
    focus_irs = ["classic9_base", "classic_top6_base", "classic_acc_w_base"]
    print("IR pools:", {k: float((v.argmax(1) == yt).mean()) for k, v in pools.items() if k in focus_irs}, flush=True)

    print("Building thermal variants...", flush=True)
    th_vars = build_thermal_variants(yt, ir_meta)
    # Prefer useful thermals for search (limit explosion)
    th_focus = []
    for k in ["th_v6_v2trio", "th_v6_soft", "th_v2_ens", "th_v5_ens"] + [k for k in th_vars if k.startswith("th_v5_") or k.startswith("mix_v6_")]:
        if k in th_vars and k not in th_focus:
            th_focus.append(k)
    # Always include all th_v5 / mix with v5
    for k in th_vars:
        if ("th_v5" in k or "mix_v6_" in k) and k not in th_focus:
            th_focus.append(k)
    print("th_focus", th_focus, flush=True)
    for k in th_focus:
        v = th_vars[k]
        nz = int(v.any(1).sum())
        acc = float((v.argmax(1) == yt)[v.any(1)].mean()) if nz else 0.0
        print(f"  {k}: nz={nz} solo={acc:.4f}", flush=True)

    mid0 = mid_alts.get("mid_ens3", mid_alts["mid_ir_v4"])
    ir0 = pools["classic9_base"]
    th0 = th_vars["th_v6_v2trio"]
    mask0 = th0.any(1) & mid_alts["mid_ir_v4"].any(1)
    v7_full, _ = apply_cfg(ir0, th0, mid_alts["mid_ir_v4"], yt, mask0, V7_CFG)
    v7_nest = nested_fixed(ir0, th0, mid_alts["mid_ir_v4"], yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(ir0, th0, mid_alts["mid_ir_v4"], mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    Ts = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    results = []
    for ir_name in focus_irs:
        ir = pools[ir_name]
        for th_name in th_focus:
            th = th_vars[th_name]
            for mid_name, md in mid_alts.items():
                mask = th.any(1) & md.any(1) & np.ones(len(yt), bool)
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
                    "full": float(b_acc), "cfg": {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in cfg.items()},
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

    # Rank by honest nested sameT first
    same = [r for r in results if r["mode"] == "sameT"]
    same.sort(key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best_same = same[0] if same else None
    best = best_same

    clears = False
    csv_path = None
    if best_same is not None:
        clears = (
            best_same["full"] >= GATE
            and best_same["honest_nested"] >= GATE
            and best_same["disagree_vs_v7"] >= MIN_DISAGREE
        )

    # CSV only if clear AND we have test logits mapping
    if clears:
        th_test = None
        mid_test = None
        # mid test
        for p in [
            TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits_ens3.npy",
            TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy",
        ]:
            if "ens3" in best_same["mid"] and "ens3" not in p.name:
                continue
            if p.exists():
                mid_test = np.load(p).astype(np.float32)
                if "ens3" in best_same["mid"]:
                    break
        if "ens3" in best_same["mid"]:
            p = TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits_ens3.npy"
            if p.exists():
                mid_test = np.load(p).astype(np.float32)
        else:
            p = TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy"
            if p.exists():
                mid_test = np.load(p).astype(np.float32)

        # thermal test
        if best_same["th"].startswith("th_v5") or "th_v5" in best_same["th"]:
            p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink" / "test_logits.npy"
            if p.exists():
                th_test = np.load(p).astype(np.float32)
        if th_test is None and best_same["th"].startswith("th_v6"):
            p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "test_logits.npy"
            if p.exists():
                th_test = np.load(p).astype(np.float32)
        if th_test is None and best_same["th"].startswith("mix_v6"):
            # mix v2trio + v5
            p2 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "test_logits.npy"
            p5 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink" / "test_logits.npy"
            if p2.exists() and p5.exists():
                a = np.load(p2).astype(np.float32); b = np.load(p5).astype(np.float32)
                if "50_50" in best_same["th"]:
                    th_test = 0.5 * a + 0.5 * b
                elif "65_35" in best_same["th"]:
                    th_test = 0.65 * a + 0.35 * b
                elif "35_65" in best_same["th"]:
                    th_test = 0.35 * a + 0.65 * b
                else:
                    th_test = 0.5 * a + 0.5 * b

        ir_test_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7"
        # reuse finalize path via members test if available through write helpers — load classic mean from v5 hold style
        # Prefer existing ir test pool used by prior submits
        ir_test = None
        for cand in [
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "test_logits_classic9.npy",
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy",
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v4" / "test_logits.npy",
        ]:
            if cand.exists():
                ir_test = np.load(cand).astype(np.float32)
                break
        # Build from member test logits if needed
        if ir_test is None:
            tests = []
            v4 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v4"
            for m in c9:
                tag = m["tag"]
                p = v4 / f"test_logits_{tag}.npy"
                if not p.exists():
                    p = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / f"test_logits_{tag}.npy"
                if p.exists():
                    tests.append(np.load(p).astype(np.float32))
            if tests:
                ir_test = np.mean(np.stack(tests, 0), 0)

        if ir_test is not None and th_test is not None and mid_test is not None and best_same["ir"] == "classic9_base":
            cfg = best_same["cfg"]
            T = cfg["T"]
            preds = (
                cfg["wa"] * softmax_np(ir_test, T)
                + cfg["wb"] * softmax_np(th_test, T)
                + cfg["wc"] * softmax_np(mid_test, T)
            ).argmax(1)
            meta = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "test_meta.json", encoding="utf-8"))
            empty = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "test_empty.json", encoding="utf-8"))
            csv_path = str(ROOT / "submission_ir_v18.csv")
            write_sub(Path(csv_path), meta, preds, empty, 0)
            print(f"WROTE {csv_path}", flush=True)
        else:
            print("GATE cleared but CSV skipped (missing test logits mapping)", flush=True)

    # thermal v5 metrics if present
    v5m = None
    for p in [ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink" / "metrics_thermal_v4.json",
              ROOT / "metrics_thermal_v5.json"]:
        if p.exists():
            v5m = json.loads(p.read_text(encoding="utf-8"))
            break
    # also read ckpt val
    v5_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink"
    thermal_v5_info = {"dir": str(v5_dir), "members": {}}
    if v5_dir.exists():
        import torch
        for ck in sorted(v5_dir.glob("pool_seed*.pt")):
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            thermal_v5_info["members"][ck.stem] = {
                "val_acc": float(blob.get("val_acc", -1)),
                "epoch": int(blob.get("epoch", -1)),
                "recipe": blob.get("recipe"),
            }

    mid_info = {}
    mp = TRACK / "baselines" / "skeleton_imu_v2" / "checkpoints_midfuse_ens_v3" / "metrics_mid_ens3.json"
    if mp.exists():
        mid_info = json.loads(mp.read_text(encoding="utf-8"))

    status = {
        "tag": "ir_v18_thermal_rethink_mid_ens3",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "thermal_v5_rethink": thermal_v5_info,
        "thermal_v5_metrics": v5m,
        "mid_ens3": mid_info,
        "best": best_same,
        "best_honest_sameT": best_same,
        "top10_sameT": [
            {k: r[k] for k in ("ir", "th", "mid", "mode", "full", "nested_fixed", "nested_retune", "honest_nested", "disagree_vs_v7", "th_solo", "mid_solo", "cfg") if k in r}
            for r in same[:10]
        ],
        "wrote_csv": bool(csv_path),
        "csv": csv_path,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "delta_vs_gate": {
            "best_full": float(best_same["full"] - GATE) if best_same else None,
            "honest_nested": float(best_same["honest_nested"] - GATE) if best_same else None,
        },
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # next_roi heuristics
    th_best = max((r.get("th_solo") or 0) for r in same) if same else 0
    mid_best = max((r.get("mid_solo") or 0) for r in same) if same else 0
    if not clears:
        status["next_roi"].append(
            f"MISS: best honest_sameT={best_same['honest_nested'] if best_same else None} full={best_same['full'] if best_same else None} dis={best_same['disagree_vs_v7'] if best_same else None} gate={GATE:.4f}"
        )
        if th_best < 0.60:
            status["next_roi"].append("Thermal rethink solo still <0.60 vs v2_trio~0.603 — stop more same R2+1D thermal seeds; try backbone/crop change or drop thermal ROI")
        else:
            status["next_roi"].append("Thermal rethink competitive — optional 2nd seed only if fuse complementary")
        if mid_best < 0.58:
            status["next_roi"].append("Mid ens3 helped but still <0.58; bone-concat MidFuse or more diverse seeds next")
        else:
            status["next_roi"].append("Mid upgrade landed; further Mid ROI diminishing vs IR diversity")
        status["next_roi"].append("Keep ir_v7; no weaker CSV")
    else:
        status["next_roi"].append("WIN: submit submission_ir_v18.csv; monitor public vs ir_v7 0.69154")

    out = ROOT / "metrics_ir_v18_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("outcome", "best_honest_sameT", "delta_vs_gate", "next_roi", "wrote_csv")}, indent=2, default=str), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
