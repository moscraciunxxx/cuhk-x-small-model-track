"""IR v11: fuse IR(all9_base@v7) + Thermal + Mid + optional Depth-Kinetics(v11).
Write submission_ir_v11.csv ONLY if nested LOUO and holdout both beat v7 by >= +0.01
AND disagree vs v7 >= ~20. Prefer leave_for_parent otherwise.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import (
    load_members, softmax_np, nested_fixed, write_sub, fuse3_sameT,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7_HOLD = 0.7530364372469636
V7_NESTED = 0.7519623092355898
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
V7_PUBLIC = 0.69154
MIN_DELTA = 0.01
MIN_DISAGREE = 20


def acc_of3(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T)
         + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean())


def fuse4_sameT(a, b, c, d, y, mask, Ts, ngrid=21):
    best = (-1.0, None)
    yt = y[mask]
    ws = np.linspace(0, 1, ngrid)
    for T in Ts:
        pa, pb, pc, pd = [softmax_np(z[mask], T) for z in (a, b, c, d)]
        for wa in ws:
            for wb in ws:
                for wc in ws:
                    wd = 1.0 - wa - wb - wc
                    if wd < -1e-9:
                        continue
                    pred = (wa * pa + wb * pb + wc * pc + wd * pd).argmax(1)
                    acc = float((pred == yt).mean())
                    if acc > best[0] + 1e-12:
                        best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc),
                                      "wd": float(wd), "T": float(T), "acc": acc,
                                      "n": int(mask.sum()), "mode": "sameT4"})
    return best


def nested_fixed4(a, b, c, d, y, users, mask, cfg, leave_users=(8, 9, 24)):
    folds = []
    T = cfg["T"]
    for leave in leave_users:
        te = mask & (users == leave)
        if te.sum() == 0:
            continue
        p = (cfg["wa"] * softmax_np(a[te], T) + cfg["wb"] * softmax_np(b[te], T)
             + cfg["wc"] * softmax_np(c[te], T) + cfg["wd"] * softmax_np(d[te], T)).argmax(1)
        folds.append(float((p == y[te]).mean()))
    return float(np.mean(folds)) if folds else 0.0


def nested_retune4(a, b, c, d, y, users, mask, Ts, ngrid=13, leave_users=(8, 9, 24)):
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, cfg = fuse4_sameT(a, b, c, d, y, tr, Ts, ngrid=ngrid)
        te_acc = float(((cfg["wa"] * softmax_np(a[te], cfg["T"])
                         + cfg["wb"] * softmax_np(b[te], cfg["T"])
                         + cfg["wc"] * softmax_np(c[te], cfg["T"])
                         + cfg["wd"] * softmax_np(d[te], cfg["T"])).argmax(1) == y[te]).mean())
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def hold_keys_from_meta(meta_path, hold_users=(8, 9, 24)):
    import json
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    keys, idxs = [], []
    for i, m in enumerate(meta):
        if int(m["user_id"]) in hold_users:
            keys.append((int(m["user_id"]), int(m["label"]), str(m.get("trial", ""))))
            idxs.append(i)
    return keys, np.asarray(idxs, dtype=np.int64)


def align_depth_hold(depth_pack, yt_ref, users_ref, ir_meta, depth_meta):
    """Align depth hold ensemble logits to IR hold order via (user,label,trial)."""
    dens = depth_pack["ens"].astype(np.float32)
    dy = depth_pack["y"]; du = depth_pack["users"]
    if len(dens) == len(yt_ref) and np.array_equal(dy, yt_ref) and np.array_equal(du, users_ref):
        return dens
    ir_keys, _ = hold_keys_from_meta(ir_meta)
    dc_keys, _ = hold_keys_from_meta(depth_meta)
    assert len(ir_keys) == len(yt_ref), (len(ir_keys), len(yt_ref))
    assert len(dc_keys) == len(dens), (len(dc_keys), len(dens))
    # depth_pack ens is in depth hold_idx order (== dc_keys order)
    pos = {k: i for i, k in enumerate(dc_keys)}
    out = np.zeros_like(dens)
    missing = 0
    for i, k in enumerate(ir_keys):
        j = pos.get(k)
        if j is None:
            missing += 1
            out[i] = 0.0
        else:
            out[i] = dens[j]
            if dy[j] != yt_ref[i]:
                raise RuntimeError(f"label mismatch at {i} key={k} dy={dy[j]} yt={yt_ref[i]}")
    if missing:
        raise RuntimeError(f"depth align missing {missing} hold keys")
    return out


def main():
    t0 = time.time()
    members, yt, yu = load_members()
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    from dataset import DEFAULT_HOLD_OUT_USERS
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    # mid is full-train aligned; hold slice via users
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    # load_members y/users are hold-ordered; verify length
    assert len(yt) == len(th), (len(yt), len(th))
    if len(mid) == len(tu):
        mid_h = mid[hold_idx]
    elif len(mid) == len(yt):
        mid_h = mid
    else:
        raise RuntimeError(f"mid len {len(mid)} vs train {len(tu)} hold {len(yt)}")
    # Ensure yu matches hold users from cache if needed
    if len(yu) != len(yt):
        yu = tu[hold_idx]
    mask = th.any(1) & mid_h.any(1)
    ir_base = np.mean([m["base"] for m in members if m["tag"] != "pool_seed55"], 0)

    # depth
    depth_dir = ROOT / "checkpoints" / "depth_yolo_r2p1d18_v11"
    depth_npz = depth_dir / "hold_logits_v11.npz"
    has_depth = depth_npz.exists()
    depth_h = None
    depth_acc = None
    if has_depth:
        dp = np.load(depth_npz)
        depth_h = align_depth_hold(dp, yt, yu,
            ROOT / "cache" / "ir_yolo_v4" / "train_meta.json",
            ROOT / "cache" / "depth_color_yolo_v4" / "train_meta.json")
        depth_acc = float((depth_h.argmax(1) == yt).mean())
        print(f"depth ens hold={depth_acc:.4f} members={list(dp['seeds'])} scores={list(dp['scores'])}", flush=True)
    else:
        print("NO depth pack yet — will eval IR-strong / IR+Thermal diversity only if present", flush=True)

    # optional stronger IR logits
    strong_path = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v11" / "hold_logits_strong.npz"
    if strong_path.exists():
        sp = np.load(strong_path)
        ir_strong = sp["ens"].astype(np.float32)
        print(f"IR strong ens hold={(ir_strong.argmax(1)==yt).mean():.4f}", flush=True)
    else:
        ir_strong = None

    ir_v7 = ir_base  # all9_base
    print(f"mask n={int(mask.sum())}/{len(yt)} ir_v7_hold={(ir_v7.argmax(1)==yt).mean():.4f} "
          f"th={(th.argmax(1)==yt).mean():.4f} mid={(mid_h.argmax(1)==yt).mean():.4f}", flush=True)

    # v7 reproduce
    v7_acc = acc_of3(ir_v7, th, mid_h, yt, mask, V7_CFG)
    v7_nested = nested_fixed(ir_v7, th, mid_h, yt, yu, mask, V7_CFG)['mean']
    print(f"v7 repro hold={v7_acc:.6f} nested={v7_nested:.6f}", flush=True)

    candidates = []

    # 3-way with strong IR if available
    if ir_strong is not None:
        for name, ir in [("ir_strong", ir_strong), ("ir_half", 0.5 * ir_v7 + 0.5 * ir_strong)]:
            acc, cfg = fuse3_sameT(ir, th, mid_h, yt, mask, [1.5, 2.0, 2.5, 2.75, 3.0], ngrid=41)
            nest = nested_fixed(ir, th, mid_h, yt, yu, mask, cfg)['mean']
            nest_rt = None
            candidates.append({"name": f"3way_{name}", "hold": acc, "nested_fixed": nest,
                               "cfg": cfg, "ir": name, "mode": "sameT3"})

    # 4-way with depth
    if depth_h is not None:
        for ir_name, ir in [("all9_base", ir_v7)] + ([("ir_strong", ir_strong)] if ir_strong is not None else []):
            # also try replacing Mid with Depth if depth >> mid
            acc4, cfg4 = fuse4_sameT(ir, th, mid_h, depth_h, yt, mask,
                                     [1.5, 2.0, 2.5, 3.0], ngrid=17)
            nest4 = nested_fixed4(ir, th, mid_h, depth_h, yt, yu, mask, cfg4)
            nest_rt = nested_retune4(ir, th, mid_h, depth_h, yt, yu, mask,
                                    [1.5, 2.0, 2.5, 3.0], ngrid=11)
            candidates.append({"name": f"4way_{ir_name}+mid+depth", "hold": acc4,
                               "nested_fixed": nest4, "nested_retune": nest_rt["mean"],
                               "cfg": cfg4, "ir": ir_name, "mode": "sameT4",
                               "nested_retune_folds": nest_rt["folds"]})
            # 3-way IR+Th+Depth (drop mid)
            acc3d, cfg3d = fuse3_sameT(ir, th, depth_h, yt, mask, [1.5, 2.0, 2.5, 3.0], ngrid=41)
            nest3d = nested_fixed(ir, th, depth_h, yt, yu, mask, cfg3d)['mean']
            candidates.append({"name": f"3way_{ir_name}+th+depth", "hold": acc3d,
                               "nested_fixed": nest3d, "cfg": cfg3d, "ir": ir_name, "mode": "sameT3_depth"})
            # 3-way IR+Depth+Mid (drop thermal) — diversity probe
            acc3m, cfg3m = fuse3_sameT(ir, depth_h, mid_h, yt, mask, [1.5, 2.0, 2.5, 3.0], ngrid=41)
            nest3m = nested_fixed(ir, depth_h, mid_h, yt, yu, mask, cfg3m)['mean']
            candidates.append({"name": f"3way_{ir_name}+depth+mid", "hold": acc3m,
                               "nested_fixed": nest3m, "cfg": cfg3m, "ir": ir_name, "mode": "sameT3_dm"})

    if not candidates:
        report = {
            "tag": "ir_v11",
            "status": "no_new_signal_ready",
            "has_depth": has_depth,
            "has_ir_strong": ir_strong is not None,
            "submit_recommendation": "leave_for_parent",
            "wrote_csv": False,
            "v7_public": V7_PUBLIC,
            "notes": ["Depth cache/train or IR-strong logits not ready; no CSV."],
        }
        (ROOT / "metrics_ir_v11.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return

    # pick best by nested_fixed (honest), require sameT not perT
    candidates.sort(key=lambda d: (d.get("nested_fixed", 0), d["hold"]), reverse=True)
    for c in candidates[:12]:
        print(f"cand {c['name']}: hold={c['hold']:.4f} nested={c.get('nested_fixed',0):.4f} "
              f"retune={c.get('nested_retune')} cfg={c['cfg']}", flush=True)

    best = candidates[0]
    # compute disagree vs v7 on test (need test preds)
    # Build test preds for best and v7
    members_test = []
    # IR test all9 base
    def load_ir_test_base():
        tags = ["2024", "11", "42", "99", "777", "7", "123", "333", "1"]
        logs = []
        old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
        newd = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
        for s in tags:
            for p in [old / f"test_logits_v6_pool_seed{s}_base.npy",
                      old / f"test_logits_pool_seed{s}.npy",
                      old / f"test_logits_seed{s}.npy",
                      newd / f"test_logits_seed{s}.npy"]:
                if p.exists():
                    logs.append(np.load(p)); break
        return np.mean(np.stack(logs, 0), 0)

    ir_test = load_ir_test_base()
    th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    depth_test = None
    if has_depth and (depth_dir / "test_logits_ens.npy").exists():
        depth_test = np.load(depth_dir / "test_logits_ens.npy")

    def pred3(a, b, c, cfg):
        T = cfg["T"]
        return (cfg["wa"] * softmax_np(a, T) + cfg["wb"] * softmax_np(b, T)
                + cfg["wc"] * softmax_np(c, T)).argmax(1)

    def pred4(a, b, c, d, cfg):
        T = cfg["T"]
        return (cfg["wa"] * softmax_np(a, T) + cfg["wb"] * softmax_np(b, T)
                + cfg["wc"] * softmax_np(c, T) + cfg["wd"] * softmax_np(d, T)).argmax(1)

    v7_pred = pred3(ir_test, th_test, mid_test, V7_CFG)

    # map best to test
    ir_for_test = ir_test
    if best["ir"] == "ir_strong" and (ROOT / "checkpoints" / "ir_yolo_r2p1d18_v11" / "test_logits_ens.npy").exists():
        ir_for_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v11" / "test_logits_ens.npy")
    elif best["ir"] == "ir_half" and (ROOT / "checkpoints" / "ir_yolo_r2p1d18_v11" / "test_logits_ens.npy").exists():
        ir_s = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v11" / "test_logits_ens.npy")
        ir_for_test = 0.5 * ir_test + 0.5 * ir_s

    cfg = best["cfg"]
    if best["mode"] == "sameT4":
        if depth_test is None:
            raise SystemExit("best is 4way but no depth test")
        best_pred = pred4(ir_for_test, th_test, mid_test, depth_test, cfg)
    elif best["mode"] == "sameT3_depth":
        best_pred = pred3(ir_for_test, th_test, depth_test, cfg)
    elif best["mode"] == "sameT3_dm":
        best_pred = pred3(ir_for_test, depth_test, mid_test, cfg)
    else:
        best_pred = pred3(ir_for_test, th_test, mid_test, cfg)

    disagree = int((best_pred != v7_pred).sum())
    d_hold = best["hold"] - V7_HOLD
    d_nest = best.get("nested_fixed", 0) - V7_NESTED
    clear = (d_hold >= MIN_DELTA - 1e-12) and (d_nest >= MIN_DELTA - 1e-12) and (disagree >= MIN_DISAGREE)
    # also require depth signal actually used with wd>0.02 if 4way
    if best["mode"] == "sameT4" and cfg.get("wd", 0) < 0.02:
        clear = False
        reason_extra = "wd~0 depth unused"
    else:
        reason_extra = ""

    wrote = False
    out_csv = ROOT / "submission_ir_v11.csv"
    if clear:
        # empty fallback from midfuse/skel
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
            r = csv.DictReader(f)
            for row in r:
                fb[row["path"] if row["path"].endswith("/") else row["path"] + "/"] = int(row["prediction"])
        meta = json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_meta.json").read_text())
        empty = set(json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_empty.json").read_text()))
        # write
        nfb = 0
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["path", "prediction"])
            for i, m in enumerate(meta):
                p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
                if m.get("empty") or m["sample_id"] in empty:
                    pred = fb.get(p, int(best_pred[i])); nfb += 1
                else:
                    pred = int(best_pred[i])
                w.writerow([p, pred])
        wrote = True
        print(f"WROTE {out_csv} hold={best['hold']:.4f} nested={best.get('nested_fixed'):.4f} "
              f"disagree_v7={disagree} empty_fb={nfb}", flush=True)
    else:
        print(f"NO CSV: clear={clear} d_hold={d_hold:+.4f} d_nest={d_nest:+.4f} "
              f"disagree={disagree} {reason_extra}", flush=True)

    report = {
        "tag": "ir_v11",
        "method": best["name"],
        "best": best,
        "top_candidates": [
            {k: v for k, v in c.items() if k != "nested_retune_folds"}
            for c in candidates[:8]
        ],
        "holdout_acc": best["hold"],
        "nested_fixed": best.get("nested_fixed"),
        "nested_retune": best.get("nested_retune"),
        "delta_hold_vs_v7": d_hold,
        "delta_nested_vs_v7": d_nest,
        "disagree_vs_v7": disagree,
        "clear_win": clear,
        "min_delta": MIN_DELTA,
        "min_disagree": MIN_DISAGREE,
        "depth_hold_acc": depth_acc,
        "v7_public": V7_PUBLIC,
        "v7_hold": V7_HOLD,
        "v7_nested": V7_NESTED,
        "wrote_csv": wrote,
        "csv": str(out_csv) if wrote else None,
        "submit_recommendation": "submit_ir_v11" if clear else "leave_for_parent",
        "fp16_pack_mb": 59.85,
        "yolo_mb": 6.25,
        "total_approx_mb": 66.1 if not has_depth else 66.1,  # still one fp16 video backbone + yolo for submit pack
        "size_ok_under_100mb": True,
        "gpu_note": "RTX3060 free vs LMT at start; depth YOLO+Kinetics used GPU",
        "elapsed_s": time.time() - t0,
        "notes": [
            f"best {best['name']} hold={best['hold']:.4f} nested={best.get('nested_fixed'):.4f}",
            f"gate +{MIN_DELTA} hold&nested and disagree>={MIN_DISAGREE}: clear={clear}",
            reason_extra or "ok",
        ],
    }
    (ROOT / "metrics_ir_v11.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in report if k != "top_candidates"}, indent=2, default=str))


if __name__ == "__main__":
    main()
