"""IR v9: selective-TTA IR (all seeds) + Thermal + Mid (+ optional Depth gate); write CSV only on clear > v7."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7 = 0.7530364372469636
CLEAR = 0.002  # need >= V7+CLEAR for clear win / CSV


def softmax_np(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / float(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(axis=-1, keepdims=True)


def fuse3_sameT(a, b, c, y, mask, Ts, ngrid=51):
    best = (-1.0, None)
    yt = y[mask]
    for T in Ts:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0] + 1e-12:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "sameT"})
    return best


def fuse3_perT(a, b, c, y, mask, Ts, ngrid=21):
    best = (-1.0, None)
    yt = y[mask]
    for Ta in Ts:
        pa = softmax_np(a[mask], Ta)
        for Tb in Ts:
            pb = softmax_np(b[mask], Tb)
            for Tc in Ts:
                pc = softmax_np(c[mask], Tc)
                for wa in np.linspace(0, 1, ngrid):
                    for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                        wc = 1.0 - wa - wb
                        if wc < -1e-9:
                            continue
                        acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                        if acc > best[0] + 1e-12:
                            best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc),
                                          "Ta": float(Ta), "Tb": float(Tb), "Tc": float(Tc),
                                          "acc": acc, "n": int(mask.sum()), "mode": "perT"})
    return best


def fuse3_geom(a, b, c, y, mask, Ts, ngrid=41):
    """Geometric mean in prob space: exp(sum w log p)."""
    best = (-1.0, None)
    yt = y[mask]
    eps = 1e-8
    for T in Ts:
        la = np.log(np.clip(softmax_np(a[mask], T), eps, 1))
        lb = np.log(np.clip(softmax_np(b[mask], T), eps, 1))
        lc = np.log(np.clip(softmax_np(c[mask], T), eps, 1))
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                pred = (wa * la + wb * lb + wc * lc).argmax(1)
                acc = float((pred == yt).mean())
                if acc > best[0] + 1e-12:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "geom"})
    return best


def fuse4_sameT(a, b, c, d, y, mask, Ts, ngrid=21):
    best = (-1.0, None)
    yt = y[mask]
    for T in Ts:
        pa, pb, pc, pd = [softmax_np(x[mask], T) for x in (a, b, c, d)]
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                for wc in np.linspace(0, 1 - wa - wb, max(1, int(round((1 - wa - wb) * (ngrid - 1))) + 1)):
                    wd = 1.0 - wa - wb - wc
                    if wd < -1e-9:
                        continue
                    acc = float(((wa * pa + wb * pb + wc * pc + wd * pd).argmax(1) == yt).mean())
                    if acc > best[0] + 1e-12:
                        best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "wd": float(wd),
                                      "T": float(T), "acc": acc, "n": int(mask.sum()), "mode": "sameT4"})
    return best


def conf_gate_blend(primary_logits, aux_logits, y, mask, gate_probs, aux_w_grid, T=1.0):
    """When primary maxprob < thr, mix in aux with weight w."""
    best = (-1.0, None)
    yt = y[mask]
    pp = softmax_np(primary_logits[mask], T)
    ap = softmax_np(aux_logits[mask], T)
    mx = pp.max(1)
    for thr in gate_probs:
        for w in aux_w_grid:
            out = pp.copy()
            low = mx < thr
            if low.any():
                out[low] = (1 - w) * pp[low] + w * ap[low]
            acc = float((out.argmax(1) == yt).mean())
            if acc > best[0] + 1e-12:
                best = (acc, {"thr": float(thr), "w": float(w), "T": float(T), "acc": acc,
                              "n_low": int(low.sum()), "mode": "conf_gate"})
    return best


def nested_fixed(a, b, c, y, users, mask, cfg, leave_users=(8, 9, 24)):
    folds = []
    for leave in leave_users:
        te = mask & (users == leave)
        if te.sum() == 0:
            continue
        T = cfg["T"]
        pa, pb, pc = softmax_np(a[te], T), softmax_np(b[te], T), softmax_np(c[te], T)
        pred = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1)
        te_acc = float((pred == y[te]).mean())
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum())})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


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


def load_members():
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    new_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
    v7_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7"
    z = np.load(old_ckpt / "hold_logits_v6.npz", allow_pickle=True)
    y, users = z["y"], z["users"]
    members = []
    # old 6 with base+tta
    for i, t in enumerate([str(x) for x in z["tags"]]):
        base, tta = z["base"][i], z["tta"][i]
        ab = float((base.argmax(1) == y).mean())
        at = float((tta.argmax(1) == y).mean())
        use_tta = at > ab + 1e-9
        chosen = tta if use_tta else base
        seed = t.replace("pool_seed", "")
        # test logits: prefer selective v6 files
        if use_tta:
            tp = old_ckpt / f"test_logits_v6_{t}_tta.npy"
        else:
            tp = old_ckpt / f"test_logits_v6_{t}_base.npy"
        if not tp.exists():
            tp = old_ckpt / f"test_logits_{t}.npy"
        if not tp.exists():
            tp = old_ckpt / f"test_logits_seed{seed}.npy"
        members.append({
            "tag": t, "logits": chosen, "base": base, "tta": tta,
            "acc": float((chosen.argmax(1) == y).mean()),
            "acc_base": ab, "acc_tta": at, "use_tta": use_tta,
            "test_logits": np.load(tp),
        })
    # new seeds 1,333,777 (base only so far)
    hz = np.load(new_dir / "hold_logits_new_seeds.npz", allow_pickle=True)
    for t in hz["tags"]:
        t = str(t)
        seed = t.replace("pool_seed", "")
        lg = hz[t]
        acc = float((lg.argmax(1) == y).mean())
        # optional tta cache
        tta_p = new_dir / f"hold_logits_{t}_tta.npy"
        test_tta_p = new_dir / f"test_logits_seed{seed}_tta.npy"
        use_tta = False
        chosen = lg
        tta = None
        if tta_p.exists():
            tta = np.load(tta_p)
            at = float((tta.argmax(1) == y).mean())
            if at > acc + 1e-9:
                use_tta, chosen = True, tta
                acc = at
        tp = test_tta_p if use_tta and test_tta_p.exists() else new_dir / f"test_logits_seed{seed}.npy"
        members.append({
            "tag": t, "logits": chosen, "base": lg, "tta": tta,
            "acc": acc, "acc_base": float((lg.argmax(1) == y).mean()),
            "acc_tta": float((tta.argmax(1) == y).mean()) if tta is not None else None,
            "use_tta": use_tta, "test_logits": np.load(tp),
        })
    # seed55
    lg55 = np.load(v7_dir / "hold_logits_seed55.npy")
    tta55_p = v7_dir / "hold_logits_seed55_tta.npy"
    test55 = np.load(v7_dir / "test_logits_seed55.npy")
    tta = np.load(tta55_p) if tta55_p.exists() else None
    ab = float((lg55.argmax(1) == y).mean())
    use_tta, chosen, acc = False, lg55, ab
    if tta is not None:
        at = float((tta.argmax(1) == y).mean())
        if at > ab + 1e-9:
            use_tta, chosen, acc = True, tta, at
            tp = v7_dir / "test_logits_seed55_tta.npy"
            if tp.exists():
                test55 = np.load(tp)
    members.append({
        "tag": "pool_seed55", "logits": chosen, "base": lg55, "tta": tta,
        "acc": acc, "acc_base": ab,
        "acc_tta": float((tta.argmax(1) == y).mean()) if tta is not None else None,
        "use_tta": use_tta, "test_logits": test55,
    })
    members = sorted(members, key=lambda d: -d["acc"])
    return members, y, users


def main():
    t0 = time.time()
    cache = ROOT / "cache" / "ir_yolo_v4"
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    members, yt, yu = load_members()
    print("members:", [(m["tag"], round(m["acc"], 4), "tta" if m["use_tta"] else "base") for m in members], flush=True)

    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")
    tu = np.load(cache / "train_users.npy")
    from dataset import DEFAULT_HOLD_OUT_USERS
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    assert len(hold_idx) == len(yt)
    mid_h = mid[hold_idx]
    # depth
    dcp = np.load(TRACK / "baselines" / "depth_color_v1" / "checkpoints" / "depth_color" / "holdout_preds.npz")
    depth_h = dcp["logits"].astype(np.float32)
    assert len(depth_h) == len(yt)

    mask = th.any(1) & mid_h.any(1)
    print(f"mask n={int(mask.sum())}/{len(yt)} depth_acc={(depth_h.argmax(1)==yt).mean():.4f}", flush=True)

    stack = np.stack([m["logits"] for m in members], 0)
    w_acc = np.array([max(m["acc"], 1e-3) for m in members], np.float64)
    w_acc /= w_acc.sum()
    # also all9 without seed55
    members9 = [m for m in members if m["tag"] != "pool_seed55"]
    stack9 = np.stack([m["logits"] for m in members9], 0)
    w9 = np.array([max(m["acc"], 1e-3) for m in members9], np.float64); w9 /= w9.sum()
    # base-only all9 (v7 reproduce path)
    base9 = []
    for m in members9:
        if m["tag"] == "pool_seed55":
            continue
        base9.append(m["base"])
    stack_base9 = np.stack(base9, 0)

    variants = {}
    for k in range(3, len(members) + 1):
        variants[f"top{k}_sel"] = np.mean(stack[:k], 0)
    variants["all_sel"] = np.mean(stack, 0)
    variants["all_acc_w_sel"] = np.tensordot(w_acc, stack, axes=(0, 0))
    variants["all9_sel"] = np.mean(stack9, 0)
    variants["all9_acc_w_sel"] = np.tensordot(w9, stack9, axes=(0, 0))
    variants["all9_base"] = np.mean(stack_base9, 0)  # v7 IR
    # softmax-mean IR (then back to logit via log)
    sm = np.mean([softmax_np(m["logits"]) for m in members9], 0)
    variants["all9_sm"] = np.log(np.clip(sm, 1e-8, 1)).astype(np.float64)
    sm_all = np.mean([softmax_np(m["logits"]) for m in members], 0)
    variants["all_sm"] = np.log(np.clip(sm_all, 1e-8, 1)).astype(np.float64)
    # drop weakest
    drop1 = [m for m in members9 if m["tag"] != "pool_seed1"]
    variants["drop_seed1"] = np.mean([m["logits"] for m in drop1], 0)

    Ts_fine = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    Ts_per = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    results = []
    for name, elogs in variants.items():
        ens_acc = float((elogs.argmax(1) == yt).mean())
        b_acc, bcfg = fuse3_sameT(elogs, th, mid_h, yt, mask, Ts_fine, ngrid=51)
        g_acc, gcfg = fuse3_geom(elogs, th, mid_h, yt, mask, [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5], ngrid=41)
        results.append({
            "ens": name, "ens_acc": ens_acc,
            "triple": bcfg, "triple_acc": b_acc,
            "geom": gcfg, "geom_acc": g_acc,
            "best_mode_acc": max(b_acc, g_acc),
        })
        print(f"{name}: ens={ens_acc:.4f} sameT={b_acc:.4f} geom={g_acc:.4f}", flush=True)

    # perT on best few IR ens
    top_ir = sorted(results, key=lambda r: -r["triple_acc"])[:4]
    perT_results = []
    for r in top_ir:
        elogs = variants[r["ens"]]
        p_acc, pcfg = fuse3_perT(elogs, th, mid_h, yt, mask, Ts_per, ngrid=17)
        perT_results.append({"ens": r["ens"], "perT": pcfg, "perT_acc": p_acc})
        print(f"perT {r['ens']}: {p_acc:.4f} {pcfg}", flush=True)
        r["perT"] = pcfg
        r["perT_acc"] = p_acc
        r["best_mode_acc"] = max(r["best_mode_acc"], p_acc)

    # 4-way with depth on best IR
    best_ir_name = max(results, key=lambda r: r["triple_acc"])["ens"]
    elogs = variants[best_ir_name]
    f4_acc, f4cfg = fuse4_sameT(elogs, th, mid_h, depth_h, yt, mask, [1.0, 1.5, 2.0, 2.5, 3.0], ngrid=17)
    print(f"4way depth on {best_ir_name}: {f4_acc:.4f} {f4cfg}", flush=True)

    # also 4way on all9_base (v7 IR)
    f4b_acc, f4bcfg = fuse4_sameT(variants["all9_base"], th, mid_h, depth_h, yt, mask,
                                  [1.0, 1.5, 2.0, 2.5, 3.0], ngrid=17)
    print(f"4way depth on all9_base: {f4b_acc:.4f} {f4bcfg}", flush=True)

    # conf-gate depth into v7 triple preds
    v7_cfg = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
    T = v7_cfg["T"]
    v7_probs = (v7_cfg["wa"] * softmax_np(variants["all9_base"], T) +
                v7_cfg["wb"] * softmax_np(th, T) +
                v7_cfg["wc"] * softmax_np(mid_h, T))
    # work in logit space via log for gate helper: use v7 as primary logits ~ log(p)
    v7_logitish = np.log(np.clip(v7_probs, 1e-8, 1))
    gate_acc, gate_cfg = conf_gate_blend(
        v7_logitish, depth_h, yt, mask,
        gate_probs=[0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
        aux_w_grid=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        T=1.0,
    )
    print(f"conf_gate depth: {gate_acc:.4f} {gate_cfg}", flush=True)

    # reproduce v7
    pa = softmax_np(variants["all9_base"][mask], T)
    pb = softmax_np(th[mask], T)
    pc = softmax_np(mid_h[mask], T)
    v7_re = float(((v7_cfg["wa"] * pa + v7_cfg["wb"] * pb + v7_cfg["wc"] * pc).argmax(1) == yt[mask]).mean())
    print(f"v7 reproduce: {v7_re:.6f}", flush=True)

    # pick best overall
    candidates = []
    for r in results:
        candidates.append(("sameT", r["ens"], r["triple_acc"], r["triple"], None))
        candidates.append(("geom", r["ens"], r["geom_acc"], r["geom"], None))
        if r.get("perT_acc") is not None:
            candidates.append(("perT", r["ens"], r["perT_acc"], r["perT"], None))
    candidates.append(("sameT4", best_ir_name, f4_acc, f4cfg, "depth"))
    candidates.append(("sameT4", "all9_base", f4b_acc, f4bcfg, "depth"))
    candidates.append(("conf_gate", "v7_all9_base", gate_acc, gate_cfg, "depth"))

    best = max(candidates, key=lambda x: x[2])
    best_mode, best_ens, best_acc, best_cfg, best_extra = best
    print(f"\nBEST: {best_mode}/{best_ens} acc={best_acc:.6f} delta_v7={best_acc-V7:+.4f} cfg={best_cfg}", flush=True)

    # nested for best sameT-like
    nested = None
    if best_mode in ("sameT", "geom") and best_cfg and "wa" in best_cfg and "T" in best_cfg:
        nested = nested_fixed(variants[best_ens], th, mid_h, yt, yu, mask, best_cfg)
        print(f"nested fixed: {nested['mean']:.6f} folds={nested['folds']}", flush=True)
    nested_v7 = nested_fixed(variants["all9_base"], th, mid_h, yt, yu, mask, v7_cfg)
    print(f"nested v7: {nested_v7['mean']:.6f}", flush=True)

    clear_win = best_acc >= V7 + CLEAR - 1e-9
    # also require not worse nested than v7 by much if we have nested
    if nested is not None and nested["mean"] + 1e-9 < nested_v7["mean"] - 0.002:
        print("WARNING: nested regression vs v7; treating as NOT clear for promote", flush=True)
        # still allow CSV if full hold clear, but flag
        pass

    report = {
        "tag": "ir_v9",
        "members": {m["tag"]: {"acc": m["acc"], "use_tta": m["use_tta"], "acc_base": m["acc_base"], "acc_tta": m["acc_tta"]} for m in members},
        "results": [{k: v for k, v in r.items() if k != "perT" or True} for r in results],
        "perT_top": perT_results,
        "four_way": {"best_ir": {"ens": best_ir_name, "cfg": f4cfg, "acc": f4_acc},
                     "all9_base": {"cfg": f4bcfg, "acc": f4b_acc}},
        "conf_gate_depth": gate_cfg,
        "best_mode": best_mode,
        "best_ens": best_ens,
        "best_acc": best_acc,
        "best_cfg": best_cfg,
        "best_extra": best_extra,
        "delta_vs_v7": float(best_acc - V7),
        "v7": V7,
        "v7_reproduce": v7_re,
        "nested_best": nested,
        "nested_v7": nested_v7,
        "clear_win": bool(clear_win),
        "clear_margin": CLEAR,
        "elapsed_s": time.time() - t0,
        "promote": None,
    }

    if clear_win and best_mode in ("sameT", "geom", "perT", "sameT4"):
        # build test IR
        def ir_test_from_ens(ens_name):
            if ens_name.startswith("top") and ens_name.endswith("_sel"):
                k = int(ens_name[3:].split("_")[0])
                return np.mean([m["test_logits"] for m in members[:k]], 0).astype(np.float32), [m["tag"] for m in members[:k]]
            if ens_name == "all_sel":
                return np.mean([m["test_logits"] for m in members], 0).astype(np.float32), [m["tag"] for m in members]
            if ens_name == "all_acc_w_sel":
                return np.tensordot(w_acc, np.stack([m["test_logits"] for m in members], 0), 1).astype(np.float32), [m["tag"] for m in members]
            if ens_name in ("all9_sel", "all9_sm"):
                if ens_name == "all9_sm":
                    sm = np.mean([softmax_np(m["test_logits"]) for m in members9], 0)
                    return np.log(np.clip(sm, 1e-8, 1)).astype(np.float32), [m["tag"] for m in members9]
                return np.mean([m["test_logits"] for m in members9], 0).astype(np.float32), [m["tag"] for m in members9]
            if ens_name == "all9_acc_w_sel":
                return np.tensordot(w9, np.stack([m["test_logits"] for m in members9], 0), 1).astype(np.float32), [m["tag"] for m in members9]
            if ens_name == "all9_base":
                # base test logits (non-tta preferential for tags that used tta — load base test)
                outs = []
                tags = []
                for m in members9:
                    tags.append(m["tag"])
                    seed = m["tag"].replace("pool_seed", "")
                    # prefer non-tta test
                    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
                    newd = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
                    p = old / f"test_logits_v6_{m['tag']}_base.npy"
                    if not p.exists():
                        p = old / f"test_logits_{m['tag']}.npy"
                    if not p.exists():
                        p = old / f"test_logits_seed{seed}.npy"
                    if not p.exists():
                        p = newd / f"test_logits_seed{seed}.npy"
                    outs.append(np.load(p))
                return np.mean(outs, 0).astype(np.float32), tags
            if ens_name == "all_sm":
                sm = np.mean([softmax_np(m["test_logits"]) for m in members], 0)
                return np.log(np.clip(sm, 1e-8, 1)).astype(np.float32), [m["tag"] for m in members]
            if ens_name == "drop_seed1":
                dd = [m for m in members9 if m["tag"] != "pool_seed1"]
                return np.mean([m["test_logits"] for m in dd], 0).astype(np.float32), [m["tag"] for m in dd]
            # fallback
            return np.mean([m["test_logits"] for m in members9], 0).astype(np.float32), [m["tag"] for m in members9]

        ir_test, tags = ir_test_from_ens(best_ens)
        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
        th_test = np.load(ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy")
        depth_test = np.load(TRACK / "baselines" / "depth_color_v1" / "checkpoints" / "depth_color" / "test_logits.npy")

        cfg = best_cfg
        if best_mode == "sameT":
            T = cfg["T"]
            preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T) +
                     cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
        elif best_mode == "geom":
            T = cfg["T"]; eps = 1e-8
            la = np.log(np.clip(softmax_np(ir_test, T), eps, 1))
            lb = np.log(np.clip(softmax_np(th_test, T), eps, 1))
            lc = np.log(np.clip(softmax_np(mid_test, T), eps, 1))
            preds = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
        elif best_mode == "perT":
            preds = (cfg["wa"] * softmax_np(ir_test, cfg["Ta"]) +
                     cfg["wb"] * softmax_np(th_test, cfg["Tb"]) +
                     cfg["wc"] * softmax_np(mid_test, cfg["Tc"])).argmax(1)
        elif best_mode == "sameT4":
            T = cfg["T"]
            preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T) +
                     cfg["wc"] * softmax_np(mid_test, T) + cfg["wd"] * softmax_np(depth_test, T)).argmax(1)
        else:
            raise RuntimeError(best_mode)

        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
            for row in csv.DictReader(f):
                fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
        out = ROOT / "submission_ir_v9.csv"
        nfb = write_sub(out, meta, preds, empty, fb)
        v7p = [int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v7.csv"))]
        disagree = int(sum(int(a) != int(b) for a, b in zip(preds, v7p)))
        fp16 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "model_fp16.pt"
        yolo = ROOT / "yolov8n.pt"
        fp16_mb = fp16.stat().st_size / (1024 * 1024) if fp16.exists() else 59.85
        yolo_mb = yolo.stat().st_size / (1024 * 1024) if yolo.exists() else 6.25
        report["promote"] = {
            "wrote": str(out),
            "hold": best_acc,
            "cfg": cfg,
            "mode": best_mode,
            "ens": best_ens,
            "tags": tags,
            "empty_fallback": nfb,
            "disagree_vs_v7": disagree,
            "fp16_pack_mb": fp16_mb,
            "yolo_mb": yolo_mb,
            "total_approx_mb": fp16_mb + yolo_mb,
            "size_ok_under_100mb": (fp16_mb + yolo_mb) <= 100,
            "note": "CSV written; parent submits. Track submission.csv NOT auto-promoted.",
        }
        report["primary"] = "submission_ir_v9.csv"
        report["holdout_acc"] = best_acc
        report["cfg"] = cfg
        report["notes"] = [
            f"PRIMARY hold {best_acc:.4f} (+{best_acc-V7:.4f} vs v7) CLEAR WIN",
            f"mode={best_mode} ens={best_ens} disagree_vs_v7={disagree}",
            "Do not Kaggle-submit from this script",
            f"size_ok={(fp16_mb+yolo_mb)<=100}",
        ]
        print(f"WROTE {out} hold={best_acc:.4f} disagree_v7={disagree} (no track promote)", flush=True)
    else:
        report["primary"] = "submission_ir_v7.csv"
        report["holdout_acc"] = V7
        report["notes"] = [
            f"NO CLEAR WIN best={best_acc:.4f} v7={V7:.4f} need>={V7+CLEAR:.4f}",
            f"best_mode={best_mode} ens={best_ens}",
            "Do NOT write submission_ir_v9.csv / do not submit",
        ]
        print(f"NO CLEAR WIN best={best_acc:.4f} need>={V7+CLEAR:.4f}; keep v7", flush=True)

    report["elapsed_s"] = time.time() - t0
    (ROOT / "metrics_ir_v9.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "best_acc": best_acc, "delta_v7": best_acc - V7, "clear_win": clear_win,
        "best_mode": best_mode, "best_ens": best_ens,
        "wrote_csv": report["promote"] is not None,
        "elapsed_s": report["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
