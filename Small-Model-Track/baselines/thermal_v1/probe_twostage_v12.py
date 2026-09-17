"""v12 probe: two-stage / confidence-gated IR -> specialist (Depth/Thermal/Mid).
No CSV unless hold+nested both >= v7+0.01 and disagree>=20.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from write_ir_v11 import align_depth_hold, hold_keys_from_meta, nested_fixed4

ROOT = Path(__file__).resolve().parent
V7_HOLD = 0.7530364372469636
V7_NESTED = 0.7519623092355898
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
MIN_DELTA = 0.01
MIN_DISAGREE = 20


def soft_pred(z, T=1.0):
    return softmax_np(z, T)


def fuse_v7(ir, th, mid, cfg=V7_CFG):
    T = cfg["T"]
    return cfg["wa"] * soft_pred(ir, T) + cfg["wb"] * soft_pred(th, T) + cfg["wc"] * soft_pred(mid, T)


def conf_stats(p, y, name):
    mx = p.max(1)
    pred = p.argmax(1)
    ok = pred == y
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (mx >= lo) & (mx < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": int(m.sum()),
            "acc": float(ok[m].mean()),
            "frac": float(m.mean()),
        })
    return {"name": name, "overall_acc": float(ok.mean()), "mean_maxp": float(mx.mean()), "bins": rows}


def gate_replace(primary_p, aux_p, y, mask, thrs, mode="replace"):
    """When primary maxp < thr: replace or blend with aux."""
    best = (-1.0, None)
    yt = y[mask]
    pp = primary_p[mask]
    ap = aux_p[mask]
    mx = pp.max(1)
    for thr in thrs:
        low = mx < thr
        out = pp.copy()
        if mode == "replace":
            out[low] = ap[low]
            cfg = {"thr": float(thr), "mode": mode, "n_low": int(low.sum())}
            acc = float((out.argmax(1) == yt).mean())
            if acc > best[0]:
                best = (acc, {**cfg, "acc": acc})
        else:
            for w in np.linspace(0.2, 1.0, 9):
                out2 = pp.copy()
                out2[low] = (1 - w) * pp[low] + w * ap[low]
                acc = float((out2.argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"thr": float(thr), "w": float(w), "mode": "blend",
                                  "n_low": int(low.sum()), "acc": acc})
    return best


def gate_specialist_argmax(primary_p, specs, y, mask, thrs):
    """low-conf: pick specialist with highest maxprob among specs (or blend equal)."""
    best = (-1.0, None)
    yt = y[mask]
    pp = primary_p[mask]
    mx = pp.max(1)
    sp = {k: v[mask] for k, v in specs.items()}
    for thr in thrs:
        low = mx < thr
        out = pp.copy()
        if low.any():
            # choose specialist with highest confidence on each low sample
            stack = np.stack([sp[k][low] for k in sp], 0)  # S,N,C
            conf = stack.max(-1)  # S,N
            which = conf.argmax(0)  # N
            chosen = stack[which, np.arange(low.sum())]
            out[low] = chosen
        acc = float((out.argmax(1) == yt).mean())
        if acc > best[0]:
            best = (acc, {"thr": float(thr), "mode": "maxconf_spec", "n_low": int(low.sum()),
                          "specs": list(specs), "acc": acc})
    return best


def apply_gate_blend(primary_p, aux_p, thr, w):
    out = primary_p.copy()
    low = primary_p.max(1) < thr
    out[low] = (1 - w) * primary_p[low] + w * aux_p[low]
    return out, low


def nested_gated(primary_p, aux_p, y, users, mask, thr, w, leave=(8, 9, 24)):
    folds = []
    for leave_u in leave:
        te = mask & (users == leave_u)
        if te.sum() == 0:
            continue
        out, _ = apply_gate_blend(primary_p, aux_p, thr, w)
        acc = float((out[te].argmax(1) == y[te]).mean())
        folds.append({"leave": int(leave_u), "te_acc": acc, "n": int(te.sum())})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def nested_retune_gate(primary_p, aux_p, y, users, mask, thrs, ws, leave=(8, 9, 24)):
    folds = []
    for leave_u in leave:
        te = mask & (users == leave_u)
        tr = mask & (users != leave_u)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # tune on train folds
        best = (-1.0, None)
        yt = y[tr]
        pp, ap = primary_p[tr], aux_p[tr]
        mx = pp.max(1)
        for thr in thrs:
            low = mx < thr
            for w in ws:
                out = pp.copy()
                if low.any():
                    out[low] = (1 - w) * pp[low] + w * ap[low]
                acc = float((out.argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"thr": float(thr), "w": float(w)})
        cfg = best[1]
        out_te, _ = apply_gate_blend(primary_p, aux_p, cfg["thr"], cfg["w"])
        te_acc = float((out_te[te].argmax(1) == y[te]).mean())
        folds.append({"leave": int(leave_u), "te_acc": te_acc, "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def disagree_count(p1, p2, mask):
    return int((p1[mask].argmax(1) != p2[mask].argmax(1)).sum())


def main():
    t0 = time.time()
    members, yt, yu = load_members()
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    cache = ROOT / "cache" / "ir_yolo_v4"
    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")
    tu = np.load(cache / "train_users.npy")
    from dataset import DEFAULT_HOLD_OUT_USERS
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid_h = mid[hold_idx] if len(mid) == len(tu) else mid
    assert len(yt) == len(th) == len(mid_h)
    if len(yu) != len(yt):
        yu = tu[hold_idx]

    # all9 base (exclude seed55 like v11)
    ir_base = np.mean([m["base"] for m in members if m["tag"] != "pool_seed55"], 0).astype(np.float32)
    # selective-tta all members (v7-ish IR ens)
    ir_sel = np.mean([m["logits"] for m in members if m["tag"] != "pool_seed55"], 0).astype(np.float32)
    # all including 55
    ir_all = np.mean([m["logits"] for m in members], 0).astype(np.float32)

    mask = th.any(1) & mid_h.any(1)
    print(f"hold n={len(yt)} mask={int(mask.sum())}", flush=True)
    print(f"IR all9_base={(ir_base.argmax(1)==yt).mean():.4f} sel={(ir_sel.argmax(1)==yt).mean():.4f} "
          f"all10={(ir_all.argmax(1)==yt).mean():.4f}", flush=True)
    print(f"Th={(th.argmax(1)==yt).mean():.4f} Mid={(mid_h.argmax(1)==yt).mean():.4f}", flush=True)

    # depth v11 aligned
    depth_pack = np.load(ROOT / "checkpoints" / "depth_yolo_r2p1d18_v11" / "hold_logits_v11.npz", allow_pickle=True)
    ir_meta = cache / "train_meta.json"
    depth_meta = ROOT / "cache" / "depth_color_yolo_v4" / "train_meta.json"
    depth_h = align_depth_hold(depth_pack, yt, yu, ir_meta, depth_meta)
    print(f"Depth ens={(depth_h.argmax(1)==yt).mean():.4f}", flush=True)

    # v7 fuse probs
    v7p = fuse_v7(ir_base, th, mid_h)
    v7_acc = float((v7p[mask].argmax(1) == yt[mask]).mean())
    print(f"v7 reconstruct hold={v7_acc:.4f} (target {V7_HOLD:.4f})", flush=True)

    # stream probs at T=2.5
    T = 2.5
    pir = soft_pred(ir_base, T)
    pth = soft_pred(th, T)
    pmid = soft_pred(mid_h, T)
    pdep = soft_pred(depth_h, T)

    # disagreement IR vs Depth
    ir_pred = pir.argmax(1)
    dep_pred = pdep.argmax(1)
    dis = ir_pred != dep_pred
    print(f"IR vs Depth disagree={int(dis.sum())}/{len(yt)} ({dis.mean():.3f})", flush=True)
    print(f"  on disagree: IR_acc={(ir_pred[dis]==yt[dis]).mean():.4f} "
          f"Dep_acc={(dep_pred[dis]==yt[dis]).mean():.4f}", flush=True)
    # when IR wrong
    ir_wrong = ir_pred != yt
    print(f"IR wrong n={int(ir_wrong.sum())}; Depth corrects={int((dep_pred[ir_wrong]==yt[ir_wrong]).sum())} "
          f"({(dep_pred[ir_wrong]==yt[ir_wrong]).mean():.3f})", flush=True)
    th_pred = pth.argmax(1)
    print(f"IR wrong; Thermal corrects={int((th_pred[ir_wrong]==yt[ir_wrong]).sum())} "
          f"({(th_pred[ir_wrong]==yt[ir_wrong]).mean():.3f})", flush=True)
    mid_pred = pmid.argmax(1)
    print(f"IR wrong; Mid corrects={int((mid_pred[ir_wrong]==yt[ir_wrong]).sum())} "
          f"({(mid_pred[ir_wrong]==yt[ir_wrong]).mean():.3f})", flush=True)
    v7_wrong = v7p.argmax(1) != yt
    print(f"v7 wrong n={int(v7_wrong[mask].sum())}; Depth corrects among them="
          f"{int((dep_pred[mask & v7_wrong]==yt[mask & v7_wrong]).sum())}/"
          f"{int((mask & v7_wrong).sum())}", flush=True)

    report = {
        "v7_recon": v7_acc,
        "stream_acc": {
            "ir_base": float((ir_pred == yt).mean()),
            "th": float((th_pred == yt).mean()),
            "mid": float((mid_pred == yt).mean()),
            "depth": float((dep_pred == yt).mean()),
        },
        "ir_vs_depth_disagree": int(dis.sum()),
        "conf_bins": {
            "ir": conf_stats(pir, yt, "ir"),
            "v7": conf_stats(v7p, yt, "v7"),
            "depth": conf_stats(pdep, yt, "depth"),
        },
    }

    thrs = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    ws = np.linspace(0.2, 1.0, 9)

    # Gate on IR alone -> specialist
    attempts = []
    for aux_name, aux_p in [("depth", pdep), ("thermal", pth), ("mid", pmid),
                            ("th+dep", 0.7 * pth + 0.3 * pdep),
                            ("th+mid", 0.7 * pth + 0.3 * pmid),
                            ("dep+th+mid", (pth + pdep + pmid) / 3)]:
        for mode in ("replace", "blend"):
            acc, cfg = gate_replace(pir, aux_p, yt, mask, thrs, mode=mode)
            attempts.append({"primary": "ir", "aux": aux_name, **cfg})

    # Gate on v7 fuse -> specialist (two-stage: keep v7 for easy, specialist for hard)
    for aux_name, aux_p in [("depth", pdep), ("thermal", pth), ("mid", pmid),
                            ("th+dep", 0.7 * pth + 0.3 * pdep),
                            ("depth_only_if_better_conf", None),
                            ("th_dep_mid_eq", (pth + pdep + pmid) / 3),
                            ("0.5th+0.5dep", 0.5 * pth + 0.5 * pdep)]:
        if aux_p is None:
            # only replace when depth conf > v7 conf
            best = (-1.0, None)
            for thr in thrs:
                out = v7p.copy()
                low = v7p.max(1) < thr
                better = low & (pdep.max(1) > v7p.max(1))
                out[better] = pdep[better]
                acc = float((out[mask].argmax(1) == yt[mask]).mean())
                if acc > best[0]:
                    best = (acc, {"thr": float(thr), "mode": "replace_if_dep_conf_higher",
                                  "n_low": int(low.sum()), "n_swap": int(better.sum()), "acc": acc})
            attempts.append({"primary": "v7", "aux": aux_name, **best[1]})
            continue
        for mode in ("replace", "blend"):
            acc, cfg = gate_replace(v7p, aux_p, yt, mask, thrs, mode=mode)
            attempts.append({"primary": "v7", "aux": aux_name, **cfg})

    # maxconf specialist among th/dep/mid when v7 low
    acc, cfg = gate_specialist_argmax(v7p, {"th": pth, "dep": pdep, "mid": pmid}, yt, mask, thrs)
    attempts.append({"primary": "v7", "aux": "maxconf(th,dep,mid)", **cfg})

    # also try IR-margin gate: top1-top2
    def margin_gate(primary_p, aux_p, y, mask, margins, ws):
        best = (-1.0, None)
        yt_ = y[mask]
        pp = primary_p[mask]
        ap = aux_p[mask]
        part = np.partition(pp, -2, axis=1)
        marg = part[:, -1] - part[:, -2]
        for mthr in margins:
            low = marg < mthr
            for w in ws:
                out = pp.copy()
                if low.any():
                    out[low] = (1 - w) * pp[low] + w * ap[low]
                acc = float((out.argmax(1) == yt_).mean())
                if acc > best[0]:
                    best = (acc, {"margin_thr": float(mthr), "w": float(w), "mode": "margin_blend",
                                  "n_low": int(low.sum()), "acc": acc})
        return best

    for aux_name, aux_p in [("depth", pdep), ("thermal", pth), ("th+dep", 0.7*pth+0.3*pdep)]:
        acc, cfg = margin_gate(v7p, aux_p, yt, mask, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4], ws)
        attempts.append({"primary": "v7", "aux": aux_name, **cfg})

    attempts = sorted(attempts, key=lambda d: -d["acc"])
    print("\n=== TOP 15 gate attempts (hold, masked) ===", flush=True)
    for a in attempts[:15]:
        print(a, flush=True)

    # Take top candidates that beat v7 and evaluate nested
    nested_results = []
    seen = set()
    for a in attempts:
        if a["acc"] < V7_HOLD - 1e-6:
            continue
        key = (a["primary"], a["aux"], a.get("mode"), round(a.get("thr", a.get("margin_thr", -1)), 3),
               round(a.get("w", -1), 3))
        if key in seen:
            continue
        seen.add(key)
        # build aux probs
        aux_map = {
            "depth": pdep, "thermal": pth, "mid": pmid,
            "th+dep": 0.7 * pth + 0.3 * pdep,
            "th+mid": 0.7 * pth + 0.3 * pmid,
            "dep+th+mid": (pth + pdep + pmid) / 3,
            "th_dep_mid_eq": (pth + pdep + pmid) / 3,
            "0.5th+0.5dep": 0.5 * pth + 0.5 * pdep,
        }
        primary = v7p if a["primary"] == "v7" else pir
        if a["aux"] not in aux_map:
            continue
        aux = aux_map[a["aux"]]
        if a.get("mode") in ("blend", "margin_blend") or "w" in a:
            thr = a.get("thr", 1.0)
            w = a.get("w", 1.0)
            if a.get("mode") == "margin_blend":
                # approximate nested with fixed margin
                part = np.partition(primary, -2, axis=1)
                marg = part[:, -1] - part[:, -2]
                out = primary.copy()
                low = marg < a["margin_thr"]
                out[low] = (1 - w) * primary[low] + w * aux[low]
                # nested fixed
                folds = []
                for leave_u in (8, 9, 24):
                    te = mask & (yu == leave_u)
                    folds.append(float((out[te].argmax(1) == yt[te]).mean()))
                nest = {"mean": float(np.mean(folds)), "folds": folds}
                nest_rt = nested_retune_gate(primary, aux, yt, yu, mask, thrs, ws)  # thr-based approx
            else:
                nest = nested_gated(primary, aux, yt, yu, mask, thr, w)
                nest_rt = nested_retune_gate(primary, aux, yt, yu, mask, thrs, ws)
            out_full, low = apply_gate_blend(primary, aux, thr if a.get("mode") != "margin_blend" else 1.0, w)
            if a.get("mode") == "margin_blend":
                part = np.partition(primary, -2, axis=1)
                marg = part[:, -1] - part[:, -2]
                out_full = primary.copy()
                low = marg < a["margin_thr"]
                out_full[low] = (1 - w) * primary[low] + w * aux[low]
            dis_v7 = disagree_count(out_full, v7p, mask)
        elif a.get("mode") == "replace":
            thr = a["thr"]
            out_full = primary.copy()
            low = primary.max(1) < thr
            out_full[low] = aux[low]
            folds = []
            for leave_u in (8, 9, 24):
                te = mask & (yu == leave_u)
                folds.append(float((out_full[te].argmax(1) == yt[te]).mean()))
            nest = {"mean": float(np.mean(folds)), "folds": folds}
            nest_rt = nested_retune_gate(primary, aux, yt, yu, mask, thrs, [1.0])
            dis_v7 = disagree_count(out_full, v7p, mask)
        else:
            continue
        row = {
            **{k: a[k] for k in a},
            "nested_fixed": nest["mean"],
            "nested_retune": nest_rt["mean"] if isinstance(nest_rt, dict) else nest_rt,
            "delta_hold": a["acc"] - V7_HOLD,
            "delta_nested": nest["mean"] - V7_NESTED,
            "disagree_vs_v7": dis_v7,
            "clear_win": (a["acc"] >= V7_HOLD + MIN_DELTA and nest["mean"] >= V7_NESTED + MIN_DELTA
                          and dis_v7 >= MIN_DISAGREE),
        }
        nested_results.append(row)
        if len(nested_results) >= 25:
            break

    nested_results = sorted(nested_results, key=lambda d: (-d["clear_win"], -d["nested_fixed"], -d["acc"]))
    print("\n=== Nested evaluation of hold-winners ===", flush=True)
    for r in nested_results[:12]:
        print({k: r[k] for k in ("primary", "aux", "mode", "thr", "margin_thr", "w", "acc",
                                 "nested_fixed", "nested_retune", "delta_hold", "delta_nested",
                                 "disagree_vs_v7", "clear_win", "n_low") if k in r}, flush=True)

    # Oracle: when v7 wrong, if we could pick depth when depth right
    oracle = v7p.copy()
    swap = mask & v7_wrong & (dep_pred == yt)
    oracle[swap] = pdep[swap]
    print(f"\nOracle depth-on-v7-wrong-if-depth-right: acc={((oracle[mask].argmax(1)==yt[mask]).mean()):.4f} "
          f"n_swap={int(swap.sum())}", flush=True)

    # Class-wise: where depth beats IR
    from collections import Counter
    ir_ok = ir_pred == yt
    dep_ok = dep_pred == yt
    dep_helps = (~ir_ok) & dep_ok
    ir_helps = ir_ok & (~dep_ok)
    print(f"Depth helps (IR wrong, Dep right) n={int(dep_helps.sum())} by class top:", flush=True)
    print(Counter(yt[dep_helps].tolist()).most_common(10), flush=True)
    print(f"IR helps (Dep wrong, IR right) n={int(ir_helps.sum())}", flush=True)

    # Entropy / mutual complementarity of mid stream
    mid_ok = mid_pred == yt
    mid_helps_v7 = mask & v7_wrong & mid_ok
    print(f"Mid helps on v7-wrong: {int(mid_helps_v7.sum())}/{int((mask&v7_wrong).sum())}", flush=True)

    out = {
        "tag": "probe_twostage_v12",
        "v7": {"hold": V7_HOLD, "nested": V7_NESTED, "recon": v7_acc},
        "stream_acc": report["stream_acc"],
        "ir_vs_depth_disagree": report["ir_vs_depth_disagree"],
        "conf_bins": report["conf_bins"],
        "top_attempts_hold": attempts[:20],
        "nested_top": nested_results[:15],
        "best_clear": next((r for r in nested_results if r["clear_win"]), None),
        "any_clear_win": any(r["clear_win"] for r in nested_results),
        "oracle_depth_on_v7_wrong": float((oracle[mask].argmax(1) == yt[mask]).mean()),
        "oracle_n_swap": int(swap.sum()),
        "elapsed_s": time.time() - t0,
        "recommend": "none" if not any(r["clear_win"] for r in nested_results) else "write_candidate",
    }
    path = ROOT / "metrics_probe_twostage_v12.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path} any_clear={out['any_clear_win']} elapsed={out['elapsed_s']:.1f}s", flush=True)
    # also print conf bins briefly
    for name in ("ir", "v7", "depth"):
        print(f"conf {name}:", report["conf_bins"][name]["bins"], flush=True)


if __name__ == "__main__":
    main()
