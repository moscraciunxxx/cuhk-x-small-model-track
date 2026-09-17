"""ir_v21c: expand nested-honest gates + 4-way fuse (IR/TH/mid_ens4/mid_v4) + IR seed soft diversity.
No new trains. CPU. Merge into metrics_ir_v21_status.json.
"""
from __future__ import annotations
import json, time, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT, fuse4_sameT
from probe_ir_v18 import (
    apply_cfg, preds_full, nested_retune, load_ir_pools, build_thermal_variants,
    V7_CFG, V7_HOLD, GATE, MIN_DISAGREE,
)
from dataset import DEFAULT_HOLD_OUT_USERS

ROOT = Path(__file__).resolve().parent
LEAVE = (8, 9, 24)
PT = timezone(timedelta(hours=-7))
warnings.filterwarnings("ignore")


def blend3(pi, pt, pm, wa, wb, wc):
    return wa * pi + wb * pt + wc * pm


def nested_dual_gate(ir, th, md, y, users, mask, T=1.5):
    """If IR low-conf, mix toward th/md; if IR high and th disagrees weakly, stay IR.
    Nested tune thr_low, w_aux, wa,wb,wc baseline.
    """
    pi, pt, pm = softmax_np(ir, T), softmax_np(th, T), softmax_np(md, T)
    mx = pi.max(1)
    thrs = np.linspace(0.3, 0.9, 13)
    ws = np.linspace(0.1, 0.9, 9)
    bases = [
        (0.56, 0.35, 0.09), (0.5, 0.3, 0.2), (0.4, 0.3, 0.3), (0.45, 0.35, 0.2),
        (0.4, 0.4, 0.2), (0.35, 0.4, 0.25), (0.5, 0.35, 0.15),
    ]
    folds = []
    oof = np.full(len(y), -1, np.int64)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        best = (-1.0, None)
        for wa, wb, wc in bases:
            base_tr = blend3(pi[tr], pt[tr], pm[tr], wa, wb, wc)
            aux = 0.55 * pt[tr] + 0.45 * pm[tr]
            for thr in thrs:
                for w in ws:
                    out = base_tr.copy()
                    low = mx[tr] < thr
                    if low.any():
                        out[low] = (1 - w) * base_tr[low] + w * aux[low]
                    acc = float((out.argmax(1) == y[tr]).mean())
                    if acc > best[0]:
                        best = (acc, (wa, wb, wc, float(thr), float(w)))
        wa, wb, wc, thr, w = best[1]
        base_te = blend3(pi[te], pt[te], pm[te], wa, wb, wc)
        aux = 0.55 * pt[te] + 0.45 * pm[te]
        out = base_te.copy()
        low = mx[te] < thr
        if low.any():
            out[low] = (1 - w) * base_te[low] + w * aux[te if False else low]  # fix below
        # rewrite carefully
        out = base_te.copy()
        low = mx[te] < thr
        if low.any():
            out[low] = (1 - w) * base_te[low] + w * aux[low]
        pred = out.argmax(1)
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": float((pred == y[te]).mean()), "n": int(te.sum()),
                      "cfg": {"wa": wa, "wb": wb, "wc": wc, "thr": thr, "w": w, "T": T}})
    # full
    best = (-1.0, None)
    for wa, wb, wc in bases:
        base = blend3(pi[mask], pt[mask], pm[mask], wa, wb, wc)
        aux = 0.55 * pt[mask] + 0.45 * pm[mask]
        for thr in thrs:
            for w in ws:
                out = base.copy()
                low = mx[mask] < thr
                if low.any():
                    out[low] = (1 - w) * base[low] + w * aux[low]
                acc = float((out.argmax(1) == y[mask]).mean())
                if acc > best[0]:
                    best = (acc, (wa, wb, wc, float(thr), float(w)))
    nested = float(np.mean([f["te_acc"] for f in folds]))
    return {"nested": nested, "full": float(best[0]), "folds": folds, "oof": oof,
            "full_cfg": {"wa": best[1][0], "wb": best[1][1], "wc": best[1][2], "thr": best[1][3], "w": best[1][4], "T": T}}


def nested_fuse4(ir, th, m1, m2, y, users, mask, Ts, ngrid=11):
    folds = []
    oof = np.full(len(y), -1, np.int64)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        acc, cfg = fuse4_sameT(ir, th, m1, m2, y, tr, Ts, ngrid=ngrid)
        T = cfg["T"]
        pa, pb, pc, pd = [softmax_np(x[te], T) for x in (ir, th, m1, m2)]
        pred = (cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc + cfg["wd"] * pd).argmax(1)
        te_acc = float((pred == y[te]).mean())
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": acc, "n": int(te.sum()), "cfg": cfg})
    # full
    facc, fcfg = fuse4_sameT(ir, th, m1, m2, y, mask, Ts, ngrid=ngrid)
    nested = float(np.mean([f["te_acc"] for f in folds]))
    return {"nested": nested, "full": float(facc), "folds": folds, "oof": oof, "full_cfg": fcfg}


def nested_ir_seed_gate(members_c9, th, md, y, users, mask, T=1.5):
    """Soft-average classic9 seeds; optionally drop low-margin seeds per sample (nested thr)."""
    stack = np.stack([m["base"] if m.get("base") is not None else m["logits"] for m in members_c9], 0)  # S,N,C
    S = stack.shape[0]
    probs = np.stack([softmax_np(stack[s], T) for s in range(S)], 0)  # S,N,C
    mx = probs.max(2)  # S,N
    thrs = [0.0, 0.2, 0.3, 0.4, 0.5]
    folds = []
    oof = np.full(len(y), -1, np.int64)
    pt, pm = softmax_np(th, T), softmax_np(md, T)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        best = (-1.0, None)
        for thr in thrs:
            # weight seeds by maxprob if above thr else 0; if all zero use uniform
            w = np.where(mx[:, tr] >= thr, mx[:, tr], 0.0)  # S,ntr
            wsum = w.sum(0, keepdims=True) + 1e-12
            w = w / wsum
            ir_tr = (w[:, :, None] * probs[:, tr, :]).sum(0)
            for wa, wb, wc in [(0.5, 0.3, 0.2), (0.4, 0.3, 0.3), (0.45, 0.35, 0.2), (0.56, 0.35, 0.09)]:
                pb = wa * ir_tr + wb * pt[tr] + wc * pm[tr]
                acc = float((pb.argmax(1) == y[tr]).mean())
                if acc > best[0]:
                    best = (acc, (thr, wa, wb, wc))
        thr, wa, wb, wc = best[1]
        w = np.where(mx[:, te] >= thr, mx[:, te], 0.0)
        wsum = w.sum(0, keepdims=True) + 1e-12
        w = w / wsum
        ir_te = (w[:, :, None] * probs[:, te, :]).sum(0)
        pred = (wa * ir_te + wb * pt[te] + wc * pm[te]).argmax(1)
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": float((pred == y[te]).mean()), "n": int(te.sum()),
                      "cfg": {"thr": thr, "wa": wa, "wb": wb, "wc": wc, "T": T}})
    nested = float(np.mean([f["te_acc"] for f in folds]))
    # full
    best = (-1.0, None)
    for thr in thrs:
        w = np.where(mx[:, mask] >= thr, mx[:, mask], 0.0)
        w = w / (w.sum(0, keepdims=True) + 1e-12)
        ir_m = (w[:, :, None] * probs[:, mask, :]).sum(0)
        for wa, wb, wc in [(0.5, 0.3, 0.2), (0.4, 0.3, 0.3), (0.45, 0.35, 0.2), (0.56, 0.35, 0.09)]:
            pb = wa * ir_m + wb * pt[mask] + wc * pm[mask]
            acc = float((pb.argmax(1) == y[mask]).mean())
            if acc > best[0]:
                best = (acc, (thr, wa, wb, wc))
    return {"nested": nested, "full": float(best[0]), "folds": folds, "oof": oof,
            "full_cfg": {"thr": best[1][0], "wa": best[1][1], "wb": best[1][2], "wc": best[1][3], "T": T}}


def main():
    t0 = time.time()
    print("ir_v21c expand gates / fuse4 / ir-seed", flush=True)
    members, yt, yu = load_members()
    ir_meta_all = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "train_meta.json", encoding="utf-8"))
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    ir_meta = [ir_meta_all[i] for i in hold_idx]
    mid_alts = {}
    for name, path in [
        ("mid_ir_v4", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy"),
        ("mid_ens3", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3.npy"),
        ("mid_ens4_bonetcn", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy"),
    ]:
        m = np.load(path)
        mid_alts[name] = (m[hold_idx] if len(m) == len(tu) else m).astype(np.float32)
    pools, c9 = load_ir_pools(members, yt, yu)
    th_vars = build_thermal_variants(yt, ir_meta)
    ir0 = pools["classic9_base"]
    mask0 = th_vars["th_v6_v2trio"].any(1) & mid_alts["mid_ir_v4"].any(1)
    v7_preds = preds_full(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], mask0, V7_CFG)

    rows = []
    # dual gate on best combos
    for ir_name, th_name, mid_name in [
        ("classic9_base", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_base", "th_v6_soft", "mid_ens4_bonetcn"),
        ("classic_acc_w_base", "th_v6_v2trio", "mid_ens4_bonetcn"),
    ]:
        ir, th, md = pools[ir_name], th_vars[th_name], mid_alts[mid_name]
        mask = th.any(1) & md.any(1)
        for T in [1.25, 1.5, 2.0, 2.5]:
            out = nested_dual_gate(ir, th, md, yt, yu, mask, T=T)
            both = mask & (out["oof"] >= 0) & (v7_preds >= 0)
            dis = int((out["oof"][both] != v7_preds[both]).sum())
            rows.append({"kind": "dual_gate", "T": T, "ir": ir_name, "th": th_name, "mid": mid_name,
                         "honest_nested": out["nested"], "full": out["full"], "disagree": dis,
                         "folds": out["folds"], "full_cfg": out["full_cfg"]})
            print(f"dual_gate {ir_name}|{th_name}|{mid_name} T={T} nested={out['nested']:.4f} full={out['full']:.4f} dis={dis}", flush=True)

    # fuse4
    ir = pools["classic9_base"]
    th = th_vars["th_v6_v2trio"]
    m1 = mid_alts["mid_ens4_bonetcn"]
    m2 = mid_alts["mid_ir_v4"]
    mask = th.any(1) & m1.any(1) & m2.any(1)
    Ts = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    out4 = nested_fuse4(ir, th, m1, m2, yt, yu, mask, Ts, ngrid=11)
    both = mask & (out4["oof"] >= 0) & (v7_preds >= 0)
    dis = int((out4["oof"][both] != v7_preds[both]).sum())
    rows.append({"kind": "fuse4", "ir": "classic9_base", "th": "th_v6_v2trio", "mid": "ens4+v4",
                 "honest_nested": out4["nested"], "full": out4["full"], "disagree": dis,
                 "folds": out4["folds"], "full_cfg": out4["full_cfg"]})
    print(f"fuse4 nested={out4['nested']:.4f} full={out4['full']:.4f} dis={dis} cfg={out4['full_cfg']}", flush=True)

    # also fuse4 with soft th
    th2 = th_vars["th_v6_soft"]
    out4b = nested_fuse4(ir, th2, m1, m2, yt, yu, mask, Ts, ngrid=11)
    both = mask & (out4b["oof"] >= 0) & (v7_preds >= 0)
    dis = int((out4b["oof"][both] != v7_preds[both]).sum())
    rows.append({"kind": "fuse4", "ir": "classic9_base", "th": "th_v6_soft", "mid": "ens4+v4",
                 "honest_nested": out4b["nested"], "full": out4b["full"], "disagree": dis,
                 "folds": out4b["folds"], "full_cfg": out4b["full_cfg"]})
    print(f"fuse4 soft nested={out4b['nested']:.4f} full={out4b['full']:.4f} dis={dis}", flush=True)

    # IR seed gate
    for T in [1.5, 2.0, 2.5]:
        out = nested_ir_seed_gate(c9, th_vars["th_v6_v2trio"], mid_alts["mid_ens4_bonetcn"], yt, yu, mask, T=T)
        both = mask & (out["oof"] >= 0) & (v7_preds >= 0)
        dis = int((out["oof"][both] != v7_preds[both]).sum())
        rows.append({"kind": "ir_seed_gate", "T": T, "ir": "classic9_seeds", "th": "th_v6_v2trio", "mid": "mid_ens4_bonetcn",
                     "honest_nested": out["nested"], "full": out["full"], "disagree": dis,
                     "folds": out["folds"], "full_cfg": out["full_cfg"]})
        print(f"ir_seed_gate T={T} nested={out['nested']:.4f} full={out['full']:.4f} dis={dis}", flush=True)

    rows.sort(key=lambda r: (r["honest_nested"], r.get("disagree", 0)), reverse=True)
    best = rows[0]

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if k != "oof"}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    prev = json.loads((ROOT / "metrics_ir_v21_status.json").read_text(encoding="utf-8"))
    prev_best_n = float(prev.get("overall_best_honest", {}).get("honest_nested") or 0)
    overall_n = max(prev_best_n, best["honest_nested"])
    if best["honest_nested"] >= prev_best_n:
        overall = {"source": "v21c_" + best["kind"], "honest_nested": best["honest_nested"], "detail": scrub(best)}
    else:
        overall = prev.get("overall_best_honest")

    clears = best["honest_nested"] >= GATE and best["full"] >= GATE and best.get("disagree", 0) >= MIN_DISAGREE
    prev["v21c_expand"] = {"best": scrub(best), "top10": scrub(rows[:10]), "elapsed_sec": round(time.time() - t0, 1)}
    prev["overall_best_honest"] = overall
    prev["outcome"] = "WIN" if (overall["honest_nested"] >= GATE and clears) else "MISS"
    prev["keep_ir_v7"] = prev["outcome"] != "WIN"
    prev["finished_at"] = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    prev["gpu_status_at_finish"] = "idle_unused_cpu_only"
    prev["delta_vs_gate"] = {
        "best_nested": float(overall["honest_nested"] - GATE),
        "v21c_best_nested": float(best["honest_nested"] - GATE),
    }
    prev["next_roi"] = [
        f"MISS gate: overall honest nested={overall['honest_nested']:.4f} ({overall['source']}) vs >={GATE:.4f}; keep ir_v7 @ 0.69154",
        f"v21c best={best['kind']} nested={best['honest_nested']:.4f} full={best['full']:.4f} dis={best.get('disagree')}",
        "Class-level LOUO stack failed (overfit). Gated/weight stack plateaus ~0.748 — Δ≈-0.015 to gate",
        "mmWave stub unusable. IMU already in midfuse dual; new IMU-spectrogram alone unlikely unless 4th complementary stream",
        "Next ROI: IR temporal-stride / multi-clip TTA on EXISTING classic9 ckpts (no new Kinetics backbone); or accept ceiling and shift to Large track",
        "GPU unused — handoff clear for LMT",
    ]
    prev["summary"] = {
        **(prev.get("summary") or {}),
        "v21c_best_nested": best["honest_nested"],
        "v21c_best_kind": best["kind"],
        "overall_honest_nested": overall["honest_nested"],
        "overall_source": overall["source"],
        "gate": GATE,
        "public_keep": "submission_ir_v7.csv @ 0.69154",
    }
    (ROOT / "metrics_ir_v21_status.json").write_text(json.dumps(scrub(prev), indent=2), encoding="utf-8")
    print(f"DONE outcome={prev['outcome']} best={best['kind']}:{best['honest_nested']:.4f} overall={overall['honest_nested']:.4f}", flush=True)


if __name__ == "__main__":
    main()
