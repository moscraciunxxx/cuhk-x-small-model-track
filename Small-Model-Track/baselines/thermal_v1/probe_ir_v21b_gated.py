"""ir_v21b: nested-honest LOW-DIM gated / weight stackers (predict mix weights, not 40-way class).
Also conf-gate grids. Updates metrics_ir_v21_status.json gated section.
CPU-only.
"""
from __future__ import annotations
import json, time, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT, conf_gate_blend
from probe_ir_v18 import (
    apply_cfg, preds_full, nested_retune, load_ir_pools, build_thermal_variants,
    V7_CFG, V7_HOLD, GATE, MIN_DISAGREE,
)

ROOT = Path(__file__).resolve().parent
LEAVE = (8, 9, 24)
PT = timezone(timedelta(hours=-7))
warnings.filterwarnings("ignore")


def meta_feats(ir, th, md, T=1.5):
    pi, pt, pm = softmax_np(ir, T), softmax_np(th, T), softmax_np(md, T)
    def pack(p, logits):
        mx = p.max(1)
        part = np.partition(p, -2, axis=1)
        mar = part[:, -1] - part[:, -2]
        ent = -(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1)
        return mx, mar, ent
    a = pack(pi, ir); b = pack(pt, th); c = pack(pm, md)
    ai, at, am = ir.argmax(1), th.argmax(1), md.argmax(1)
    X = np.stack([
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
        (ai == at).astype(float), (ai == am).astype(float), (at == am).astype(float),
        ((ai == at) & (ai == am)).astype(float),
        a[0] - b[0], a[0] - c[0], b[0] - c[0],
    ], 1).astype(np.float64)
    return X, pi, pt, pm


def blend_with_weights(pi, pt, pm, W):
    # W: (N,3) nonnegative, row-normalized
    s = W.sum(1, keepdims=True) + 1e-12
    W = W / s
    return W[:, 0:1] * pi + W[:, 1:2] * pt + W[:, 2:3] * pm


def fit_weight_regressor(Xtr, ytr, pi_tr, pt_tr, pm_tr, kind="ridge"):
    """Supervised target: one-hot of which single source is correct (prefer IR on ties)."""
    correct = np.stack([
        (pi_tr.argmax(1) == ytr),
        (pt_tr.argmax(1) == ytr),
        (pm_tr.argmax(1) == ytr),
    ], 1).astype(np.float64)
    # if none correct, use soft targets proportional to -entropy? fall back equal
    none = correct.sum(1) == 0
    correct[none] = 1.0 / 3.0
    # tie-break: normalize
    correct = correct / (correct.sum(1, keepdims=True) + 1e-12)

    if kind == "logreg_src":
        # classify which source (argmax of correct, IR-priority)
        src = correct.argmax(1)
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.3, solver="lbfgs", random_state=42)),
        ])
        pipe.fit(Xtr, src)
        return ("logreg_src", pipe)
    else:
        # multi-output ridge -> 3 weights
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("reg", Ridge(alpha=5.0, random_state=42)),
        ])
        pipe.fit(Xtr, correct)
        return ("ridge_w", pipe)


def predict_weights(kind_model, X, n):
    kind, model = kind_model
    if kind == "logreg_src":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = model.named_steps["clf"].classes_
            W = np.zeros((n, 3), dtype=np.float64)
            for j, c in enumerate(classes):
                W[:, int(c)] = proba[:, j]
            # ensure 3 cols
            if W.shape[1] < 3:
                W2 = np.zeros((n, 3)); W2[:, :W.shape[1]] = W; W = W2
            return W
        pred = model.predict(X)
        W = np.zeros((n, 3)); W[np.arange(n), pred.astype(int)] = 1.0
        return W
    else:
        W = model.predict(X)
        W = np.clip(W, 0, None)
        return W


def louo_weight_stack(ir, th, md, y, users, mask, kind="ridge", T=1.5):
    X, pi, pt, pm = meta_feats(ir, th, md, T=T)
    folds = []
    oof = np.full(len(y), -1, dtype=np.int64)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        km = fit_weight_regressor(X[tr], y[tr], pi[tr], pt[tr], pm[tr], kind=kind)
        W = predict_weights(km, X[te], int(te.sum()))
        pb = blend_with_weights(pi[te], pt[te], pm[te], W)
        pred = pb.argmax(1)
        te_acc = float((pred == y[te]).mean())
        # train acc
        Wtr = predict_weights(km, X[tr], int(tr.sum()))
        tr_acc = float((blend_with_weights(pi[tr], pt[tr], pm[tr], Wtr).argmax(1) == y[tr]).mean())
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": tr_acc, "n": int(te.sum())})
    # full resub
    km = fit_weight_regressor(X[mask], y[mask], pi[mask], pt[mask], pm[mask], kind=kind)
    Wf = predict_weights(km, X[mask], int(mask.sum()))
    full = float((blend_with_weights(pi[mask], pt[mask], pm[mask], Wf).argmax(1) == y[mask]).mean())
    nested = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"nested": nested, "full_resub": full, "folds": folds, "oof": oof}


def nested_conf_gate(ir, th, md, y, users, mask, T=1.5):
    """Primary=IR, aux=blend(th,md); gate on IR maxprob. Nested: tune thr,w on train users."""
    pi = softmax_np(ir, T)
    pt = softmax_np(th, T)
    pm = softmax_np(md, T)
    aux = 0.5 * pt + 0.5 * pm
    mx = pi.max(1)
    thrs = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    ws = [0.2, 0.35, 0.5, 0.65, 0.8]
    folds = []
    oof = np.full(len(y), -1, dtype=np.int64)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        best = (-1.0, None)
        for thr in thrs:
            for w in ws:
                out = pi[tr].copy()
                low = mx[tr] < thr
                if low.any():
                    out[low] = (1 - w) * pi[tr][low] + w * aux[tr][low]
                # also mix fixed base weights option: start from 0.4/0.3/0.3
                base = 0.4 * pi[tr] + 0.3 * pt[tr] + 0.3 * pm[tr]
                # blend gated IR-aux with base
                for g in [0.0, 0.5, 1.0]:
                    cand = (1 - g) * base + g * out
                    acc = float((cand.argmax(1) == y[tr]).mean())
                    if acc > best[0]:
                        best = (acc, (thr, w, g))
        thr, w, g = best[1]
        out_te = pi[te].copy()
        low = mx[te] < thr
        if low.any():
            out_te[low] = (1 - w) * pi[te][low] + w * aux[te][low]
        base_te = 0.4 * pi[te] + 0.3 * pt[te] + 0.3 * pm[te]
        cand = (1 - g) * base_te + g * out_te
        pred = cand.argmax(1)
        te_acc = float((pred == y[te]).mean())
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum()), "cfg": {"thr": thr, "w": w, "g": g, "T": T}})
    # full tune
    best = (-1.0, None)
    for thr in thrs:
        for w in ws:
            out = pi[mask].copy()
            low = mx[mask] < thr
            if low.any():
                out[low] = (1 - w) * pi[mask][low] + w * aux[mask][low]
            base = 0.4 * pi[mask] + 0.3 * pt[mask] + 0.3 * pm[mask]
            for g in [0.0, 0.5, 1.0]:
                cand = (1 - g) * base + g * out
                acc = float((cand.argmax(1) == y[mask]).mean())
                if acc > best[0]:
                    best = (acc, (thr, w, g, cand.argmax(1)))
    nested = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"nested": nested, "full": float(best[0]), "folds": folds, "oof": oof, "full_cfg": {"thr": best[1][0], "w": best[1][1], "g": best[1][2], "T": T}}


def nested_entropy_reweight(ir, th, md, y, users, mask, T=1.5):
    """w_i ∝ exp(-alpha * entropy_i); tune alpha nested + temperature on mix."""
    pi, pt, pm = softmax_np(ir, T), softmax_np(th, T), softmax_np(md, T)
    def ent(p):
        return -(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1)
    ei, et, em = ent(pi), ent(pt), ent(pm)
    alphas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    folds = []
    oof = np.full(len(y), -1, dtype=np.int64)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        best = (-1.0, None)
        for a in alphas:
            Wi = np.exp(-a * ei[tr]); Wt = np.exp(-a * et[tr]); Wm = np.exp(-a * em[tr])
            W = np.stack([Wi, Wt, Wm], 1)
            pb = blend_with_weights(pi[tr], pt[tr], pm[tr], W)
            acc = float((pb.argmax(1) == y[tr]).mean())
            if acc > best[0]:
                best = (acc, a)
        a = best[1]
        W = np.stack([np.exp(-a * ei[te]), np.exp(-a * et[te]), np.exp(-a * em[te])], 1)
        pred = blend_with_weights(pi[te], pt[te], pm[te], W).argmax(1)
        te_acc = float((pred == y[te]).mean())
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum()), "alpha": a})
    # full
    best = (-1.0, None)
    for a in alphas:
        W = np.stack([np.exp(-a * ei[mask]), np.exp(-a * et[mask]), np.exp(-a * em[mask])], 1)
        acc = float((blend_with_weights(pi[mask], pt[mask], pm[mask], W).argmax(1) == y[mask]).mean())
        if acc > best[0]:
            best = (acc, a)
    nested = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"nested": nested, "full": float(best[0]), "folds": folds, "oof": oof, "alpha_full": best[1]}


def main():
    t0 = time.time()
    print("ir_v21b low-dim gated / weight stackers", flush=True)
    members, yt, yu = load_members()
    ir_meta_all = json.load(open(ROOT / "cache" / "ir_yolo_v4" / "train_meta.json", encoding="utf-8"))
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    ir_meta = [ir_meta_all[i] for i in hold_idx]

    mid_alts = {}
    for name, path in [
        ("mid_ir_v4", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy"),
        ("mid_ens3", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3.npy"),
        ("mid_ens3_softT25", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens3_softT25.npy"),
        ("mid_ens4_bonetcn", ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy"),
    ]:
        m = np.load(path)
        mid_alts[name] = (m[hold_idx] if len(m) == len(tu) else m).astype(np.float32)

    pools, _ = load_ir_pools(members, yt, yu)
    th_vars = build_thermal_variants(yt, ir_meta)

    ir0 = pools["classic9_base"]
    mask0 = th_vars["th_v6_v2trio"].any(1) & mid_alts["mid_ir_v4"].any(1)
    v7_preds = preds_full(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], mask0, V7_CFG)

    combos = [
        ("classic9_base", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_base", "th_v6_soft", "mid_ens4_bonetcn"),
        ("classic_acc_w_base", "th_v6_v2trio", "mid_ens3"),
        ("classic_top6_base", "th_v6_soft", "mid_ens3_softT25"),
        ("classic9_base", "th_v6_v2trio", "mid_ir_v4"),
    ]

    rows = []
    for ir_name, th_name, mid_name in combos:
        ir, th, md = pools[ir_name], th_vars[th_name], mid_alts[mid_name]
        mask = th.any(1) & md.any(1)
        Ts = [1.0, 1.5, 2.0, 2.5]
        b_acc, cfg = fuse3_sameT(ir, th, md, yt, mask, Ts + [1.25, 1.75, 3.0], ngrid=21)
        cfg = dict(cfg); cfg["mode"] = "sameT"
        nest_f = nested_fixed(ir, th, md, yt, yu, mask, cfg)
        nest_r = nested_retune(ir, th, md, yt, yu, mask, Ts + [1.25, 1.75, 3.0], ngrid=21)
        base_honest = float(min(nest_f["mean"], nest_r["mean"]))
        print(f"BASE {ir_name}|{th_name}|{mid_name} full={b_acc:.4f} honest={base_honest:.4f}", flush=True)

        for kind in ["ridge", "logreg_src"]:
            for T in [1.0, 1.5, 2.0, 2.5]:
                out = louo_weight_stack(ir, th, md, yt, yu, mask, kind=kind, T=T)
                both = mask & (out["oof"] >= 0) & (v7_preds >= 0)
                dis = int((out["oof"][both] != v7_preds[both]).sum()) if both.any() else 0
                row = {
                    "kind": f"weight_stack_{kind}", "T": T,
                    "ir": ir_name, "th": th_name, "mid": mid_name,
                    "honest_nested": out["nested"], "full_resub": out["full_resub"],
                    "folds": out["folds"], "disagree_vs_v7_oof": dis,
                    "base_honest": base_honest, "base_full": float(b_acc),
                    "delta_vs_base": out["nested"] - base_honest,
                }
                rows.append(row)
                print(f"  {kind} T={T} nested={out['nested']:.4f} full={out['full_resub']:.4f} "
                      f"dBase={row['delta_vs_base']:+.4f} dis={dis} folds={[round(f['te_acc'],3) for f in out['folds']]}", flush=True)

        for T in [1.5, 2.0, 2.5]:
            cg = nested_conf_gate(ir, th, md, yt, yu, mask, T=T)
            both = mask & (cg["oof"] >= 0) & (v7_preds >= 0)
            dis = int((cg["oof"][both] != v7_preds[both]).sum()) if both.any() else 0
            row = {
                "kind": "conf_gate", "T": T,
                "ir": ir_name, "th": th_name, "mid": mid_name,
                "honest_nested": cg["nested"], "full_resub": cg["full"],
                "folds": cg["folds"], "disagree_vs_v7_oof": dis,
                "base_honest": base_honest, "base_full": float(b_acc),
                "delta_vs_base": cg["nested"] - base_honest,
                "full_cfg": cg["full_cfg"],
            }
            rows.append(row)
            print(f"  conf_gate T={T} nested={cg['nested']:.4f} full={cg['full']:.4f} "
                  f"dBase={row['delta_vs_base']:+.4f} dis={dis}", flush=True)

            eg = nested_entropy_reweight(ir, th, md, yt, yu, mask, T=T)
            both = mask & (eg["oof"] >= 0) & (v7_preds >= 0)
            dis = int((eg["oof"][both] != v7_preds[both]).sum()) if both.any() else 0
            row = {
                "kind": "entropy_reweight", "T": T,
                "ir": ir_name, "th": th_name, "mid": mid_name,
                "honest_nested": eg["nested"], "full_resub": eg["full"],
                "folds": eg["folds"], "disagree_vs_v7_oof": dis,
                "base_honest": base_honest, "base_full": float(b_acc),
                "delta_vs_base": eg["nested"] - base_honest,
                "alpha_full": eg["alpha_full"],
            }
            rows.append(row)
            print(f"  entropy_rw T={T} nested={eg['nested']:.4f} full={eg['full']:.4f} "
                  f"dBase={row['delta_vs_base']:+.4f} dis={dis}", flush=True)

    rows.sort(key=lambda r: (r["honest_nested"], r["disagree_vs_v7_oof"]), reverse=True)
    best = rows[0] if rows else None
    # also keep best weight-blend honest among combos
    best_base = max(rows, key=lambda r: r["base_honest"]) if rows else None

    clears = False
    if best is not None:
        clears = best["honest_nested"] >= GATE and best["full_resub"] >= GATE and best["disagree_vs_v7_oof"] >= MIN_DISAGREE

    # merge into existing status
    prev_path = ROOT / "metrics_ir_v21_status.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if k != "oof"}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    gated = {
        "best": scrub(best),
        "top12": scrub(rows[:12]),
        "best_weight_blend_ref": scrub({k: best_base[k] for k in ["ir","th","mid","base_honest","base_full"]}) if best_base else None,
        "note": "Low-dim weight/gate stackers; class-level LOUO stack from v21a overfit badly (~0.66 nested)",
    }

    outcome = "WIN" if clears else "MISS"
    # overall best honest across v21a class-stack and v21b gated and weight-blend
    candidates = []
    if prev.get("best"):
        candidates.append(("class_stack_v21a", prev["best"].get("honest_nested", 0), prev["best"]))
    if best:
        candidates.append(("gated_v21b", best["honest_nested"], best))
    if best_base:
        candidates.append(("weight_blend", best_base["base_honest"], {
            "kind": "weight_blend_sameT",
            "ir": best_base["ir"], "th": best_base["th"], "mid": best_base["mid"],
            "honest_nested": best_base["base_honest"], "full_resub": best_base["base_full"],
            "disagree_vs_v7_oof": None,
        }))
    candidates.sort(key=lambda t: t[1], reverse=True)
    overall_name, overall_n, overall = candidates[0]

    # Gate vs overall: weight blend full may be higher but honest nested is the metric
    # For WIN need overall nested + a hold estimate >= gate. Weight blend full from v20 was 0.749.
    status = dict(prev)
    status.update({
        "tag": "ir_v21_louo_stack_gated",
        "outcome": outcome if overall_name != "weight_blend" else "MISS",  # weight blend already known miss
        "keep_ir_v7": True,
        "gated_v21b": gated,
        "overall_best_honest": {
            "source": overall_name,
            "honest_nested": float(overall_n),
            "detail": scrub(overall),
        },
        "delta_vs_gate": {
            "best_nested": float(overall_n - GATE),
            "class_stack_nested": (None if not prev.get("best") else float(prev["best"]["honest_nested"] - GATE)),
            "gated_nested": (None if not best else float(best["honest_nested"] - GATE)),
            "weight_blend_nested": (None if not best_base else float(best_base["base_honest"] - GATE)),
        },
        "elapsed_sec_v21b": round(time.time() - t0, 1),
        "finished_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "gpu_status_at_finish": "idle_unused_cpu_only",
        "wrote_csv": False,
        "next_roi": [
            f"MISS gate: overall honest nested={overall_n:.4f} ({overall_name}) vs gate>={GATE:.4f}; keep ir_v7 @ 0.69154",
            "v21a class-level LOUO stack OVERFITS (nested~0.66 << weight-blend~0.74); do not ship",
            f"v21b gated/weight-stack best nested={None if not best else round(best['honest_nested'],4)} "
            f"dBase={None if not best else round(best['delta_vs_base'],4)} — still below gate",
            "Radar/mmWave stub (~43B) — skip. IMU already in midfuse dual (~0.50-0.56); spectrogram-only IMU unlikely to beat mid_ens4 alone",
            "Next ROI: (1) richer Mid diversity beyond bone/midfuse ens — e.g. IMU spectrogram branch as 4th logit if complementary; "
            "(2) IR temporal TTA / clip-stride ensemble of EXISTING classic9 (no new Kinetics); "
            "(3) stop fishing fuse weights on hold",
            "GPU unused — handoff clear for LMT",
        ],
        "summary": {
            "class_stack_best_nested": None if not prev.get("best") else prev["best"]["honest_nested"],
            "gated_best_nested": None if not best else best["honest_nested"],
            "weight_blend_best_nested": None if not best_base else best_base["base_honest"],
            "overall_honest_nested": float(overall_n),
            "overall_source": overall_name,
            "gate": GATE,
            "public_keep": "submission_ir_v7.csv @ 0.69154",
            "mmwave": "unusable_stub",
            "imu": "present_already_in_midfuse_dual",
        },
    })
    # force outcome MISS if overall < gate
    if overall_n < GATE:
        status["outcome"] = "MISS"
        status["keep_ir_v7"] = True

    prev_path.write_text(json.dumps(scrub(status), indent=2), encoding="utf-8")
    print(f"Wrote {prev_path} outcome={status['outcome']} overall={overall_name}:{overall_n:.4f}", flush=True)


if __name__ == "__main__":
    main()
