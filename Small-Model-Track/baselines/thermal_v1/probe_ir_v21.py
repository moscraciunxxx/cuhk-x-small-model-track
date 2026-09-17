"""ir_v21: nested-honest LOUO calibrated stacker on existing IR+Thermal+Mid logits.
No new IR Kinetics / Thermal R2+1D seeds. CPU-only.
Gate: hold+nested >= ~0.763 AND >=20 disagrees vs ir_v7.
"""
from __future__ import annotations
import json, time, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from probe_ir_v18 import (
    apply_cfg, preds_full, nested_retune, load_ir_pools, build_thermal_variants,
    honest_of, V7_CFG, V7_HOLD, GATE, MIN_DISAGREE,
)

ROOT = Path(__file__).resolve().parent
LEAVE_USERS = (8, 9, 24)
PT = timezone(timedelta(hours=-7))
warnings.filterwarnings("ignore", category=UserWarning)


def entropy_np(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin_np(p):
    # top1 - top2
    part = np.partition(p, -2, axis=1)
    return part[:, -1] - part[:, -2]


def feat_from_logits(logits, T=1.0, prefix=""):
    """probs (C) + maxprob + margin + entropy (+ optional logit max/std)."""
    p = softmax_np(logits, T).astype(np.float64)
    mx = p.max(1, keepdims=True)
    mar = margin_np(p)[:, None]
    ent = entropy_np(p)[:, None]
    # also temperature-free logit stats
    lg = logits.astype(np.float64)
    lmx = lg.max(1, keepdims=True)
    lstd = lg.std(1, keepdims=True)
    X = np.concatenate([p, mx, mar, ent, lmx, lstd], axis=1)
    names = (
        [f"{prefix}p{i}" for i in range(p.shape[1])]
        + [f"{prefix}max", f"{prefix}margin", f"{prefix}ent", f"{prefix}lmax", f"{prefix}lstd"]
    )
    return X, names


def build_stack_features(ir, th, md, T_list=(1.0, 1.5, 2.0, 2.5)):
    """Multi-T probs/meta + pairwise disagree + conf diffs."""
    blocks = []
    names = []
    for T in T_list:
        Xi, ni = feat_from_logits(ir, T, prefix=f"irT{T}_")
        Xt, nt = feat_from_logits(th, T, prefix=f"thT{T}_")
        Xm, nm = feat_from_logits(md, T, prefix=f"mdT{T}_")
        blocks += [Xi, Xt, Xm]
        names += ni + nt + nm
        # blend probs at equal weights as extra channels
        pi = softmax_np(ir, T)
        pt = softmax_np(th, T)
        pm = softmax_np(md, T)
        for tag, w in [("eq", (1 / 3,) * 3), ("irh", (0.5, 0.3, 0.2)), ("thh", (0.35, 0.45, 0.2))]:
            pb = w[0] * pi + w[1] * pt + w[2] * pm
            blocks.append(pb)
            names += [f"blend_{tag}_T{T}_c{i}" for i in range(pb.shape[1])]
            blocks.append(pb.max(1, keepdims=True))
            names.append(f"blend_{tag}_T{T}_max")
    # disagreements / agreement
    pi = ir.argmax(1); pt = th.argmax(1); pm = md.argmax(1)
    blocks.append(np.stack([
        (pi == pt).astype(np.float64),
        (pi == pm).astype(np.float64),
        (pt == pm).astype(np.float64),
        ((pi == pt) & (pi == pm)).astype(np.float64),
    ], 1))
    names += ["agree_ir_th", "agree_ir_md", "agree_th_md", "agree_all"]
    X = np.concatenate(blocks, axis=1).astype(np.float64)
    return X, names


def build_stack_features_compact(ir, th, md, T=1.5):
    """Lower-dim: probs@T + meta + agrees (~3*45 + 4)."""
    Xi, _ = feat_from_logits(ir, T, "ir_")
    Xt, _ = feat_from_logits(th, T, "th_")
    Xm, _ = feat_from_logits(md, T, "md_")
    pi, pt, pm = ir.argmax(1), th.argmax(1), md.argmax(1)
    agree = np.stack([
        (pi == pt).astype(np.float64),
        (pi == pm).astype(np.float64),
        (pt == pm).astype(np.float64),
        ((pi == pt) & (pi == pm)).astype(np.float64),
    ], 1)
    # classic weighted blend probs
    pb = 0.4 * softmax_np(ir, T) + 0.3 * softmax_np(th, T) + 0.3 * softmax_np(md, T)
    X = np.concatenate([Xi, Xt, Xm, pb, agree], axis=1).astype(np.float64)
    return X


def make_models():
    return {
        "logreg_l2": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, solver="lbfgs",
                C=0.2, random_state=42,
            )),
        ]),
        "logreg_l2_C1": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, solver="lbfgs",
                C=1.0, random_state=42,
            )),
        ]),
        "logreg_l2_C05": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, solver="lbfgs",
                C=0.05, random_state=42,
            )),
        ]),
        "ridge": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RidgeClassifier(alpha=10.0, random_state=42)),
        ]),
        "ridge_a2": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RidgeClassifier(alpha=2.0, random_state=42)),
        ]),
        "mlp_small": Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64,), activation="relu",
                alpha=1e-2, max_iter=400, random_state=42, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20,
            )),
        ]),
        "mlp_tiny": Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu",
                alpha=5e-2, max_iter=400, random_state=42, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=20,
            )),
        ]),
    }


def predict_proba_safe(model, X, n_classes=40):
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # ensure all classes
            classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
            out = np.zeros((len(X), n_classes), dtype=np.float64)
            for j, c in enumerate(classes):
                out[:, int(c)] = proba[:, j]
            return out
        except Exception:
            pass
    # decision_function / predict fallback
    if hasattr(model[-1] if hasattr(model, "__getitem__") else model, "decision_function") or hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(X)
            if dec.ndim == 1:
                # binary unlikely
                out = np.zeros((len(X), n_classes))
                classes = model.named_steps["clf"].classes_
                # one-vs-rest style
                return softmax_np(dec if dec.ndim == 2 else np.column_stack([-dec, dec]), 1.0)
            classes = model.named_steps["clf"].classes_
            out = np.full((len(X), n_classes), -1e9, dtype=np.float64)
            for j, c in enumerate(classes):
                out[:, int(c)] = dec[:, j]
            return softmax_np(out, 1.0)
        except Exception:
            pass
    pred = model.predict(X)
    out = np.zeros((len(X), n_classes), dtype=np.float64)
    out[np.arange(len(X)), pred.astype(int)] = 1.0
    return out


def louo_stack(X, y, users, mask, model_factory, n_classes=40):
    """Nested-honest LOUO: train on hold users != leave, eval on leave."""
    folds = []
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    oof_proba = np.zeros((len(y), n_classes), dtype=np.float64)
    for leave in LEAVE_USERS:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        model = model_factory()
        model.fit(X[tr], y[tr])
        proba = predict_proba_safe(model, X[te], n_classes)
        pred = proba.argmax(1)
        te_acc = float((pred == y[te]).mean())
        tr_acc = float((model.predict(X[tr]) == y[tr]).mean())
        oof_pred[te] = pred
        oof_proba[te] = proba
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": tr_acc, "n": int(te.sum())})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    # full (optimistic): fit on all masked, score same
    model_full = model_factory()
    model_full.fit(X[mask], y[mask])
    full_pred = model_full.predict(X[mask])
    full_acc = float((full_pred == y[mask]).mean())
    return {
        "nested": mean,
        "folds": folds,
        "full_resub": full_acc,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "model_full": model_full,
    }


def weight_blend_baseline(ir, th, md, y, users, mask):
    Ts = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    b_acc, cfg = fuse3_sameT(ir, th, md, y, mask, Ts, ngrid=21)
    cfg = dict(cfg); cfg["mode"] = "sameT"
    nest_f = nested_fixed(ir, th, md, y, users, mask, cfg)
    nest_r = nested_retune(ir, th, md, y, users, mask, Ts, ngrid=21)
    row = {
        "full": float(b_acc),
        "cfg": {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in cfg.items()},
        "nested_fixed": float(nest_f["mean"]),
        "nested_retune": float(nest_r["mean"]),
        "honest_nested": float(min(nest_f["mean"], nest_r["mean"])),
    }
    return row, preds_full(ir, th, md, mask, cfg)


def inventory_modalities():
    train = Path(r"D:\CUHK-X\Small-Model-Track\Training\data\HAR\data")
    mods = {}
    for m in sorted(p.name for p in train.iterdir() if p.is_dir()):
        # sample one clip sizes
        act = next(train.joinpath(m).iterdir())
        user = next(act.iterdir())
        files = [p for p in user.rglob("*") if p.is_file()]
        sizes = [p.stat().st_size for p in files]
        mods[m] = {
            "n_actions": len(list(train.joinpath(m).iterdir())),
            "sample_user": user.name,
            "n_files_sample": len(files),
            "mean_file_bytes": float(np.mean(sizes)) if sizes else 0.0,
            "max_file_bytes": int(max(sizes)) if sizes else 0,
            "usable_hint": "stub_or_empty" if (sizes and max(sizes) < 200) else ("csv_timeseries" if sizes else "unknown"),
        }
    return mods


def main():
    t0 = time.time()
    print("ir_v21 LOUO calibrated stacker (CPU)", flush=True)
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

    pools, c9 = load_ir_pools(members, yt, yu)
    th_vars = build_thermal_variants(yt, ir_meta)

    # focus combos from v20 best + a couple alts
    combos = [
        ("classic9_base", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_base", "th_v6_soft", "mid_ens4_bonetcn"),
        ("classic9_base", "th_v2_ens", "mid_ens4_bonetcn"),
        ("classic_acc_w_base", "th_v6_v2trio", "mid_ens3"),
        ("classic_top6_base", "th_v6_soft", "mid_ens3_softT25"),
        ("classic9_base", "th_v6_v2trio", "mid_ir_v4"),
    ]
    # add th_v5 mixes if present
    for k in list(th_vars.keys()):
        if k.startswith("mix_v6_") or k.startswith("th_v5_ens"):
            combos.append(("classic9_base", k, "mid_ens4_bonetcn"))

    ir0 = pools["classic9_base"]
    mask0 = th_vars["th_v6_v2trio"].any(1) & mid_alts["mid_ir_v4"].any(1)
    v7_full, _ = apply_cfg(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], yt, mask0, V7_CFG)
    v7_nest = nested_fixed(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(ir0, th_vars["th_v6_v2trio"], mid_alts["mid_ir_v4"], mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    mods = inventory_modalities()
    print("modalities:", json.dumps(mods, indent=2), flush=True)

    model_specs = make_models()
    results = []

    for ir_name, th_name, mid_name in combos:
        if ir_name not in pools or th_name not in th_vars or mid_name not in mid_alts:
            print(f"skip missing {ir_name}|{th_name}|{mid_name}", flush=True)
            continue
        ir = pools[ir_name]
        th = th_vars[th_name]
        md = mid_alts[mid_name]
        mask = th.any(1) & md.any(1)
        if mask.sum() < 400:
            print(f"skip small mask {ir_name}|{th_name}|{mid_name} n={mask.sum()}", flush=True)
            continue

        base_row, base_preds = weight_blend_baseline(ir, th, md, yt, yu, mask)
        base_dis = int(((base_preds != v7_preds) & (base_preds >= 0) & (v7_preds >= 0)).sum())
        print(
            f"BASE {ir_name}|{th_name}|{mid_name} full={base_row['full']:.4f} "
            f"honest={base_row['honest_nested']:.4f} dis={base_dis}",
            flush=True,
        )

        # feature sets
        X_compact = build_stack_features_compact(ir, th, md, T=1.5)
        X_multi, _ = build_stack_features(ir, th, md, T_list=(1.0, 2.0))
        feat_sets = {
            "compact_T15": X_compact,
            "multiT_12": X_multi,
        }

        for feat_name, X in feat_sets.items():
            for model_name, proto in model_specs.items():
                def factory(proto=proto):
                    # clone via sklearn clone
                    from sklearn.base import clone
                    return clone(proto)

                try:
                    out = louo_stack(X, yt, yu, mask, factory, n_classes=40)
                except Exception as e:
                    print(f"  FAIL {model_name}/{feat_name}: {e}", flush=True)
                    continue
                oof = out["oof_pred"]
                # disagree vs v7 on oof where both valid
                both = mask & (oof >= 0) & (v7_preds >= 0)
                dis = int((oof[both] != v7_preds[both]).sum()) if both.any() else 0
                # also full-resub disagree (optimistic)
                full_pred = out["model_full"].predict(X[mask])
                full_preds_arr = np.full(len(yt), -1, dtype=np.int64)
                full_preds_arr[mask] = full_pred
                dis_full = int(((full_preds_arr != v7_preds) & (full_preds_arr >= 0) & (v7_preds >= 0)).sum())

                row = {
                    "kind": "louo_stack",
                    "ir": ir_name,
                    "th": th_name,
                    "mid": mid_name,
                    "feat": feat_name,
                    "model": model_name,
                    "full_resub": float(out["full_resub"]),
                    "honest_nested": float(out["nested"]),
                    "folds": out["folds"],
                    "disagree_vs_v7_oof": dis,
                    "disagree_vs_v7_full": dis_full,
                    "n_mask": int(mask.sum()),
                    "feat_dim": int(X.shape[1]),
                    "base_weight_honest": base_row["honest_nested"],
                    "base_weight_full": base_row["full"],
                    "delta_vs_base_nested": float(out["nested"] - base_row["honest_nested"]),
                    "delta_vs_gate_nested": float(out["nested"] - GATE),
                }
                # use min(full_resub, nested) is NOT right for gate — gate wants hold AND nested.
                # Hold estimate for stacker: we report full_resub as optimistic hold; honest is nested.
                # Also compute LOUO-mean as primary.
                results.append(row)
                print(
                    f"  {model_name}|{feat_name} nested={out['nested']:.4f} "
                    f"full_resub={out['full_resub']:.4f} dis_oof={dis} "
                    f"dBase={row['delta_vs_base_nested']:+.4f} folds="
                    f"{[round(f['te_acc'],3) for f in out['folds']]}",
                    flush=True,
                )

    # rank by honest nested then disagree
    results.sort(key=lambda r: (r["honest_nested"], r["disagree_vs_v7_oof"], r["full_resub"]), reverse=True)
    best = results[0] if results else None

    # thermal soft diversity without new train: stride/soft-T ensemble of EXISTING th logits
    soft_results = []
    if "th_v6_v2trio" in th_vars:
        th0 = th_vars["th_v6_v2trio"]
        # soft-prob average across temperatures as thermal diversity
        softs = []
        for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
            softs.append(softmax_np(th0, T))
        th_softens = np.mean(softs, 0)
        # convert back to logits via log
        th_soft_logit = np.log(np.clip(th_softens, 1e-8, 1.0)).astype(np.float32)
        th_vars["th_v6_softens_Tavg"] = th_soft_logit
        ir = pools["classic9_base"]
        md = mid_alts.get("mid_ens4_bonetcn", mid_alts["mid_ir_v4"])
        mask = th_soft_logit.any(1) & md.any(1)
        base_row, base_preds = weight_blend_baseline(ir, th_soft_logit, md, yt, yu, mask)
        X = build_stack_features_compact(ir, th_soft_logit, md, T=1.5)

        def factory_log():
            from sklearn.base import clone
            return clone(model_specs["logreg_l2"])

        out = louo_stack(X, yt, yu, mask, factory_log, n_classes=40)
        oof = out["oof_pred"]
        both = mask & (oof >= 0) & (v7_preds >= 0)
        dis = int((oof[both] != v7_preds[both]).sum()) if both.any() else 0
        soft_results.append({
            "kind": "thermal_softens_Tavg+logreg",
            "weight_blend_honest": base_row["honest_nested"],
            "weight_blend_full": base_row["full"],
            "stack_nested": float(out["nested"]),
            "stack_full_resub": float(out["full_resub"]),
            "disagree_oof": dis,
            "folds": out["folds"],
        })
        print(f"THERMAL softens Tavg weight_honest={base_row['honest_nested']:.4f} "
              f"stack_nested={out['nested']:.4f}", flush=True)

    clears = False
    if best is not None:
        # Gate: need hold+nested >= gate. For stacker, honest nested is primary;
        # full_resub is optimistic — require nested>=gate AND (full_resub>=gate) AND dis>=20
        clears = (
            best["honest_nested"] >= GATE
            and best["full_resub"] >= GATE
            and best["disagree_vs_v7_oof"] >= MIN_DISAGREE
        )

    status = {
        "tag": "ir_v21_louo_stack",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "modalities_inventory": mods,
        "mmwave_note": "Radar CSVs ~43B stubs — not usable; skip mmWave train",
        "imu_note": "IMU CSVs present (~7-30KB) — candidate next_roi if stack insufficient",
        "best": best,
        "top15": results[:15],
        "thermal_soft_probe": soft_results,
        "wrote_csv": False,
        "csv": None,
        "best_public": {
            "csv": "submission_ir_v7.csv",
            "public": 0.69154,
            "hold": V7_HOLD,
        },
        "delta_vs_gate": {
            "best_nested": (None if best is None else float(best["honest_nested"] - GATE)),
            "best_full_resub": (None if best is None else float(best["full_resub"] - GATE)),
        },
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "gpu_status_at_finish": "idle_unused_cpu_only",
        "finished_at": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "summary": {},
    }

    if best is None:
        status["next_roi"] = ["No stack results — check logits"]
    else:
        bn = best["honest_nested"]
        status["summary"] = {
            "best_model": best["model"],
            "best_feat": best["feat"],
            "best_combo": f"{best['ir']}|{best['th']}|{best['mid']}",
            "honest_nested": bn,
            "full_resub": best["full_resub"],
            "disagree_oof": best["disagree_vs_v7_oof"],
            "gate": GATE,
            "delta_nested": bn - GATE,
            "base_weight_honest": best["base_weight_honest"],
            "public_keep": "submission_ir_v7.csv @ 0.69154",
        }
        if clears:
            status["next_roi"] = [
                "WIN: LOUO stack cleared gate — write CSV carefully with nested-honest recipe",
                "Still avoid IR Kinetics churn; pack stacker + existing ckpts under 100MB",
            ]
        else:
            status["next_roi"] = [
                f"MISS gate: best LOUO nested={bn:.4f} full_resub={best['full_resub']:.4f} "
                f"dis={best['disagree_vs_v7_oof']} vs gate>={GATE:.4f}; keep ir_v7 @ 0.69154",
                f"Stack vs weight-blend dNested={best['delta_vs_base_nested']:+.4f} "
                f"(best {best['model']}/{best['feat']} on {best['ir']}|{best['th']}|{best['mid']})",
                "Radar/mmWave unusable (stub CSVs). IMU present — next: cheap IMU 1D-CNN/spectrogram aligned to clips",
                "Thermal soft-T ens without new R2+1D already probed; no more thermal seeds",
                "GPU unused (CPU stack) — handoff clear for LMT",
            ]

    out_path = ROOT / "metrics_ir_v21_status.json"
    # drop non-serializable
    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if k != "model_full"}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    out_path.write_text(json.dumps(scrub(status), indent=2), encoding="utf-8")
    print(f"Wrote {out_path} outcome={status['outcome']} best_nested="
          f"{None if best is None else round(best['honest_nested'],4)}", flush=True)
    print("next_roi:", status["next_roi"], flush=True)


if __name__ == "__main__":
    main()
