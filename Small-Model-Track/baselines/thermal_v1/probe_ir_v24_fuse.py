"""ir_v24 fuse probe: nest-honest sameT of new IR (X3D-M / R50) vs classic9 + th_v6 + mid_ens.
Gate: hold+nested >= 0.758 (prefer), disagree>=20 vs ir_v7. No CSV unless clears.
"""
from __future__ import annotations
import json, time, argparse
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS

ROOT = Path(__file__).resolve().parent
V7_HOLD = 0.7530364372469636
GATE = 0.758  # relaxed slightly per user order
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}


def softmax_np(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


def fuse3_sameT(a, b, c, y, mask, Ts, ngrid=21):
    best = (-1.0, None)
    yt = y[mask]
    pa0 = a[mask]; pb0 = b[mask]; pc0 = c[mask]
    for T in Ts:
        pa, pb, pc = softmax_np(pa0, T), softmax_np(pb0, T), softmax_np(pc0, T)
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, ngrid):
                wc = 1 - wa - wb
                if wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T), "acc": acc, "n": int(mask.sum()), "mode": "sameT"})
    return best


def apply_cfg(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T) + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return float((p == y[mask]).mean()), p


def preds_full(a, b, c, mask, cfg):
    out = np.full(len(a), -1, dtype=np.int64)
    T = cfg["T"]
    out[mask] = (cfg["wa"] * softmax_np(a[mask], T) + cfg["wb"] * softmax_np(b[mask], T) + cfg["wc"] * softmax_np(c[mask], T)).argmax(1)
    return out


def nested_fixed(a, b, c, y, users, mask, cfg):
    folds = []
    for leave in (8, 9, 24):
        te = mask & (users == leave)
        if te.sum() < 5:
            continue
        te_acc, _ = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum())})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=17):
    folds = []
    for leave in (8, 9, 24):
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc, _ = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "n": int(te.sum()), "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def load_classic9():
    from fuse_ir_v9 import load_members
    members, yt, yu = load_members()
    c9 = sorted([m for m in members if m["tag"] != "pool_seed55"], key=lambda d: -d.get("acc_base", d["acc"]))
    def base_of(m):
        return m["base"] if m.get("base") is not None else m["logits"]
    classic9 = np.mean([base_of(m) for m in c9], 0).astype(np.float32)
    return classic9, yt, yu, c9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-hold", required=True, help="npz/npy with ens hold logits or .npy")
    ap.add_argument("--tag", default="x3d_m_s42")
    ap.add_argument("--status", default=str(ROOT / "metrics_ir_v24_status.json"))
    args = ap.parse_args()
    t0 = time.time()

    classic9, yt, yu, c9 = load_classic9()
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)
    assert len(mid) == len(yt) == len(th) == len(classic9)

    p = Path(args.new_hold)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        ir_new = z["ens"].astype(np.float32) if "ens" in z.files else z[z.files[0]].astype(np.float32)
    else:
        ir_new = np.load(p).astype(np.float32)
    assert len(ir_new) == len(yt)
    new_acc = float((ir_new.argmax(1) == yt).mean())
    print(f"new IR {args.tag} solo hold={new_acc:.4f}", flush=True)

    pools = {
        "classic9_base": classic9,
        f"classic9_plus_{args.tag}": np.mean([classic9, ir_new], 0).astype(np.float32),
        f"new_only_{args.tag}": ir_new,
        f"classic9_mix2_{args.tag}": (0.7 * classic9 + 0.3 * ir_new).astype(np.float32),
        f"classic9_mix3_{args.tag}": (0.5 * classic9 + 0.5 * ir_new).astype(np.float32),
    }
    # swap weakest if stronger
    weak = c9[-1]
    weak_acc = weak.get("acc_base", weak["acc"])
    if new_acc > weak_acc:
        kept = [m["base"] if m.get("base") is not None else m["logits"] for m in c9[:-1]]
        pools[f"classic9_swap_{args.tag}"] = np.mean(kept + [ir_new], 0).astype(np.float32)
        print(f"swap weak {weak['tag']}={weak_acc:.4f} for new={new_acc:.4f}", flush=True)

    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    Ts_fine = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    Ts_med = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    results = []
    for ir_name, ir in pools.items():
        mask = mask0
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask, Ts_fine, ngrid=31)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask, bcfg)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask, Ts_med, ngrid=17)
        pred = preds_full(ir, th, mid, mask, bcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = min(float(nest_fixed["mean"]), float(nest_rt["mean"]))
        row = {
            "ir": ir_name, "th": "th_v6_v2trio", "mid": "mid_ens4_bonetcn",
            "full": b_acc, "cfg": bcfg,
            "nested_fixed": float(nest_fixed["mean"]),
            "nested_retune": float(nest_rt["mean"]),
            "honest_nested": honest,
            "disagree_vs_v7": disagree,
            "ir_solo": float((ir.argmax(1) == yt).mean()),
            "clears": bool(b_acc >= GATE and honest >= GATE and disagree >= MIN_DISAGREE),
        }
        results.append(row)
        print(f"{ir_name} full={b_acc:.4f} nestF={row['nested_fixed']:.4f} nestR={row['nested_retune']:.4f} "
              f"honest={honest:.4f} dis={disagree} solo={row['ir_solo']:.4f} clear={row['clears']}", flush=True)

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best = ranked[0]
    clears = best["clears"]

    status = {
        "tag": "ir_v24_x3d_r50",
        "outcome": "WIN" if clears else "IN_PROGRESS_OR_MISS",
        "keep_ir_v7": not clears,
        "finished_partial": True,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": v7_full, "nested": float(v7_nest["mean"])},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "new_ir": {"tag": args.tag, "solo_hold": new_acc, "path": str(p)},
        "best": best,
        "top8": ranked[:8],
        "wrote_csv": False,
        "csv": None,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(args.status).write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("wrote", args.status, "best_honest", best["honest_nested"], "clears", clears, flush=True)


if __name__ == "__main__":
    main()
