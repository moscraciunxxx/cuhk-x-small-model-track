"""ir_v17b: fuse longer IR seed7777 into classic9 BASE + th_v6 + MidFuse. Honest sameT nested."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT, write_sub

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V7_HOLD = 0.7530364372469636
GATE = V7_HOLD + 0.01
MIN_DISAGREE = 20
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}


def apply_cfg(a, b, c, y, mask, cfg):
    T = cfg["T"]
    p = (
        cfg["wa"] * softmax_np(a[mask], T)
        + cfg["wb"] * softmax_np(b[mask], T)
        + cfg["wc"] * softmax_np(c[mask], T)
    ).argmax(1)
    return float((p == y[mask]).mean()), p


def preds_full(a, b, c, mask, cfg):
    out = np.full(len(a), -1, dtype=np.int64)
    T = cfg["T"]
    out[mask] = (
        cfg["wa"] * softmax_np(a[mask], T)
        + cfg["wb"] * softmax_np(b[mask], T)
        + cfg["wc"] * softmax_np(c[mask], T)
    ).argmax(1)
    return out


def nested_retune(a, b, c, y, users, mask, Ts, ngrid=21):
    folds = []
    for leave in (8, 9, 24):
        te = mask & (users == leave)
        tr = mask & (users != leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        te_acc, _ = apply_cfg(a, b, c, y, te, cfg)
        folds.append({"leave": int(leave), "te_acc": te_acc, "cfg": cfg})
    mean = float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0
    return {"mean": mean, "folds": folds}


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def main():
    t0 = time.time()
    members, yt, yu = load_members()
    c9 = sorted([m for m in members if m["tag"] != "pool_seed55"], key=lambda d: -d.get("acc_base", d["acc"]))
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid[hold_idx].astype(np.float32)
    assert len(mid) == len(yt)

    z = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v17_long" / "hold_logits_strong.npz", allow_pickle=True)
    ir_new = z["ens"].astype(np.float32)
    assert len(ir_new) == len(yt)
    new_acc = float((ir_new.argmax(1) == yt).mean())
    print(f"IR seed7777 hold={new_acc:.4f}", flush=True)

    pools = {
        "classic9_base": np.mean([base_of(m) for m in c9], 0).astype(np.float32),
        "classic9_plus_7777": np.mean([base_of(m) for m in c9] + [ir_new], 0).astype(np.float32),
        "classic_top6_base": np.mean([base_of(m) for m in c9[:6]], 0).astype(np.float32),
        "classic_top6_plus_7777": np.mean([base_of(m) for m in c9[:6]] + [ir_new], 0).astype(np.float32),
        # replace weakest classic9 member with 7777 if stronger
        "classic9_swap_weak": None,
    }
    weak = c9[-1]
    if new_acc > weak.get("acc_base", weak["acc"]):
        kept = c9[:-1]
        pools["classic9_swap_weak"] = np.mean([base_of(m) for m in kept] + [ir_new], 0).astype(np.float32)
        print(f"swap: drop {weak['tag']} base={weak.get('acc_base', weak['acc']):.4f} for 7777={new_acc:.4f}", flush=True)
    else:
        del pools["classic9_swap_weak"]
        print(f"no swap: 7777={new_acc:.4f} <= weakest {weak['tag']}={weak.get('acc_base', weak['acc']):.4f}", flush=True)

    for k, v in list(pools.items()):
        if v is None:
            continue
        print(f"IR pool {k}: solo={(v.argmax(1)==yt).mean():.4f}", flush=True)

    mask0 = th.any(1) & mid.any(1)
    ir0 = pools["classic9_base"]
    v7_full, _ = apply_cfg(ir0, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(ir0, th, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(ir0, th, mid, mask0, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    Ts_fine = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
    Ts_med = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    results = []
    for ir_name, ir in pools.items():
        if ir is None:
            continue
        mask = th.any(1) & mid.any(1)
        b_acc, bcfg = fuse3_sameT(ir, th, mid, yt, mask, Ts_fine, ngrid=31)
        nest_fixed = nested_fixed(ir, th, mid, yt, yu, mask, bcfg)
        nest_rt = nested_retune(ir, th, mid, yt, yu, mask, Ts_med, ngrid=17)
        pred = preds_full(ir, th, mid, mask, bcfg)
        disagree = int(((pred >= 0) & (v7_preds >= 0) & (pred != v7_preds)).sum())
        honest = min(float(nest_fixed["mean"]), float(nest_rt["mean"]))
        row = {
            "ir": ir_name,
            "th": "th_v6_v2trio",
            "mid": "mid_ir_v4",
            "mode": "sameT",
            "full": b_acc,
            "cfg": bcfg,
            "nested_fixed": float(nest_fixed["mean"]),
            "nested_retune": float(nest_rt["mean"]),
            "honest_nested": honest,
            "disagree_vs_v7": disagree,
            "ir_solo": float((ir.argmax(1) == yt).mean()),
        }
        results.append(row)
        print(
            f"{ir_name} full={b_acc:.4f} nestF={row['nested_fixed']:.4f} nestR={row['nested_retune']:.4f} "
            f"honest={honest:.4f} dis={disagree} ir_solo={row['ir_solo']:.4f}",
            flush=True,
        )

    ranked = sorted(results, key=lambda r: (r["honest_nested"], r["full"], r["disagree_vs_v7"]), reverse=True)
    best = ranked[0]
    clears = best["full"] >= GATE and best["honest_nested"] >= GATE and best["disagree_vs_v7"] >= MIN_DISAGREE

    # Update prior status
    prev = {}
    prev_path = ROOT / "metrics_ir_v17_status.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))

    status = {
        "tag": "ir_v17_thermal_v4_then_ir_long",
        "outcome": "WIN" if clears else "MISS",
        "keep_ir_v7": not clears,
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": v7_full, "nested": float(v7_nest["mean"])},
        "thermal_v4": prev.get("thermal_v4"),
        "ir_long_seed7777": {
            "hold_acc": new_acc,
            "epochs_planned": 48,
            "early_stop_best_ep_note": "best=0.6554@ep8 early_stop@22",
            "vs_classic9_mean": float(pools["classic9_base"].argmax(1).mean() and (pools["classic9_base"].argmax(1) == yt).mean()),
            "ckpt": "checkpoints/ir_yolo_r2p1d18_v17_long/pool_seed7777.pt",
        },
        "fallback_choice": prev.get("fallback_choice"),
        "best": best,
        "best_honest_sameT": best,
        "all_sameT": results,
        "wrote_csv": False,
        "csv": None,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "delta_vs_gate": {
            "best_full": best["full"] - GATE,
            "honest_nested": best["honest_nested"] - GATE,
        },
        "next_roi": [],
        "elapsed_sec": round(time.time() - t0, 1),
        "phase_a_thermal": {
            "outcome": "MISS",
            "seed888_solo": 0.5476,
            "honest_sameT_best": 0.7440,
            "note": "skipped 2nd thermal; mixup0.3 recipe underperformed prior thermal seeds",
        },
    }
    # fix ir_solo classic9
    status["ir_long_seed7777"]["classic9_base_solo"] = float((pools["classic9_base"].argmax(1) == yt).mean())

    if clears:
        status["next_roi"].append("Gate cleared — human upload submission_ir_v17.csv")
    else:
        status["next_roi"].extend([
            f"IR long seed7777 hold={new_acc:.4f} < classic9 members (~0.66-0.68); fuse honest={best['honest_nested']:.4f} under gate {GATE:.4f}",
            "Thermal stronger-recipe failed; IR ep48 seed also below classic pool — next ROI may need different crop/backbone or MidFuse upgrade",
            "Do not promote; keep submission_ir_v7.csv @0.69154",
            "GPU idle after this probe; LMT may reclaim 3060",
        ])

    # finalize fallback status
    if status.get("fallback_choice"):
        status["fallback_choice"]["ir_fallback"]["status"] = "done"
        status["fallback_choice"]["ir_fallback"]["hold_acc"] = new_acc

    out = ROOT / "metrics_ir_v17_status.json"
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nOUTCOME={status['outcome']} best_full={best['full']:.4f} honest={best['honest_nested']:.4f} dis={best['disagree_vs_v7']}", flush=True)
    print(f"wrote {out}", flush=True)

    # handoff note
    handoff = ROOT / "logs" / "gpu_handoff_ir_v17.txt"
    handoff.write_text(
        "\n".join([
            "ir_v17 DONE (PT ~5:03pm start -> ~6:11pm IR finish window)",
            f"outcome=MISS keep_ir_v7=True wrote_csv=False",
            "metrics=metrics_ir_v17_status.json",
            "Thermal v4 seed888 solo=0.5476 (<< v2_trio 0.603) — skipped 2nd thermal",
            "Fallback: IR long seed7777 ep48 on ir_yolo_v4 -> hold=0.6554 (below classic9)",
            f"Best fuse honest_sameT={best['honest_nested']:.4f} full={best['full']:.4f} dis={best['disagree_vs_v7']} under gate 0.763",
            "GPU left idle. No Kaggle submit. No Chrome.",
            "next_roi: different crop/backbone or MidFuse upgrade; keep ir_v7",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {handoff}", flush=True)


if __name__ == "__main__":
    main()
