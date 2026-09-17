"""ir_v30: Thermal lift fuse — classic9 (+ optional v29b SAFE swaps) + new thermal + classic mid.
SAFE gate vs best public ir_v29b (0.70149): nested_fixed >= v29b local AND disagree <= 15 vs v29b/v7.
Writes submission_ir_v30.csv only if clears. No kaggle submit.
"""
from __future__ import annotations
import csv, json, shutil, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from dataset import DEFAULT_HOLD_OUT_USERS
from probe_ir_v24_fuse import (
    softmax_np, fuse3_sameT, apply_cfg, preds_full, nested_fixed, nested_retune, V7_CFG,
)
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
PT = timezone(timedelta(hours=-7))
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
CK_V6 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v6_lift"
CK_V5 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v5_rethink"
CK_V2 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2"
CK_V3 = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3"

# Promote vs v29b local nested (not just v7 floor)
V29B_NESTED = 0.75405874529429
NESTED_MIN = V29B_NESTED
MAX_DIS = 15
PMAX, AMIN, MAXCH = 0.40, 0.55, 15


def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]


def selective_swap(primary, aux, T, pmax, amin, maxch, allow_mask=None):
    pp, ap = softmax_np(primary, T), softmax_np(aux, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    pconf, aconf = pp.max(1), ap.max(1)
    mask = (pa != aa) & (pconf <= pmax) & (aconf >= amin)
    if allow_mask is not None:
        mask = mask & allow_mask
    cand = np.where(mask)[0]
    order = cand[np.lexsort((-aconf[cand], pconf[cand]))][:maxch]
    out = primary.copy()
    for i in order:
        out[i] = aux[i]
    return out, order.tolist()


def fuse3_sameT_near(a, b, c, y, mask, center, Ts, ngrid=11, radius=0.12):
    best = (-1.0, None)
    yt = y[mask]
    pa0, pb0, pc0 = a[mask], b[mask], c[mask]
    wa0, wb0, wc0 = center["wa"], center["wb"], center["wc"]
    was = np.unique(np.clip(np.linspace(wa0 - radius, wa0 + radius, ngrid), 0.0, 1.0))
    for T in Ts:
        pa, pb, pc = softmax_np(pa0, T), softmax_np(pb0, T), softmax_np(pc0, T)
        for wa in was:
            for wb in np.unique(np.clip(np.linspace(wb0 - radius, wb0 + radius, ngrid), 0.0, 1.0 - wa)):
                wc = 1.0 - wa - wb
                if abs(wc - wc0) > radius + 1e-9 or wc < -1e-9:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "sameT_near"})
    return best


def load_hold_from_ckpt(ck: Path, n_hold: int):
    if not ck.exists():
        return None, None
    blob = __import__("torch").load(ck, map_location="cpu", weights_only=False)
    h = blob.get("hold_logits")
    if h is None:
        return None, float(blob.get("val_acc", -1))
    h = np.asarray(h, dtype=np.float32)
    if h.shape[0] != n_hold:
        # pad/truncate carefully — thermal hold is 504 vs ir 505 sometimes
        return h, float(blob.get("val_acc", -1))
    return h, float(blob.get("val_acc", -1))


def align_thermal_to_ir(th_hold, yt_len=505):
    """hold_thermal_v6 is 505 with some zero rows; member holds are often 504."""
    if th_hold is None:
        return None
    if th_hold.shape[0] == yt_len:
        return th_hold.astype(np.float32)
    # Insert zero row where v6 has empty — match by copying into non-zero mask slots
    v6 = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy")
    out = np.zeros((yt_len, th_hold.shape[1]), np.float32)
    nz = np.where(v6.any(1))[0]
    if len(nz) == len(th_hold):
        out[nz] = th_hold
        return out
    # fallback: place into first len rows of nz or truncate
    n = min(len(nz), len(th_hold))
    out[nz[:n]] = th_hold[:n]
    return out


def main():
    t0 = time.time()
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th_v6 = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = mid_full[hold_idx].astype(np.float32)
    t24 = np.load(CK24 / "hold_logits_seed42.npy").astype(np.float32)
    mask0 = th_v6.any(1) & mid.any(1)
    T = V7_CFG["T"]

    # IR pools: classic9 and v29b-style swap
    ir_v29b, ch = selective_swap(classic9, t24, T, PMAX, AMIN, MAXCH)
    ir_pools = {
        "classic9": classic9,
        "v29b_errdrive": ir_v29b,
    }

    # Thermal pools
    th_pools = {"th_v6_v2trio": th_v6}
    # v5 seed2026
    h5, a5 = load_hold_from_ckpt(CK_V5 / "pool_seed2026.pt", 504)
    if h5 is not None:
        th_pools["th_v5_s2026"] = align_thermal_to_ir(h5, len(yt))
        print(f"th_v5_s2026 acc={a5:.4f}", flush=True)
    # v6 lift seeds
    v6_holds = []
    v6_accs = []
    for seed in (3141, 7777, 2026, 4096, 1337):
        ck = CK_V6 / f"pool_seed{seed}.pt"
        h, a = load_hold_from_ckpt(ck, 504)
        if h is None:
            continue
        aligned = align_thermal_to_ir(h, len(yt))
        th_pools[f"th_v6_s{seed}"] = aligned
        v6_holds.append(aligned)
        v6_accs.append(a)
        print(f"th_v6_s{seed} acc={a:.4f}", flush=True)
    if len(v6_holds) >= 2:
        ens = np.mean(np.stack(v6_holds, 0), 0).astype(np.float32)
        th_pools["th_v6_ens"] = ens
        # blend with v2trio
        th_pools["th_v6_ens_0.5_v2"] = (0.5 * ens + 0.5 * th_v6).astype(np.float32)
        th_pools["th_v6_ens_0.3_v2"] = (0.3 * ens + 0.7 * th_v6).astype(np.float32)
        th_pools["th_v6_ens_0.7_v2"] = (0.7 * ens + 0.3 * th_v6).astype(np.float32)
    elif len(v6_holds) == 1:
        th_pools["th_v6_ens_0.5_v2"] = (0.5 * v6_holds[0] + 0.5 * th_v6).astype(np.float32)
        th_pools["th_v6_ens_0.3_v2"] = (0.3 * v6_holds[0] + 0.7 * th_v6).astype(np.float32)
    if "th_v5_s2026" in th_pools:
        th_pools["th_v5_0.5_v2"] = (0.5 * th_pools["th_v5_s2026"] + 0.5 * th_v6).astype(np.float32)
        if v6_holds:
            th_pools["th_v5v6_0.5_v2"] = (
                0.35 * th_pools["th_v5_s2026"] + 0.35 * v6_holds[0] + 0.30 * th_v6
            ).astype(np.float32)

    # Baseline refs
    v7_full, _ = apply_cfg(classic9, th_v6, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th_v6, mid, yt, yu, mask0, V7_CFG)
    v7_preds = preds_full(classic9, th_v6, mid, mask0, V7_CFG)
    v29_full, _ = apply_cfg(ir_v29b, th_v6, mid, yt, mask0, V7_CFG)
    v29_nest = nested_fixed(ir_v29b, th_v6, mid, yt, yu, mask0, V7_CFG)
    v29_preds = preds_full(ir_v29b, th_v6, mid, mask0, V7_CFG)
    print(f"v7 full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)
    print(f"v29b full={v29_full:.6f} nested={v29_nest['mean']:.6f} (gate nested_min={NESTED_MIN})", flush=True)
    print(f"thermal pools: {list(th_pools.keys())}", flush=True)

    Ts_near = [2.0, 2.25, 2.5, 2.75, 3.0]
    Ts_med = [1.5, 2.0, 2.5, 3.0]
    results = []

    for ir_name, ir in ir_pools.items():
        for th_name, th in th_pools.items():
            mask = th.any(1) & mid.any(1)
            # fixed V7_CFG
            full_f, _ = apply_cfg(ir, th, mid, yt, mask, V7_CFG)
            nest_f = nested_fixed(ir, th, mid, yt, yu, mask, V7_CFG)
            pred_f = preds_full(ir, th, mid, mask, V7_CFG)
            dis_v7 = int(((pred_f >= 0) & (v7_preds >= 0) & (pred_f != v7_preds)).sum())
            dis_v29 = int(((pred_f >= 0) & (v29_preds >= 0) & (pred_f != v29_preds)).sum())
            honest = float(nest_f["mean"])
            clears = bool(
                honest >= NESTED_MIN
                and dis_v7 <= MAX_DIS
                and dis_v29 <= MAX_DIS
                and honest > float(v29_nest["mean"]) + 1e-12
            )
            row = {
                "ir": ir_name, "th": th_name, "mid": "classic_aligned_mid", "path": "fixed_v7cfg",
                "full": float(full_f), "cfg": dict(V7_CFG),
                "nested_fixed": honest, "disagree_vs_v7": dis_v7, "disagree_vs_v29b": dis_v29,
                "delta_vs_v29b": honest - float(v29_nest["mean"]),
                "clears": clears,
            }
            results.append(row)

            # near-v7 sameT
            b_acc, bcfg = fuse3_sameT_near(ir, th, mid, yt, mask, V7_CFG, Ts_near, ngrid=9, radius=0.10)
            if bcfg is None:
                continue
            nest_n = nested_fixed(ir, th, mid, yt, yu, mask, bcfg)
            pred_n = preds_full(ir, th, mid, mask, bcfg)
            dis_v7n = int(((pred_n >= 0) & (v7_preds >= 0) & (pred_n != v7_preds)).sum())
            dis_v29n = int(((pred_n >= 0) & (v29_preds >= 0) & (pred_n != v29_preds)).sum())
            honest_n = float(nest_n["mean"])
            clears_n = bool(
                honest_n >= NESTED_MIN
                and dis_v7n <= MAX_DIS
                and dis_v29n <= MAX_DIS
                and honest_n > float(v29_nest["mean"]) + 1e-12
            )
            row_n = {
                "ir": ir_name, "th": th_name, "mid": "classic_aligned_mid", "path": "near_v7_sameT",
                "full": float(b_acc), "cfg": bcfg,
                "nested_fixed": honest_n, "disagree_vs_v7": dis_v7n, "disagree_vs_v29b": dis_v29n,
                "delta_vs_v29b": honest_n - float(v29_nest["mean"]),
                "clears": clears_n,
            }
            results.append(row_n)
            if clears or clears_n or dis_v29 <= 20 or dis_v29n <= 20:
                print(
                    f"{ir_name}|{th_name} fixed n={honest:.4f} d29={dis_v29} d7={dis_v7} | "
                    f"near n={honest_n:.4f} d29={dis_v29n} d7={dis_v7n} clear={clears or clears_n}",
                    flush=True,
                )

    ranked = sorted(
        results,
        key=lambda r: (r["clears"], r["nested_fixed"], -r["disagree_vs_v29b"], -r["disagree_vs_v7"], r["full"]),
        reverse=True,
    )
    clears_list = [r for r in ranked if r["clears"]]
    best = clears_list[0] if clears_list else ranked[0]

    wrote_csv = False
    out_csv = ROOT / "submission_ir_v30.csv"
    test_info = None
    if clears_list:
        best = clears_list[0]
        cfg = best["cfg"]
        ir_name, th_name = best["ir"], best["th"]
        # Build test IR
        ir_test = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
        t24_test = np.load(CK24 / "test_logits_seed42.npy").astype(np.float32)
        if ir_name == "v29b_errdrive":
            ir_test, ch_te = selective_swap(ir_test, t24_test, cfg["T"], PMAX, AMIN, MAXCH)
        else:
            ch_te = []
        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
        # Thermal test
        th_test = None
        if th_name == "th_v6_v2trio" or th_name.endswith("_v2") and "ens" not in th_name and "v5" not in th_name:
            th_p = CK_V3 / "test_logits.npy"
            if not th_p.exists():
                th_p = CK_V3 / "test_logits_final.npy"
            th_test = np.load(th_p).astype(np.float32)
        elif th_name.startswith("th_v6_s"):
            seed = int(th_name.replace("th_v6_s", ""))
            p = CK_V6 / f"test_logits_seed{seed}.npy"
            if p.exists():
                th_test = np.load(p).astype(np.float32)
        elif th_name == "th_v6_ens":
            parts = []
            for seed in (3141, 7777):
                p = CK_V6 / f"test_logits_seed{seed}.npy"
                if p.exists():
                    parts.append(np.load(p).astype(np.float32))
            if parts:
                th_test = np.mean(np.stack(parts, 0), 0).astype(np.float32)
        elif "v6_ens" in th_name and "v2" in th_name:
            parts = []
            for seed in (3141, 7777):
                p = CK_V6 / f"test_logits_seed{seed}.npy"
                if p.exists():
                    parts.append(np.load(p).astype(np.float32))
            th_old = np.load(CK_V3 / "test_logits.npy").astype(np.float32) if (CK_V3 / "test_logits.npy").exists() else None
            if parts and th_old is not None:
                ens = np.mean(np.stack(parts, 0), 0)
                if "0.5" in th_name:
                    th_test = (0.5 * ens + 0.5 * th_old).astype(np.float32)
                elif "0.3" in th_name:
                    th_test = (0.3 * ens + 0.7 * th_old).astype(np.float32)
                elif "0.7" in th_name:
                    th_test = (0.7 * ens + 0.3 * th_old).astype(np.float32)
        elif th_name.startswith("th_v5"):
            p = CK_V5 / "test_logits_seed2026.npy"
            th_new = np.load(p).astype(np.float32) if p.exists() else None
            th_old = np.load(CK_V3 / "test_logits.npy").astype(np.float32) if (CK_V3 / "test_logits.npy").exists() else None
            if th_name == "th_v5_s2026" and th_new is not None:
                th_test = th_new
            elif th_name == "th_v5_0.5_v2" and th_new is not None and th_old is not None:
                th_test = (0.5 * th_new + 0.5 * th_old).astype(np.float32)
            elif th_name == "th_v5v6_0.5_v2" and th_new is not None and th_old is not None:
                p6 = CK_V6 / "test_logits_seed3141.npy"
                t6 = np.load(p6).astype(np.float32) if p6.exists() else th_new
                th_test = (0.35 * th_new + 0.35 * t6 + 0.30 * th_old).astype(np.float32)

        if th_test is None:
            # fallback: keep v2trio test so we don't invent
            th_p = CK_V3 / "test_logits.npy"
            th_test = np.load(th_p).astype(np.float32)
            print(f"WARN: no mapped test thermal for {th_name}; using v3 test_logits", flush=True)

        preds_v7 = (V7_CFG["wa"] * softmax_np(ir_test if ir_name == "classic9" else
                    np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32),
                    V7_CFG["T"]) + V7_CFG["wb"] * softmax_np(
                    np.load(CK_V3 / "test_logits.npy").astype(np.float32), V7_CFG["T"])
                    + V7_CFG["wc"] * softmax_np(mid_test, V7_CFG["T"])).argmax(1)
        # cleaner v7 and v29b preds
        ir_test_c9 = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy").astype(np.float32)
        th_test_v2 = np.load(CK_V3 / "test_logits.npy").astype(np.float32)
        preds_v7 = (V7_CFG["wa"] * softmax_np(ir_test_c9, V7_CFG["T"]) + V7_CFG["wb"] * softmax_np(th_test_v2, V7_CFG["T"])
                    + V7_CFG["wc"] * softmax_np(mid_test, V7_CFG["T"])).argmax(1)
        ir_test_v29, _ = selective_swap(ir_test_c9, t24_test, V7_CFG["T"], PMAX, AMIN, MAXCH)
        preds_v29 = (V7_CFG["wa"] * softmax_np(ir_test_v29, V7_CFG["T"]) + V7_CFG["wb"] * softmax_np(th_test_v2, V7_CFG["T"])
                     + V7_CFG["wc"] * softmax_np(mid_test, V7_CFG["T"])).argmax(1)
        preds = (cfg["wa"] * softmax_np(ir_test, cfg["T"]) + cfg["wb"] * softmax_np(th_test, cfg["T"])
                 + cfg["wc"] * softmax_np(mid_test, cfg["T"])).argmax(1)
        test_dis_v7 = int((preds != preds_v7).sum())
        test_dis_v29 = int((preds != preds_v29).sum())
        test_info = {"n_swap_ir": len(ch_te), "disagree_vs_v7": test_dis_v7, "disagree_vs_v29b": test_dis_v29,
                     "th_name": th_name, "ir_name": ir_name, "cfg": cfg}

        if test_dis_v7 <= MAX_DIS and test_dis_v29 <= MAX_DIS:
            v7_rows = list(csv.DictReader(open(ROOT / "submission_ir_v7.csv", encoding="utf-8")))
            meta = json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_meta.json").read_text(encoding="utf-8"))
            empty = set(json.loads((ROOT / "cache" / "ir_yolo_v4" / "test_empty.json").read_text(encoding="utf-8")))
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["path", "prediction"])
                for i, row in enumerate(v7_rows):
                    sid = meta[i]["sample_id"] if i < len(meta) else None
                    if sid in empty:
                        w.writerow([row["path"], row["prediction"]])
                    else:
                        w.writerow([row["path"], int(preds[i])])
            wrote_csv = True
            # promote track only if improves nested vs v29b
            shutil.copy2(out_csv, TRACK / "submission.csv")
            print(f"WROTE {out_csv} test_dis_v7={test_dis_v7} test_dis_v29={test_dis_v29} PROMOTED track", flush=True)
        else:
            print(f"HOLD clear but test disagree too high: v7={test_dis_v7} v29={test_dis_v29}; no CSV", flush=True)

    now = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")
    status = {
        "tag": "ir_v30_thermal_lift",
        "updated_at": now,
        "best_public": {"csv": "submission_ir_v29b.csv", "public": 0.70149},
        "outcome": "SAFE_WIN" if wrote_csv else ("HOLD_CLEAR_NO_CSV" if clears_list else "MISS_OR_WAITING"),
        "wrote_csv": wrote_csv,
        "csv": str(out_csv) if wrote_csv else None,
        "gate": {"nested_min": NESTED_MIN, "max_disagree": MAX_DIS, "vs": "v29b_local_and_v7"},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "v29b_reproduce": {"full": float(v29_full), "nested": float(v29_nest["mean"])},
        "v6_member_accs": {f"s{s}": a for s, a in zip((3141, 7777), v6_accs)} if v6_accs else {},
        "best": best,
        "n_clear": len(clears_list),
        "top15": ranked[:15],
        "top_clears": clears_list[:10],
        "test": test_info,
        "elapsed_sec": round(time.time() - t0, 1),
        "next_roi": [
            "If MISS: wait for stronger v6 seeds or try mixup0 + unfreeze@12",
            "If SAFE_WIN: parent may kaggle-submit submission_ir_v30.csv if looks >0.70149",
            "Yield GPU to HOTC/LMT when idle",
        ],
    }
    (ROOT / "metrics_ir_v30_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v30_thermal_lift.json").write_text(
        json.dumps({"best": best, "clears": clears_list[:20], "top30": ranked[:30], "n": len(results)}, indent=2),
        encoding="utf-8",
    )
    print("BEST", best.get("path"), best.get("ir"), best.get("th"),
          "nested", best.get("nested_fixed"), "d29", best.get("disagree_vs_v29b"),
          "clears", best.get("clears"), "n_clear", len(clears_list), "wrote", wrote_csv)
    # refresh handoff
    hand = ROOT / "HOTC_GPU_HANDOFF.md"
    hand.write_text(
        f"""# CUHK-X Small Model Track - status

## BEST PUBLIC: ir_v29b @ **0.70149**
## ir_v30 thermal lift: {status['outcome']}
- best nested={best.get('nested_fixed')} d29={best.get('disagree_vs_v29b')} d7={best.get('disagree_vs_v7')}
- clears={len(clears_list)} wrote_csv={wrote_csv}
- v6 accs={status.get('v6_member_accs')}

## GPU: see nvidia-smi; Thermal v6 may still train

Updated {now}
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

