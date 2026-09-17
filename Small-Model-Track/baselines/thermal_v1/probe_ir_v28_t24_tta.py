"""ir_v28: TTA on existing T24 focal_ft seeds (esp s42@0.697) + classic-mid near-v7 SAFE gate.
No new train. CSV/kaggle only if nested_fixed>=0.75196 AND disagree_vs_v7<=15.
strongb finished early-stop 0.663 << aim — this is the fallback path.
"""
from __future__ import annotations
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES
from probe_ir_v24_fuse import softmax_np, apply_cfg, preds_full, nested_fixed, V7_CFG
from fuse_ir_v9 import load_members

ROOT = Path(__file__).resolve().parent
PT = timezone(timedelta(hours=-7))
HOLD = set(DEFAULT_HOLD_OUT_USERS)
CK24 = ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_t24_v24"
CACHE = ROOT / "cache" / "ir_yolo_ft_v24_t24"
NESTED_MIN = 0.75196
MAX_DIS = 15
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

def now_pt():
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")

def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)

def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

@torch.no_grad()
def eval_views(model, loader, device):
    """base / flip / trev / mean combos on hold. x from CachedClipDataset: B,T,C,H,W"""
    model.eval()
    buckets = {k: [] for k in ("base", "flip", "trev", "flip_trev")}
    ys = []
    for x, y, _u, _i in loader:
        x = x.to(device)  # B,T,C,H,W
        def fwd(xx):
            xx = normalize(xx).permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,H,W
            return model(xx).float()
        b = fwd(x)
        f = fwd(torch.flip(x, dims=[-1]))          # flip W
        tr = fwd(torch.flip(x, dims=[1]))          # reverse T
        ft = fwd(torch.flip(torch.flip(x, dims=[1]), dims=[-1]))
        buckets["base"].append(b.cpu().numpy())
        buckets["flip"].append(f.cpu().numpy())
        buckets["trev"].append(tr.cpu().numpy())
        buckets["flip_trev"].append(ft.cpu().numpy())
        ys.append(y.numpy())
    yt = np.concatenate(ys)
    out = {k: np.concatenate(v).astype(np.float32) for k, v in buckets.items()}
    out["bf"] = (0.5 * (out["base"] + out["flip"])).astype(np.float32)
    out["bft"] = ((out["base"] + out["flip"] + out["trev"] + out["flip_trev"]) / 4.0).astype(np.float32)
    out["b_f_tr"] = ((out["base"] + out["flip"] + out["trev"]) / 3.0).astype(np.float32)
    return out, yt

def base_of(m):
    return m["base"] if m.get("base") is not None else m["logits"]

def near_cfg(ir, th, mid, yt, mask0):
    best = (-1.0, None)
    yt_m = yt[mask0]
    for T in (2.25, 2.5, 2.75, 3.0):
        pa = softmax_np(ir[mask0], T)
        pb = softmax_np(th[mask0], T)
        pc = softmax_np(mid[mask0], T)
        for wa in np.linspace(0.50, 0.62, 7):
            for wb in np.linspace(0.28, 0.40, 7):
                wc = 1 - wa - wb
                if not (0.05 <= wc <= 0.15):
                    continue
                if abs(wc - 0.09) > 0.06:
                    continue
                acc = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt_m).mean())
                if acc > best[0]:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T), "acc": acc, "mode": "near"})
    return best

def conf_gate(primary, aux, T=2.5, thr=0.55, max_changes=15):
    pp = softmax_np(primary, T)
    ap = softmax_np(aux, T)
    pa, aa = pp.argmax(1), ap.argmax(1)
    conf = pp.max(1)
    cand = np.where((pa != aa) & (conf <= thr))[0]
    order = cand[np.argsort(conf[cand])][:max_changes]
    out = primary.copy()
    out[order] = aux[order]
    return out, int(len(order))

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    y = np.load(CACHE / "train_y.npy")
    users = np.load(CACHE / "train_users.npy")
    X = np.memmap(CACHE / "train_x_t24_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 24, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    vl = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=4, shuffle=False)

    # Prefer strongest existing seeds for TTA: 42, 888, 2025
    seed_ckpts = []
    for sid in (42, 888, 2025):
        p = CK24 / f"pool_seed{sid}.pt"
        if p.exists():
            seed_ckpts.append((sid, p))
    print("TTA seeds", [s for s, _ in seed_ckpts], flush=True)

    tta_by_seed = {}
    for sid, p in seed_ckpts:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model = build().to(device)
        model.load_state_dict(blob["model"])
        views, yt_hold = eval_views(model, vl, device)
        for vk, logits in views.items():
            acc = float((logits.argmax(1) == yt_hold).mean())
            print(f"seed{sid} {vk:8s} hold={acc:.4f}", flush=True)
        tta_by_seed[sid] = views
        # persist best useful views
        np.save(CK24 / f"hold_logits_seed{sid}_tta_bf.npy", views["bf"])
        np.save(CK24 / f"hold_logits_seed{sid}_tta_bft.npy", views["bft"])
        del model
        torch.cuda.empty_cache()

    # Build IR pools for fuse
    members, yt, yu = load_members()
    allc = sorted(members, key=lambda d: -d.get("acc_base", d["acc"]))
    c9m = [m for m in allc if m["tag"] != "pool_seed55"][:9]
    classic9 = np.mean([base_of(m) for m in c9m], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx_v4 = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy")[hold_idx_v4].astype(np.float32)
    ff = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_focal_ft_v24" / "hold_logits_strong.npz")["ens"].astype(np.float32)

    seeds_base = {int(p.stem.replace("hold_logits_seed", "")): np.load(p).astype(np.float32)
                  for p in CK24.glob("hold_logits_seed*.npy") if "_tta_" not in p.stem}
    phaseA = np.mean([seeds_base[42], seeds_base[888], seeds_base[2024]], 0).astype(np.float32)
    t42 = seeds_base[42]
    t42_bf = tta_by_seed[42]["bf"]
    t42_bft = tta_by_seed[42]["bft"]
    top2_bf = np.mean([tta_by_seed[42]["bf"], tta_by_seed[888]["bf"]], 0).astype(np.float32) if 888 in tta_by_seed else t42_bf
    top3_bf = np.mean([tta_by_seed[s]["bf"] for s in (42, 888, 2025) if s in tta_by_seed], 0).astype(np.float32)
    top3_bft = np.mean([tta_by_seed[s]["bft"] for s in (42, 888, 2025) if s in tta_by_seed], 0).astype(np.float32)

    mask0 = th.any(1) & mid.any(1)
    v7_full, _ = apply_cfg(classic9, th, mid, yt, mask0, V7_CFG)
    v7_nest = nested_fixed(classic9, th, mid, yt, yu, mask0, V7_CFG)["mean"]
    v7_preds = preds_full(classic9, th, mid, mask0, V7_CFG)
    print(f"v7 full={v7_full:.6f} nested_fixed={v7_nest:.6f}", flush=True)

    pools = {
        "classic9": classic9,
        "t42_base": t42,
        "t42_bf": t42_bf,
        "t42_bft": t42_bft,
        "top2_bf": top2_bf,
        "top3_bf": top3_bf,
        "top3_bft": top3_bft,
        "c9_0.90_t42bf_0.10": (0.90 * classic9 + 0.10 * t42_bf).astype(np.float32),
        "c9_0.85_t42bf_0.15": (0.85 * classic9 + 0.15 * t42_bf).astype(np.float32),
        "c9_0.80_t42bf_0.20": (0.80 * classic9 + 0.20 * t42_bf).astype(np.float32),
        "c9_0.90_t42bft_0.10": (0.90 * classic9 + 0.10 * t42_bft).astype(np.float32),
        "c9_0.85_t42bft_0.15": (0.85 * classic9 + 0.15 * t42_bft).astype(np.float32),
        "c9_0.90_top3bf_0.10": (0.90 * classic9 + 0.10 * top3_bf).astype(np.float32),
        "c9_0.85_top3bf_0.15": (0.85 * classic9 + 0.15 * top3_bf).astype(np.float32),
        "c9_0.80_top3bf_0.20": (0.80 * classic9 + 0.20 * top3_bf).astype(np.float32),
        "c9_0.90_top3bft_0.10": (0.90 * classic9 + 0.10 * top3_bft).astype(np.float32),
        "c9_0.85_top3bft_0.15": (0.85 * classic9 + 0.15 * top3_bft).astype(np.float32),
        "c9_0.90_pa_0.10": (0.90 * classic9 + 0.10 * phaseA).astype(np.float32),
        "c9_0.85_pa_0.15": (0.85 * classic9 + 0.15 * phaseA).astype(np.float32),
        "c9_0.92_pa_0.04_t42bf_0.04": (0.92 * classic9 + 0.04 * phaseA + 0.04 * t42_bf).astype(np.float32),
    }
    for thr in (0.45, 0.55, 0.65):
        for name, aux in (("t42bf", t42_bf), ("t42bft", t42_bft), ("top3bf", top3_bf), ("top3bft", top3_bft), ("pa", phaseA), ("ff", ff)):
            gated, n = conf_gate(classic9, aux, T=2.5, thr=thr, max_changes=15)
            pools[f"cgate15_{name}_thr{thr}"] = gated

    rows = []
    for ir_name, ir in pools.items():
        full_f, _ = apply_cfg(ir, th, mid, yt, mask0, V7_CFG)
        nest_f = nested_fixed(ir, th, mid, yt, yu, mask0, V7_CFG)["mean"]
        pred_f = preds_full(ir, th, mid, mask0, V7_CFG)
        dis_f = int(((pred_f >= 0) & (v7_preds >= 0) & (pred_f != v7_preds)).sum())
        rows.append({"ir": ir_name, "path": "fixed", "full": float(full_f), "nested": float(nest_f), "dis": dis_f,
                     "solo": float((ir.argmax(1) == yt).mean()), "delta": float(nest_f - v7_nest), "cfg": dict(V7_CFG)})
        bacc, bcfg = near_cfg(ir, th, mid, yt, mask0)
        if bcfg:
            nest_n = nested_fixed(ir, th, mid, yt, yu, mask0, bcfg)["mean"]
            pred_n = preds_full(ir, th, mid, mask0, bcfg)
            dis_n = int(((pred_n >= 0) & (v7_preds >= 0) & (pred_n != v7_preds)).sum())
            rows.append({"ir": ir_name, "path": "near", "full": float(bacc), "nested": float(nest_n), "dis": dis_n,
                         "solo": float((ir.argmax(1) == yt).mean()), "delta": float(nest_n - v7_nest), "cfg": bcfg})
        print(f"{ir_name:32s} fixed nest={nest_f:.4f} full={full_f:.4f} d={dis_f} solo={float((ir.argmax(1)==yt).mean()):.4f}", flush=True)

    # fuse-level conf gate vs v7 probs
    T = V7_CFG["T"]
    p_v7 = (V7_CFG["wa"] * softmax_np(classic9, T) + V7_CFG["wb"] * softmax_np(th, T) + V7_CFG["wc"] * softmax_np(mid, T))
    for aux_name, aux_ir in (
        ("t42bf", t42_bf), ("t42bft", t42_bft), ("top3bf", top3_bf), ("top3bft", top3_bft),
        ("c9t42bf", pools["c9_0.85_t42bf_0.15"]), ("c9top3bf", pools["c9_0.85_top3bf_0.15"]),
    ):
        p_alt = (V7_CFG["wa"] * softmax_np(aux_ir, T) + V7_CFG["wb"] * softmax_np(th, T) + V7_CFG["wc"] * softmax_np(mid, T))
        for thr in (0.40, 0.50, 0.60):
            for maxch in (5, 10, 15):
                conf = p_v7.max(1)
                pv, pa = p_v7.argmax(1), p_alt.argmax(1)
                cand = np.where(mask0 & (pv != pa) & (conf <= thr))[0]
                order = cand[np.argsort(conf[cand])][:maxch]
                pred = pv.copy(); pred[order] = pa[order]
                full = float((pred[mask0] == yt[mask0]).mean())
                folds = [float((pred[mask0 & (yu == leave)] == yt[mask0 & (yu == leave)]).mean())
                         for leave in (8, 9, 24) if (mask0 & (yu == leave)).sum() >= 5]
                nested = float(np.mean(folds)) if folds else 0.0
                dis = int(((pred[mask0] != v7_preds[mask0]) & (v7_preds[mask0] >= 0)).sum())
                rows.append({"ir": f"fcgate_{aux_name}_t{thr}_m{maxch}", "path": "fcgate", "full": full, "nested": nested,
                             "dis": dis, "solo": float((aux_ir.argmax(1) == yt).mean()), "delta": nested - v7_nest,
                             "cfg": dict(V7_CFG), "nswaps": len(order)})

    clears = sorted([r for r in rows if r["nested"] >= NESTED_MIN - 1e-12 and r["dis"] <= MAX_DIS],
                    key=lambda r: (r["nested"], -r["dis"], r["full"]), reverse=True)
    small = sorted([r for r in rows if r["dis"] <= MAX_DIS], key=lambda r: (r["nested"], r["full"]), reverse=True)
    print("n_rows", len(rows), "n_clear", len(clears), flush=True)
    for r in clears[:15]:
        print(f"CLEAR {r['path']:6s} {r['ir'][:50]:50s} nest={r['nested']:.5f} full={r['full']:.5f} d={r['dis']:2d} delta={r['delta']:+.5f}", flush=True)
    for r in small[:15]:
        print(f"SMALL {r['path']:6s} {r['ir'][:50]:50s} nest={r['nested']:.5f} full={r['full']:.5f} d={r['dis']:2d} delta={r['delta']:+.5f}", flush=True)

    seed_accs = {}
    for sid, views in tta_by_seed.items():
        seed_accs[str(sid)] = {vk: float((views[vk].argmax(1) == yt).mean()) for vk in ("base", "bf", "bft", "b_f_tr")}

    out = {
        "tag": "ir_v28_t24_tta",
        "outcome": "KEEP_V7_NO_SUBMIT" if not clears else "SAFE_CANDIDATE",
        "keep_ir_v7": len(clears) == 0,
        "wrote_csv": False,
        "public_submit": None,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154},
        "strongb_final": {"best": 0.663366, "ep": 6, "early_stop": True, "note": "MISS <<0.69"},
        "v7_reproduce": {"full": float(v7_full), "nested_fixed": float(v7_nest)},
        "gate": {"nested_fixed_min": NESTED_MIN, "disagree_max": MAX_DIS, "mid": "classic_aligned_mid"},
        "tta_seed_hold": seed_accs,
        "n_clear": len(clears),
        "clears": clears[:20],
        "top_small_dis": small[:20],
        "elapsed_s": round(time.time() - t0, 1),
        "updated_at": now_pt(),
        "next_roi": [
            "No SAFE clear from T24 TTA near-v7 — keep ir_v7",
            "Avoid large-disagree fuses (v26 lesson)",
            "Yield GPU / consider new IR recipe only if capacity; HOTC/LMT priority",
        ] if not clears else [
            "SAFE candidate found — write CSV only after recheck gate vs v7",
            "Submit via kaggle.exe only if local looks better without big swap",
        ],
    }
    (ROOT / "metrics_ir_v28_status.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # also refresh v27 status with strongb final
    v27 = {
        "tag": "ir_v27_progress",
        "outcome": "KEEP_V7_STRONGB_MISS",
        "keep_ir_v7": True,
        "wrote_csv": False,
        "csv": None,
        "public_submit": None,
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "note": "ir_v26 public 0.67164 WORSE"},
        "track_submission_restored_to_v7": True,
        "promote_gate": {"nested_fixed_min": 0.75196, "disagree_vs_v7_max": 15, "mid": "classic_aligned_mid"},
        "v27b_safe_fuse_classic_mid": {
            "v7_reproduce": {"full": 0.753036, "nested_fixed": 0.751962},
            "n_clear": 0,
            "best_near": {"nested_fixed": 0.75011, "dis": 6, "delta": -0.00185},
        },
        "strong_t24": {
            "warmstart_stopped_best": 0.687,
            "mildaug_scratch_stopped_best": 0.6713,
            "strongb_final": {
                "best": 0.663366,
                "best_ep": 6,
                "early_stop_ep": 18,
                "ckpt": "checkpoints/ir_yolo_r2p1d18_focal_ft_t24_strongb",
                "log": "logs/train_t24_strongb_s42.log",
                "prior_v24_s42": 0.69703,
                "verdict": "MISS_below_0.69",
            },
        },
        "v28_followup": "probe_ir_v28_t24_tta.py running/done — see metrics_ir_v28_status.json",
        "updated_at": now_pt(),
    }
    (ROOT / "metrics_ir_v27_status.json").write_text(json.dumps(v27, indent=2), encoding="utf-8")
    print("wrote metrics_ir_v28_status.json / metrics_ir_v27_status.json", flush=True)
    print("DONE clears", len(clears), "elapsed", round(time.time()-t0,1), flush=True)

if __name__ == "__main__":
    main()

