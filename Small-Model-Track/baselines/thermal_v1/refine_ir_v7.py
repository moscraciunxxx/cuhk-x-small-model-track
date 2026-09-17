"""IR v7 quick refine: finer triple fuse grid + nested LOUO; optional write submission_ir_v7.csv."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
HOLD = set(DEFAULT_HOLD_OUT_USERS)
V6 = 0.7469635627530364
CLEAR = 0.002  # need >= V6+CLEAR for promote
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def softmax_np(z, T=1.0):
    z = z / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    outs, ys = [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        outs.append(model(x).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


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
    """Coarser weight grid, independent T — report only; often overfits."""
    best = (-1.0, None)
    yt = y[mask]
    pas = {T: softmax_np(a[mask], T) for T in Ts}
    pbs = {T: softmax_np(b[mask], T) for T in Ts}
    pcs = {T: softmax_np(c[mask], T) for T in Ts}
    for Ta in Ts:
        for Tb in Ts:
            for Tc in Ts:
                pa, pb, pc = pas[Ta], pbs[Tb], pcs[Tc]
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
    best = (-1.0, None)
    yt = y[mask]
    eps = 1e-8
    for T in Ts:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                # weighted geometric mean in prob space
                g = np.exp(wa * np.log(pa + eps) + wb * np.log(pb + eps) + wc * np.log(pc + eps))
                g = g / g.sum(1, keepdims=True)
                acc = float((g.argmax(1) == yt).mean())
                if acc > best[0] + 1e-12:
                    best = (acc, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                  "acc": acc, "n": int(mask.sum()), "mode": "geom"})
    return best


def nested_louo(a, b, c, y, users, mask, Ts, ngrid=26):
    hold_users = sorted(set(users[mask].tolist()))
    folds = []
    for leave in hold_users:
        tr = mask & (users != leave)
        te = mask & (users == leave)
        if te.sum() < 5 or tr.sum() < 20:
            continue
        acc, cfg = fuse3_sameT(a, b, c, y, tr, Ts, ngrid=ngrid)
        # apply cfg on te
        T = cfg["T"]
        pa = softmax_np(a[te], T); pb = softmax_np(b[te], T); pc = softmax_np(c[te], T)
        te_acc = float(((cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1) == y[te]).mean())
        folds.append({"leave": int(leave), "te_acc": te_acc, "tr_acc": acc, **{k: cfg[k] for k in ("wa","wb","wc","T")}})
    mean_te = float(np.mean([f["te_acc"] for f in folds])) if folds else -1.0
    # mean cfg applied to full mask
    if folds:
        wa = float(np.mean([f["wa"] for f in folds]))
        wb = float(np.mean([f["wb"] for f in folds]))
        wc = float(np.mean([f["wc"] for f in folds]))
        # renormalize
        s = wa + wb + wc; wa, wb, wc = wa/s, wb/s, wc/s
        T = float(np.mean([f["T"] for f in folds]))
        pa = softmax_np(a[mask], T); pb = softmax_np(b[mask], T); pc = softmax_np(c[mask], T)
        full = float(((wa * pa + wb * pb + wc * pc).argmax(1) == y[mask]).mean())
        mean_cfg = {"wa": wa, "wb": wb, "wc": wc, "T": T, "full_acc": full}
    else:
        mean_cfg = None
    return mean_te, folds, mean_cfg


def write_sub(path, meta, preds, empty, fb):
    nfb = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i])); nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / "cache" / "ir_yolo_v4"
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)

    z = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_logits_v6.npz", allow_pickle=True)
    old_tags = [str(t) for t in z["tags"]]
    old_base = z["base"]; yt = z["y"]; yu = z["users"]
    assert np.allclose(yt, y[hold_idx])

    members = []
    old_ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    for i, t in enumerate(old_tags):
        acc = float((old_base[i].argmax(1) == yt).mean())
        tl_path = old_ckpt / f"test_logits_{t}.npy"
        if not tl_path.exists():
            # try alternate naming
            seed = t.replace("pool_seed", "")
            tl_path = old_ckpt / f"test_logits_seed{seed}.npy"
        members.append({"tag": t, "logits": old_base[i], "acc": acc, "test_logits": np.load(tl_path)})

    # new seeds hold eval (cache if present)
    new_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6"
    hold_cache = new_dir / "hold_logits_new_seeds.npz"
    new_hold = {}
    if hold_cache.exists():
        hz = np.load(hold_cache, allow_pickle=True)
        for t in hz["tags"]:
            new_hold[str(t)] = hz[str(t)]
        print(f"loaded cached new hold logits: {list(new_hold)}", flush=True)

    for p in sorted(new_dir.glob("pool_seed*.pt")):
        tag = p.stem
        if tag in new_hold:
            lg = new_hold[tag]
        else:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            model = build().to(device); model.load_state_dict(blob["model"])
            lg, yt2 = eval_logits(model, loader, device)
            assert np.allclose(yt2, yt)
            del model; torch.cuda.empty_cache()
            new_hold[tag] = lg
            print(f"eval {tag} hold={float((lg.argmax(1)==yt).mean()):.4f}", flush=True)
        acc = float((lg.argmax(1) == yt).mean())
        seed = tag.replace("pool_seed", "")
        tl = np.load(new_dir / f"test_logits_seed{seed}.npy")
        members.append({"tag": tag, "logits": lg, "acc": acc, "test_logits": tl})

    # save new hold cache
    np.savez_compressed(hold_cache, tags=np.array(list(new_hold.keys())), **{k: v for k, v in new_hold.items()})

    members = sorted(members, key=lambda d: -d["acc"])
    print("members:", [(m["tag"], round(m["acc"], 4)) for m in members], flush=True)

    th = np.load(old_ckpt / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")[hold_idx]
    mask = th.any(1) & mid.any(1)
    print(f"mask n={mask.sum()} / {len(yt)}", flush=True)

    Ts_fine = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]
    Ts_med = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
    Ts_per = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = []
    # build IR ens variants
    stack = np.stack([m["logits"] for m in members], 0)
    w_acc = np.array([max(m["acc"], 1e-3) for m in members], dtype=np.float64); w_acc /= w_acc.sum()
    variants = {}
    for k in range(3, len(members) + 1):
        variants[f"top{k}"] = np.mean(stack[:k], 0)
    variants["all_mean"] = np.mean(stack, 0)
    variants["all_acc_w"] = np.tensordot(w_acc, stack, axes=(0, 0))
    # power mean of softmax @T=1 then back to logit-ish: use mean of softmax as ens
    sm = np.stack([softmax_np(m["logits"], 1.0) for m in members], 0)
    variants["sm_mean"] = np.log(np.mean(sm, 0) + 1e-8)  # logprob as pseudo-logit
    variants["sm_acc_w"] = np.log(np.tensordot(w_acc, sm, axes=(0, 0)) + 1e-8)

    for name, elogs in variants.items():
        ens_acc = float((elogs.argmax(1) == yt).mean()) if name not in ("sm_mean", "sm_acc_w") else float((np.exp(elogs).argmax(1) == yt).mean())
        # for sm_* elogs are logprobs; fuse still works as 'logits' with T scaling
        b_acc, bcfg = fuse3_sameT(elogs, th, mid, yt, mask, Ts_fine, ngrid=51)
        g_acc, gcfg = fuse3_geom(elogs, th, mid, yt, mask, Ts_med, ngrid=41)
        results.append({"ens": name, "ens_acc": ens_acc, "triple": bcfg, "geom": gcfg,
                        "triple_acc": b_acc, "geom_acc": g_acc})
        print(f"{name}: ens={ens_acc:.4f} triple={b_acc:.4f} {bcfg} | geom={g_acc:.4f}", flush=True)

    best = max(results, key=lambda r: max(r["triple_acc"], r["geom_acc"]))
    use_geom = best["geom_acc"] > best["triple_acc"] + 1e-12
    best_acc = best["geom_acc"] if use_geom else best["triple_acc"]
    best_cfg = best["geom"] if use_geom else best["triple"]
    print(f"\nBEST full-fit: {best['ens']} {'geom' if use_geom else 'triple'}={best_acc:.6f} delta_v6={best_acc-V6:+.4f}", flush=True)

    # pick IR ens for nested (prefer all_mean / top9 which was v6 primary)
    elogs_primary = variants.get(best["ens"], variants["all_mean"])
    nest_mean, nest_folds, nest_cfg = nested_louo(elogs_primary, th, mid, yt, yu, mask, Ts_med, ngrid=26)
    print(f"nested LOUO mean={nest_mean:.4f} folds={nest_folds} mean_cfg={nest_cfg}", flush=True)

    # also nested for all_mean specifically
    nest_mean2, nest_folds2, nest_cfg2 = nested_louo(variants["all_mean"], th, mid, yt, yu, mask, Ts_med, ngrid=26)
    print(f"all_mean nested={nest_mean2:.4f} cfg={nest_cfg2}", flush=True)

    # perT on best ens (diagnostic)
    p_acc, pcfg = fuse3_perT(elogs_primary, th, mid, yt, mask, Ts_per, ngrid=17)
    print(f"perT diagnostic={p_acc:.4f} {pcfg}", flush=True)

    # reproduce v6 cfg
    v6_cfg = {"wa": 0.56, "wb": 0.36, "wc": 0.08, "T": 1.5}
    T = v6_cfg["T"]
    pa = softmax_np(variants["all_mean"][mask], T); pb = softmax_np(th[mask], T); pc = softmax_np(mid[mask], T)
    v6_re = float(((v6_cfg["wa"]*pa + v6_cfg["wb"]*pb + v6_cfg["wc"]*pc).argmax(1) == yt[mask]).mean())
    print(f"v6 cfg reproduce on all_mean: {v6_re:.6f}", flush=True)

    report = {
        "tag": "ir_v7_refine",
        "members": {m["tag"]: m["acc"] for m in members},
        "results": [{k: (v if k != "triple" and k != "geom" else v) for k, v in r.items()} for r in results],
        "best_ens": best["ens"],
        "best_mode": "geom" if use_geom else "sameT",
        "best_acc": best_acc,
        "best_cfg": best_cfg,
        "delta_vs_v6": float(best_acc - V6),
        "v6": V6,
        "v6_reproduce": v6_re,
        "nested_best_ens": {"mean": nest_mean, "folds": nest_folds, "cfg": nest_cfg},
        "nested_all_mean": {"mean": nest_mean2, "folds": nest_folds2, "cfg": nest_cfg2},
        "perT": pcfg,
        "elapsed_s": time.time() - t0,
    }

    clear_win = best_acc >= V6 + CLEAR - 1e-9
    # Prefer nested-mean cfg if it also improves full-fit vs v6
    promote_cfg = None
    promote_mode = None
    promote_ens = None
    promote_acc = None
    if clear_win and best_cfg["mode"] != "perT":
        # require nested not worse than v6 nested (~0.7247)
        if nest_mean >= 0.72 or nest_mean2 >= 0.72:
            promote_cfg = best_cfg
            promote_mode = best_cfg["mode"]
            promote_ens = best["ens"]
            promote_acc = best_acc

    # Also consider nested_mean_cfg if full_acc clearly > v6
    for label, nc in [("nested_best", nest_cfg), ("nested_all", nest_cfg2)]:
        if nc and nc["full_acc"] >= V6 + CLEAR - 1e-9:
            if promote_acc is None or nc["full_acc"] > promote_acc:
                promote_cfg = {**nc, "mode": "sameT", "acc": nc["full_acc"], "n": int(mask.sum())}
                promote_mode = "nested_mean_cfg"
                promote_ens = "all_mean" if label == "nested_all" else best["ens"]
                promote_acc = nc["full_acc"]

    report["clear_win"] = bool(clear_win)
    report["promote"] = None

    if promote_cfg is not None and promote_acc >= V6 + CLEAR - 1e-9:
        # build IR test ens matching promote_ens
        if promote_ens.startswith("top"):
            k = int(promote_ens[3:])
            sel = members[:k]
            ir_test = np.mean([m["test_logits"] for m in sel], 0).astype(np.float32)
            tags = [m["tag"] for m in sel]
        elif promote_ens == "all_acc_w":
            ir_test = np.tensordot(w_acc, np.stack([m["test_logits"] for m in members], 0), axes=(0, 0)).astype(np.float32)
            tags = [m["tag"] for m in members]
        elif promote_ens in ("sm_mean", "sm_acc_w"):
            tls = np.stack([m["test_logits"] for m in members], 0)
            sms = np.stack([softmax_np(t, 1.0) for t in tls], 0)
            if promote_ens == "sm_mean":
                ir_test = np.log(np.mean(sms, 0) + 1e-8).astype(np.float32)
            else:
                ir_test = np.log(np.tensordot(w_acc, sms, axes=(0, 0)) + 1e-8).astype(np.float32)
            tags = [m["tag"] for m in members]
        else:
            ir_test = np.mean([m["test_logits"] for m in members], 0).astype(np.float32)
            tags = [m["tag"] for m in members]

        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
        if not th_p.exists():
            th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
        th_test = np.load(th_p)
        cfg = promote_cfg
        if promote_mode == "geom" or cfg.get("mode") == "geom":
            eps = 1e-8
            T = cfg["T"]
            pa, pb, pc = softmax_np(ir_test, T), softmax_np(th_test, T), softmax_np(mid_test, T)
            g = np.exp(cfg["wa"]*np.log(pa+eps) + cfg["wb"]*np.log(pb+eps) + cfg["wc"]*np.log(pc+eps))
            preds = (g / g.sum(1, keepdims=True)).argmax(1)
        else:
            T = cfg["T"]
            preds = (cfg["wa"]*softmax_np(ir_test, T) + cfg["wb"]*softmax_np(th_test, T) + cfg["wc"]*softmax_np(mid_test, T)).argmax(1)

        meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
        empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
        fb = {}
        with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
            for row in csv.DictReader(f):
                fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
        out = ROOT / "submission_ir_v7.csv"
        nfb = write_sub(out, meta, preds, empty, fb)
        # promote track submission
        track_sub = TRACK / "submission.csv"
        track_sub.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        report["promote"] = {
            "wrote": str(out),
            "promoted_track": str(track_sub),
            "hold": promote_acc,
            "cfg": promote_cfg,
            "ens": promote_ens,
            "mode": promote_mode,
            "tags": tags,
            "empty_fallback": nfb,
        }
        print(f"PROMOTED v7 hold={promote_acc:.4f} -> {out} + track submission.csv", flush=True)
    else:
        print(f"NO CLEAR WIN (best={best_acc:.4f} v6={V6:.4f} need>={V6+CLEAR:.4f}); leave v6 primary", flush=True)

    (ROOT / "metrics_ir_v7_refine.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"best_acc": best_acc, "delta": best_acc - V6, "clear_win": clear_win,
                      "promoted": report["promote"] is not None, "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
