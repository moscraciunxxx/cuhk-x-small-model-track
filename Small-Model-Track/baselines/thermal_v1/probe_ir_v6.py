"""v6 probe: selective TTA, seed ablation, nested LOUO fuse, finer grids. Cache hold logits then CPU search."""
from __future__ import annotations
import csv, json, argparse
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
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)
V5 = 0.7388663967611336


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
def eval_logits(model, loader, device, tta=False):
    model.eval()
    outs, ys, us = [], [], []
    for x, y, u, _i in loader:
        x = x.to(device)
        def _fwd(xx):
            xx = normalize(xx).permute(0, 2, 1, 3, 4).contiguous()
            return model(xx).float()
        logits = _fwd(x)
        if tta:
            logits = 0.5 * (logits + _fwd(torch.flip(x, dims=[-1])))
        outs.append(logits.cpu().numpy())
        ys.append(y.numpy())
        us.append(u.numpy())
    return np.concatenate(outs), np.concatenate(ys), np.concatenate(us)


@torch.no_grad()
def infer_array(model, Xt, device, bs=8, tta=False):
    model.eval()
    n = len(Xt)
    o = np.zeros((n, NUM_CLASSES), np.float32)
    for i in range(0, n, bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        def _fwd(xx):
            xx = xx.permute(0, 1, 4, 2, 3).contiguous()
            xx = normalize(xx).permute(0, 2, 1, 3, 4).contiguous()
            return model(xx).float()
        logits = _fwd(x)
        if tta:
            logits = 0.5 * (logits + _fwd(torch.flip(x, dims=[-2])))
        o[i:i + len(x)] = logits.cpu().numpy()
    return o


def acc(logits, y):
    return float((logits.argmax(1) == y).mean())


def fuse2(a, b, y, mask, Ts=None, ws=None):
    best = (-1.0, None)
    yt = y[mask]
    Ts = Ts or [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    ws = ws if ws is not None else np.linspace(0, 1, 51)
    for T in Ts:
        pa, pb = softmax_np(a[mask], T), softmax_np(b[mask], T)
        for w in ws:
            a_ = float(((w * pa + (1 - w) * pb).argmax(1) == yt).mean())
            if a_ > best[0]:
                best = (a_, {"w": float(w), "T": float(T), "acc": a_, "n": int(mask.sum())})
    return best


def fuse3(a, b, c, y, mask, Ts=None, ngrid=26):
    best = (-1.0, None)
    yt = y[mask]
    Ts = Ts or [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
    for T in Ts:
        pa, pb, pc = softmax_np(a[mask], T), softmax_np(b[mask], T), softmax_np(c[mask], T)
        for wa in np.linspace(0, 1, ngrid):
            for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                a_ = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                if a_ > best[0]:
                    best = (a_, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T),
                                 "acc": a_, "n": int(mask.sum())})
    return best


def fuse3_perT(a, b, c, y, mask, ngrid=21):
    """Independent temperatures per stream."""
    best = (-1.0, None)
    yt = y[mask]
    Ts = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
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
                        a_ = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                        if a_ > best[0]:
                            best = (a_, {"wa": float(wa), "wb": float(wb), "wc": float(wc),
                                         "Ta": float(Ta), "Tb": float(Tb), "Tc": float(Tc),
                                         "acc": a_, "n": int(mask.sum())})
    return best


def fuse3_nested_louo(a, b, c, y, users, mask, ngrid=21):
    """Tune on leave-one-holdout-user-out, report mean held-out user acc + full-fit cfg."""
    u_hold = users[mask]
    y_m = y[mask]
    a_m, b_m, c_m = a[mask], b[mask], c[mask]
    uniq = sorted(set(int(u) for u in u_hold))
    fold_accs = []
    fold_cfgs = []
    for leave in uniq:
        tr = u_hold != leave
        te = u_hold == leave
        if te.sum() < 10 or tr.sum() < 10:
            continue
        # index into masked arrays
        # build fake full-length with only train mask for fuse3 API
        # simpler: run grid on train directly
        best = (-1.0, None)
        yt = y_m[tr]
        for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
            pa, pb, pc = softmax_np(a_m[tr], T), softmax_np(b_m[tr], T), softmax_np(c_m[tr], T)
            for wa in np.linspace(0, 1, ngrid):
                for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * (ngrid - 1))) + 1)):
                    wc = 1.0 - wa - wb
                    if wc < -1e-9:
                        continue
                    ac = float(((wa * pa + wb * pb + wc * pc).argmax(1) == yt).mean())
                    if ac > best[0]:
                        best = (ac, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T)})
        cfg = best[1]
        pa, pb, pc = softmax_np(a_m[te], cfg["T"]), softmax_np(b_m[te], cfg["T"]), softmax_np(c_m[te], cfg["T"])
        te_acc = float(((cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc).argmax(1) == y_m[te]).mean())
        fold_accs.append(te_acc)
        fold_cfgs.append({**cfg, "leave": int(leave), "te_acc": te_acc, "tr_acc": best[0]})
    # also full-fit on all masked for deployment cfg
    full_acc, full_cfg = fuse3(a, b, c, y, mask, ngrid=ngrid)
    return {
        "nested_mean": float(np.mean(fold_accs)) if fold_accs else None,
        "nested_folds": fold_cfgs,
        "full_fit_acc": full_acc,
        "full_fit_cfg": full_cfg,
        # mean weights from nested folds (honest)
        "nested_mean_cfg": {
            "wa": float(np.mean([f["wa"] for f in fold_cfgs])),
            "wb": float(np.mean([f["wb"] for f in fold_cfgs])),
            "wc": float(np.mean([f["wc"] for f in fold_cfgs])),
            "T": float(np.mean([f["T"] for f in fold_cfgs])),
        } if fold_cfgs else None,
    }


def apply_cfg3(a, b, c, cfg):
    if "Ta" in cfg:
        pa, pb, pc = softmax_np(a, cfg["Ta"]), softmax_np(b, cfg["Tb"]), softmax_np(c, cfg["Tc"])
    else:
        T = cfg["T"]
        pa, pb, pc = softmax_np(a, T), softmax_np(b, T), softmax_np(c, T)
    return cfg["wa"] * pa + cfg["wb"] * pb + cfg["wc"] * pc


def write_sub(path, meta, preds, empty, fb):
    nfb = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i])); nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"))
    ap.add_argument("--write", action="store_true", help="write submission_ir_v6 if clear win")
    ap.add_argument("--min-delta", type=float, default=0.002, help="clear beat threshold vs v5")
    args = ap.parse_args()
    cache, ckpt_dir = Path(args.cache_dir), Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy")
    users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(y), 16, 112, 112, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False), batch_size=12, shuffle=False)
    logit_cache = ckpt_dir / "hold_logits_v6.npz"

    members = []
    if logit_cache.exists():
        z = np.load(logit_cache, allow_pickle=True)
        tags = list(z["tags"])
        yt = z["y"]; u_hold = z["users"]
        for i, tag in enumerate(tags):
            members.append({
                "tag": str(tag),
                "base": z["base"][i],
                "tta": z["tta"][i],
                "acc_base": float(z["acc_base"][i]),
                "acc_tta": float(z["acc_tta"][i]),
            })
            print(f"cached {tag} base={members[-1]['acc_base']:.4f} tta={members[-1]['acc_tta']:.4f}", flush=True)
    else:
        yt = u_hold = None
        base_list, tta_list, tags, ab, at = [], [], [], [], []
        for p in sorted(ckpt_dir.glob("pool_seed*.pt")):
            blob = torch.load(p, map_location="cpu", weights_only=False)
            model = build().to(device)
            model.load_state_dict(blob["model"])
            lb, yt, u_hold = eval_logits(model, loader, device, tta=False)
            lt, _, _ = eval_logits(model, loader, device, tta=True)
            ab_ = acc(lb, yt); at_ = acc(lt, yt)
            print(f"{p.stem} base={ab_:.4f} tta={at_:.4f} d={at_-ab_:+.4f}", flush=True)
            members.append({"tag": p.stem, "base": lb, "tta": lt, "acc_base": ab_, "acc_tta": at_, "state": blob["model"]})
            base_list.append(lb); tta_list.append(lt); tags.append(p.stem); ab.append(ab_); at.append(at_)
            del model; torch.cuda.empty_cache()
        np.savez_compressed(logit_cache, base=np.stack(base_list), tta=np.stack(tta_list),
                            tags=np.array(tags), acc_base=np.array(ab), acc_tta=np.array(at),
                            y=yt, users=u_hold)
        print(f"saved {logit_cache}", flush=True)

    # choose per-seed best of base/tta
    chosen = []
    for m in members:
        if m["acc_tta"] > m["acc_base"] + 1e-9:
            chosen.append(m["tta"]); mode = "tta"
        else:
            chosen.append(m["base"]); mode = "base"
        print(f"select {m['tag']} -> {mode}", flush=True)
    stack_sel = np.stack(chosen, 0)
    stack_base = np.stack([m["base"] for m in members], 0)
    ens_base = stack_base.mean(0)
    ens_sel = stack_sel.mean(0)
    # also top-k by base acc
    order = np.argsort([-m["acc_base"] for m in members])
    results = {
        "ens_base": acc(ens_base, yt),
        "ens_selective_tta": acc(ens_sel, yt),
    }
    for k in [3, 4, 5, 6]:
        if k > len(members):
            continue
        idx = order[:k]
        results[f"top{k}_base"] = acc(stack_base[idx].mean(0), yt)
        results[f"top{k}_sel"] = acc(stack_sel[idx].mean(0), yt)
    # acc-weighted
    w = np.array([max(m["acc_base"], 1e-3) for m in members], dtype=np.float64); w /= w.sum()
    results["ens_acc_w_base"] = acc(np.tensordot(w, stack_base, axes=(0, 0)), yt)
    w2 = np.array([max(max(m["acc_base"], m["acc_tta"]), 1e-3) for m in members], dtype=np.float64); w2 /= w2.sum()
    results["ens_acc_w_sel"] = acc(np.tensordot(w2, stack_sel, axes=(0, 0)), yt)
    # softmax mean
    results["ens_sm_base"] = float(np.mean([softmax_np(m["base"]) for m in members], 0).argmax(1).__eq__(yt).mean()) if False else float((np.mean([softmax_np(x) for x in stack_base], 0).argmax(1) == yt).mean())
    results["ens_sm_sel"] = float((np.mean([softmax_np(x) for x in stack_sel], 0).argmax(1) == yt).mean())
    print("IR ens variants:", json.dumps(results, indent=2), flush=True)

    best_ens_name = max(results, key=results.get)
    # pick actual logits for best
    ens_map = {
        "ens_base": ens_base,
        "ens_selective_tta": ens_sel,
        "ens_acc_w_base": np.tensordot(w, stack_base, axes=(0, 0)),
        "ens_acc_w_sel": np.tensordot(w2, stack_sel, axes=(0, 0)),
        "ens_sm_base": np.log(np.clip(np.mean([softmax_np(x) for x in stack_base], 0), 1e-8, 1)),
        "ens_sm_sel": np.log(np.clip(np.mean([softmax_np(x) for x in stack_sel], 0), 1e-8, 1)),
    }
    for k in list(results):
        if k.startswith("top"):
            kk, kind = k.split("_", 1)
            n = int(kk[3:])
            idx = order[:n]
            ens_map[k] = (stack_sel if kind == "sel" else stack_base)[idx].mean(0)
    ens = ens_map[best_ens_name]
    print(f"BEST IR ens={best_ens_name} {results[best_ens_name]:.4f}", flush=True)

    # mid + thermal
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")[hold_idx]
    mask = mid.any(1)
    ir_meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    th_cache = ROOT / "cache" / "thermal_yolo"
    th_y = np.load(th_cache / "train_y.npy"); th_u = np.load(th_cache / "train_users.npy")
    th_hold = np.where(np.isin(th_u, list(HOLD)))[0]
    th_meta = json.loads((th_cache / "train_meta.json").read_text(encoding="utf-8"))
    key = lambda m: (int(m["user_id"]), int(m["label"]), str(m.get("trial", "")), str(m.get("action_name", "")))
    th_paths = [
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18" / "holdout_train.pt",
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "exact_seed123.pt",
        ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v2" / "exact_seed7.pt",
    ]
    th_X = np.memmap(th_cache / "train_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(th_y), 16, 112, 112, 3))
    th_loader = DataLoader(CachedClipDataset(th_X, th_y, th_u, th_hold, train=False), batch_size=12, shuffle=False)
    th_cache_p = ckpt_dir / "hold_thermal_v6.npy"
    if th_cache_p.exists():
        th_ens = np.load(th_cache_p)
        print("loaded thermal hold cache", flush=True)
    else:
        th_logs = []
        for p in th_paths:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            model = build().to(device); model.load_state_dict(blob["model"])
            lg, _, _ = eval_logits(model, th_loader, device, tta=False)
            th_logs.append(lg); del model; torch.cuda.empty_cache()
            print("thermal", p.name, flush=True)
        th_ens_raw = np.mean(th_logs, 0)
        th_hold_meta = [th_meta[i] for i in th_hold]
        th_map = {key(m): j for j, m in enumerate(th_hold_meta)}
        th_ens = np.zeros_like(ens)
        for j, ii in enumerate(hold_idx):
            k = key(ir_meta[ii])
            if k in th_map:
                th_ens[j] = th_ens_raw[th_map[k]]
        np.save(th_cache_p, th_ens)

    th_mask = th_ens.any(1) & mask

    # Try several IR ens candidates for fuse
    fuse_report = {}
    best_overall = (-1.0, None, None, None)  # acc, name, cfg, ens_logits
    for ename, elogs in [
        ("ens_base", ens_base),
        ("ens_selective_tta", ens_sel),
        ("top4_base", ens_map.get("top4_base")),
        ("top5_sel", ens_map.get("top5_sel")),
        ("best_ir", ens),
    ]:
        if elogs is None:
            continue
        b3_acc, b3 = fuse3(elogs, th_ens, mid, yt, th_mask, ngrid=26)
        b2_acc, b2 = fuse2(elogs, th_ens, yt, th_ens.any(1))
        bm_acc, bm = fuse2(elogs, mid, yt, mask)
        print(f"FUSE[{ename}] triple={b3_acc:.4f} ir_th={b2_acc:.4f} ir_mid={bm_acc:.4f} cfg3={b3}", flush=True)
        fuse_report[ename] = {"triple": b3, "ir_th": b2, "ir_mid": bm}
        for nm, ac, cfg in [("triple", b3_acc, b3), ("ir_th", b2_acc, b2), ("ir_mid", bm_acc, bm)]:
            if ac > best_overall[0]:
                best_overall = (ac, f"{ename}+{nm}", cfg, elogs)

    # per-T and nested on best IR ens so far
    elogs = best_overall[3]
    pt_acc, pt = fuse3_perT(elogs, th_ens, mid, yt, th_mask, ngrid=21)
    print(f"perT triple={pt_acc:.4f} {pt}", flush=True)
    nested = fuse3_nested_louo(elogs, th_ens, mid, yt, u_hold, th_mask, ngrid=21)
    print(f"nested={json.dumps({k:nested[k] for k in nested if k!='nested_folds'}, indent=2)}", flush=True)
    # evaluate nested_mean_cfg on full holdout
    if nested["nested_mean_cfg"]:
        pred = apply_cfg3(elogs[th_mask], th_ens[th_mask], mid[th_mask], nested["nested_mean_cfg"]).argmax(1)
        nested_full = float((pred == yt[th_mask]).mean())
        print(f"nested_mean_cfg full-hold acc={nested_full:.4f}", flush=True)
        nested["nested_mean_cfg_full_acc"] = nested_full
        if nested_full > best_overall[0]:
            best_overall = (nested_full, best_overall[1].split("+")[0] + "+nested_mean", nested["nested_mean_cfg"], elogs)
    if pt_acc > best_overall[0]:
        best_overall = (pt_acc, best_overall[1].split("+")[0] + "+perT", pt, elogs)

    # power-mean style: average of log-probs = geometric
    def geom_fuse(a, b, c, y, mask):
        best = (-1.0, None)
        yt_ = y[mask]
        for T in [1.0, 1.5, 2.0, 2.5, 3.0]:
            la, lb, lc = np.log(np.clip(softmax_np(a[mask], T), 1e-8, 1)), np.log(np.clip(softmax_np(b[mask], T), 1e-8, 1)), np.log(np.clip(softmax_np(c[mask], T), 1e-8, 1))
            for wa in np.linspace(0, 1, 21):
                for wb in np.linspace(0, 1 - wa, max(1, int(round((1 - wa) * 20)) + 1)):
                    wc = 1.0 - wa - wb
                    if wc < -1e-9: continue
                    pred = (wa * la + wb * lb + wc * lc).argmax(1)
                    ac = float((pred == yt_).mean())
                    if ac > best[0]:
                        best = (ac, {"wa": float(wa), "wb": float(wb), "wc": float(wc), "T": float(T), "acc": ac, "mode": "geom"})
        return best
    g_acc, gcfg = geom_fuse(elogs, th_ens, mid, yt, th_mask)
    print(f"geom triple={g_acc:.4f} {gcfg}", flush=True)
    if g_acc > best_overall[0]:
        best_overall = (g_acc, best_overall[1].split("+")[0] + "+geom", gcfg, elogs)

    print(f"\nBEST_OVERALL {best_overall[1]} acc={best_overall[0]:.6f} cfg={best_overall[2]}", flush=True)
    print(f"delta vs v5 {best_overall[0] - V5:+.6f} (need >= +{args.min_delta})", flush=True)

    report = {
        "ir_ens_variants": results,
        "best_ir_ens": best_ens_name,
        "fuse_report": {k: {kk: vv for kk, vv in v.items()} for k, v in fuse_report.items()},
        "perT": pt,
        "nested": {k: nested[k] for k in nested if k != "nested_folds"},
        "nested_folds": nested.get("nested_folds"),
        "geom": gcfg,
        "best_name": best_overall[1],
        "best_acc": best_overall[0],
        "best_cfg": best_overall[2],
        "delta_vs_v5": float(best_overall[0] - V5),
        "clear_win": bool(best_overall[0] >= V5 + args.min_delta),
        "v5": V5,
    }
    (ROOT / "metrics_ir_v6_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not (args.write and report["clear_win"]):
        print("NO WRITE (no clear win or --write not set). v5 remains primary.", flush=True)
        print(json.dumps({k: report[k] for k in ["best_name","best_acc","delta_vs_v5","clear_win"]}, indent=2))
        return

    # ---- write submission ----
    print("CLEAR WIN -> writing submission_ir_v6.csv", flush=True)
    # reload states for test infer
    members_state = []
    for p in sorted(ckpt_dir.glob("pool_seed*.pt")):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        tag = p.stem
        m = next(x for x in members if x["tag"] == tag)
        use_tta = m["acc_tta"] > m["acc_base"] + 1e-9
        members_state.append({"tag": tag, "state": blob["model"], "tta": use_tta, "acc": max(m["acc_base"], m["acc_tta"])})

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    Xt = np.memmap(cache / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta), 16, 112, 112, 3))
    outs = []
    for m in members_state:
        model = build().to(device); model.load_state_dict(m["state"])
        o = infer_array(model, Xt, device, bs=8, tta=m["tta"])
        np.save(ckpt_dir / f"test_logits_v6_{m['tag']}_{'tta' if m['tta'] else 'base'}.npy", o)
        outs.append(o); del model; torch.cuda.empty_cache()
        print(f"test {m['tag']} tta={m['tta']}", flush=True)

    # match ens construction
    ename = best_overall[1].split("+")[0]
    stack_t = np.stack(outs, 0)
    if ename == "ens_selective_tta" or ename == "best_ir":
        # outs already selective
        ir_test = stack_t.mean(0)
    elif ename == "ens_base":
        # need base-only infer — re-infer without tta for fairness
        outs_b = []
        for m in members_state:
            model = build().to(device); model.load_state_dict(m["state"])
            outs_b.append(infer_array(model, Xt, device, bs=8, tta=False))
            del model; torch.cuda.empty_cache()
        ir_test = np.stack(outs_b, 0).mean(0)
    elif ename.startswith("top"):
        n = int(ename[3:].split("_")[0])
        kind = ename.split("_", 1)[1]
        idx = order[:n]
        if kind == "base":
            outs_b = []
            for m in members_state:
                model = build().to(device); model.load_state_dict(m["state"])
                outs_b.append(infer_array(model, Xt, device, bs=8, tta=False))
                del model; torch.cuda.empty_cache()
            ir_test = np.stack(outs_b, 0)[idx].mean(0)
        else:
            ir_test = stack_t[idx].mean(0)
    else:
        ir_test = stack_t.mean(0)

    np.save(ckpt_dir / "test_logits_ens_v6.npy", ir_test.astype(np.float32))
    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_test_p.exists():
        th_test_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_test_p)
    cfg = best_overall[2]
    kind = best_overall[1].split("+", 1)[1]
    if "ir_th" in kind:
        preds = (cfg["w"] * softmax_np(ir_test, cfg["T"]) + (1 - cfg["w"]) * softmax_np(th_test, cfg["T"])).argmax(1)
    elif "ir_mid" in kind:
        preds = (cfg["w"] * softmax_np(ir_test, cfg["T"]) + (1 - cfg["w"]) * softmax_np(mid_test, cfg["T"])).argmax(1)
    elif "geom" in kind:
        T = cfg["T"]
        la, lb, lc = np.log(np.clip(softmax_np(ir_test, T), 1e-8, 1)), np.log(np.clip(softmax_np(th_test, T), 1e-8, 1)), np.log(np.clip(softmax_np(mid_test, T), 1e-8, 1))
        preds = (cfg["wa"] * la + cfg["wb"] * lb + cfg["wc"] * lc).argmax(1)
    else:
        preds = apply_cfg3(ir_test, th_test, mid_test, cfg).argmax(1)

    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])
    out_p = ROOT / "submission_ir_v6.csv"
    nfb = write_sub(out_p, meta, preds, empty, fb)
    report["submission"] = str(out_p)
    report["empty_fallback"] = nfb
    (ROOT / "metrics_ir_v6.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"WROTE {out_p} hold={best_overall[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()

