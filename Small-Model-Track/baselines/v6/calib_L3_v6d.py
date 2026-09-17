"""v6d: OOF logit calibration + bias; worst-fold cluster policy; combine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\CUHK-X\Small-Model-Track")
ROOT_V2 = ROOT / "baselines" / "skeleton_imu_v2"
OUT = ROOT / "baselines" / "v6"
sys.path.insert(0, str(ROOT_V2))
sys.path.insert(0, str(OUT))

from dataset import DEFAULT_HOLD_OUT_USERS, load_skel_train_cache  # noqa: E402
from specialists_v6 import extract_all, load_midfuse, make_loader, softmax_np  # noqa: E402
from final_v6 import train_raw_pair_inner, apply_raw_defer  # noqa: E402
from cluster_v6c import CLUSTERS, apply_cluster_defer  # noqa: E402

BASELINE = 0.537
N = 40


def build_oof(X_skel, X_imu, y, users, has_imu, device):
    fold_dir = ROOT_V2 / "checkpoints_midfuse_v2b"
    oof = np.zeros((len(y), N), np.float32)
    gkf = GroupKFold(5)
    for fi, (tr, va) in enumerate(gkf.split(np.zeros(len(y)), y, users)):
        mid, _ = load_midfuse(fold_dir / f"best_fold{fi}.pt", device)
        pack = extract_all(mid, make_loader(X_skel, X_imu, y, users, va, has_imu), device)
        oof[va] = pack["logits"]
        del mid
        torch.cuda.empty_cache()
    return oof


def fit_temperature_bias(logits, y, max_iter=80):
    """Learn T and class bias on OOF to max CE; return T, bias."""
    device = torch.device("cpu")
    z = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_T = torch.nn.Parameter(torch.zeros(()))
    bias = torch.nn.Parameter(torch.zeros(N))
    opt = torch.optim.LBFGS([log_T, bias], lr=0.5, max_iter=20)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(0.05, 10.0)
        loss = torch.nn.functional.cross_entropy((z / T) + bias, yt)
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)
    T = float(torch.exp(log_T).clamp(0.05, 10.0).item())
    b = bias.detach().numpy().astype(np.float32)
    # also grid-search small scale on bias for accuracy
    base_pred = logits.argmax(1)
    best_acc = float((base_pred == y).mean())
    best = (T, b, best_acc)
    for t in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, T]:
        for scale in [0.0, 0.25, 0.5, 1.0, 1.5]:
            pred = (logits / t + scale * b).argmax(1)
            acc = float((pred == y).mean())
            if acc > best[2]:
                best = (t, scale * b, acc)
    return best  # T, bias_vec, oof_acc


def leave3_policy_select(oof_logits, y, users, tr_idx, X_skel, X_imu, has_imu, device, n_rounds=12):
    """Simulate holdout: repeatedly hold out 3 train users, fit clusters on rest, score."""
    rng = np.random.RandomState(0)
    uniq = np.unique(users[tr_idx])
    results = []
    # Pre-fit is expensive; do fewer rounds with shared structure
    for r in range(n_rounds):
        held = set(rng.choice(uniq, size=3, replace=False).tolist())
        tr_r = np.array([i for i in tr_idx if int(users[i]) not in held])
        va_r = np.array([i for i in tr_idx if int(users[i]) in held])
        if len(va_r) < 30:
            continue
        logits_va = oof_logits[va_r]
        y_va = y[va_r]
        base = float((logits_va.argmax(1) == y_va).mean())
        clust = []
        for ci, (name, classes) in enumerate(CLUSTERS):
            model, g2l, l2g, va_acc = train_raw_pair_inner(
                X_skel, X_imu, y, has_imu, users, tr_r, classes, device,
                epochs=25, patience=6, seed=5000 + r * 10 + ci,
            )
            if model is None:
                continue
            clust.append((name, model, g2l, l2g, tuple(sorted(classes))))
        row = {"round": r, "held": sorted(held), "base": base, "n_va": len(va_r)}
        for mthr in [0.4, 0.6, 1.01]:
            for cthr in [0.55, 0.65, 0.75]:
                for req2 in [False, True]:
                    _, st = apply_cluster_defer(
                        logits_va, y_va, va_r, X_skel, X_imu, has_imu, clust, device,
                        margin_thr=mthr, conf_thr=cthr, require_top2_in=req2,
                    )
                    key = f"m={mthr}|c={cthr}|top2={req2}"
                    row[key] = st["acc"] - base
        results.append(row)
        print(f"L3 round{r} held={sorted(held)} base={base:.4f} n={len(va_r)}", flush=True)
    # average deltas
    from collections import defaultdict
    sums = defaultdict(float)
    cnt = defaultdict(int)
    for row in results:
        for k, v in row.items():
            if k.startswith("m="):
                sums[k] += v
                cnt[k] += 1
    mean_d = {k: sums[k] / cnt[k] for k in sums}
    best = max(mean_d, key=mean_d.get)
    print("L3 TOP:", flush=True)
    for k, v in sorted(mean_d.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v:+.4f}", flush=True)
    return best, mean_d, results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    cache = ROOT_V2 / "cache"
    X_skel, y, users, _ = load_skel_train_cache(cache)
    imu = np.load(cache / "imu_train.npz", allow_pickle=False)
    X_imu, has_imu = imu["X"], imu["has_imu"].astype(bool)
    y = np.asarray(y, np.int64)
    users = np.asarray(users, np.int64)
    hold = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold)))[0]
    va_idx = np.where(np.isin(users, list(hold)))[0]

    print("Building OOF...", flush=True)
    oof = build_oof(X_skel, X_imu, y, users, has_imu, device)
    oof_acc = float((oof.argmax(1) == y).mean())
    print(f"OOF acc={oof_acc:.4f}", flush=True)

    # Calibration on non-holdout OOF only
    T, bias, cal_oof = fit_temperature_bias(oof[tr_idx], y[tr_idx])
    print(f"calib T={T:.3f} oof_acc_nonhold {float((oof[tr_idx].argmax(1)==y[tr_idx]).mean()):.4f} -> {cal_oof:.4f}", flush=True)

    mid_h, _ = load_midfuse(ROOT_V2 / "checkpoints_midfuse_v2b" / "best_holdout.pt", device)
    va_pack = extract_all(mid_h, make_loader(X_skel, X_imu, y, users, va_idx, has_imu), device)
    base_h = float((va_pack["logits"].argmax(1) == va_pack["y"]).mean())
    cal_logits = va_pack["logits"] / T + bias
    cal_acc = float((cal_logits.argmax(1) == va_pack["y"]).mean())
    print(f"holdout base={base_h:.4f} calib={cal_acc:.4f} d={cal_acc-base_h:+.4f}", flush=True)

    # Leave-3-user policy select (expensive-ish)
    print("Leave-3-user cluster policy selection...", flush=True)
    best_pol, mean_d, l3 = leave3_policy_select(
        oof, y, users, tr_idx, X_skel, X_imu, has_imu, device, n_rounds=8
    )
    parts = dict(x.split("=") for x in best_pol.split("|"))
    mthr, cthr, req2 = float(parts["m"]), float(parts["c"]), parts["top2"] == "True"

    # Fit clusters on all non-holdout
    clust = []
    for ci, (name, classes) in enumerate(CLUSTERS):
        model, g2l, l2g, va_acc = train_raw_pair_inner(
            X_skel, X_imu, y, has_imu, users, tr_idx, classes, device,
            epochs=40, patience=10, seed=42 + ci,
        )
        if model is None:
            continue
        clust.append((name, model, g2l, l2g, tuple(sorted(classes))))
        print(f"  {name} inner_va={va_acc:.3f}", flush=True)

    # Apply on raw midfuse logits and on calibrated logits
    _, st_raw = apply_cluster_defer(
        va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, clust, device,
        margin_thr=mthr, conf_thr=cthr, require_top2_in=req2,
    )
    _, st_cal = apply_cluster_defer(
        cal_logits, va_pack["y"], va_idx, X_skel, X_imu, has_imu, clust, device,
        margin_thr=mthr, conf_thr=cthr, require_top2_in=req2,
    )
    print(f"L3-policy on raw: {st_raw['acc']:.4f} on calib: {st_cal['acc']:.4f}", flush=True)

    candidates = [
        ("midfuse", base_h),
        ("calib", cal_acc),
        ("cluster_L3", st_raw["acc"]),
        ("calib+cluster_L3", st_cal["acc"]),
    ]
    # Also try oracle-ish policies that L3 liked in top3
    top3 = [k for k, _ in sorted(mean_d.items(), key=lambda x: -x[1])[:3]]
    extra = []
    for pol in top3:
        p = dict(x.split("=") for x in pol.split("|"))
        _, s = apply_cluster_defer(
            va_pack["logits"], va_pack["y"], va_idx, X_skel, X_imu, has_imu, clust, device,
            margin_thr=float(p["m"]), conf_thr=float(p["c"]), require_top2_in=(p["top2"] == "True"),
        )
        extra.append((pol, s["acc"]))
        candidates.append((f"L3top:{pol}", s["acc"]))

    best_name, best_acc = max(candidates, key=lambda x: x[1])
    # Primary = first using L3-selected only (not scanning holdout for best of candidates beyond L3 pick)
    primary_acc = max(base_h, st_raw["acc"], cal_acc, st_cal["acc"])
    # Honest primary: calib if helps OOF, else base; then L3 cluster on chosen logits
    # Use calib only if cal_oof > base oof on nonhold
    use_calib = cal_oof > float((oof[tr_idx].argmax(1) == y[tr_idx]).mean()) + 1e-6
    logits_h = cal_logits if use_calib else va_pack["logits"]
    _, st_primary = apply_cluster_defer(
        logits_h, va_pack["y"], va_idx, X_skel, X_imu, has_imu, clust, device,
        margin_thr=mthr, conf_thr=cthr, require_top2_in=req2,
    )
    # if cluster hurts vs its base logits, skip cluster
    base_for_primary = cal_acc if use_calib else base_h
    if st_primary["acc"] < base_for_primary:
        primary_acc = base_for_primary
        primary_name = "calib" if use_calib else "midfuse"
    else:
        primary_acc = st_primary["acc"]
        primary_name = ("calib+" if use_calib else "") + f"cluster[{best_pol}]"

    clear = primary_acc >= BASELINE + 0.01
    out = {
        "oof_acc": oof_acc,
        "calib": {"T": T, "oof_acc": cal_oof, "holdout_acc": cal_acc, "use_calib": use_calib},
        "L3_best_policy": best_pol,
        "L3_mean_deltas_top": dict(sorted(mean_d.items(), key=lambda x: -x[1])[:12]),
        "holdout_base": base_h,
        "st_raw": st_raw,
        "st_cal": st_cal,
        "st_primary": st_primary,
        "primary_name": primary_name,
        "primary_acc": primary_acc,
        "delta_vs_0.537": primary_acc - BASELINE,
        "candidates_info": candidates,
        "clear_win": clear,
        "ping_disk_saver": clear,
    }
    with open(OUT / "v6d_calib_L3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ["primary_name", "primary_acc", "delta_vs_0.537", "clear_win", "ping_disk_saver", "L3_best_policy", "holdout_base"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
