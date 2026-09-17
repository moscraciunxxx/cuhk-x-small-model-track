"""ir_v23: tiny IMU spectrogram as 4th complementary soft stream (not Mid replacement).
Train IMUSpec CNN on holdout users {8,9,24}; nested-honest sameT 4-way fuse with
classic9_base + th_v6_v2trio + mid_ens4 + imu_spec. Conf-gate variants a la ir_v21b.
CSV only if hold+nested >= gate (~0.763) AND >=20 disagrees vs ir_v7.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import DEFAULT_HOLD_OUT_USERS
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from probe_ir_v18 import apply_cfg, preds_full, V7_CFG, V7_HOLD, GATE, MIN_DISAGREE
from write_ir_v11 import fuse4_sameT, nested_fixed4, nested_retune4

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
SI = TRACK / "baselines" / "skeleton_imu_v2"
CKPT_DIR = ROOT / "checkpoints" / "imu_spec_v23"
CACHE_IR = ROOT / "cache" / "ir_yolo_v4"
LEAVE = (8, 9, 24)
PT = timezone(timedelta(hours=-7))
warnings.filterwarnings("ignore")

N_FFT = 16
HOP = 4
N_FREQ = N_FFT // 2 + 1  # 9
IMU_CH = 30
N_CLASS = 40


def now_pt() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")


def key_meta(m):
    return (str(m["action_name"]), int(m["user_id"]), str(m["trial"]))


def align_imu_to_ir():
    """Return X_imu_ir (N_ir, T, 30), has_imu (N_ir,), y, users, hold_idx matching IR train order."""
    ir_meta = json.loads((CACHE_IR / "train_meta.json").read_text(encoding="utf-8"))
    sk_meta = json.loads((SI / "cache" / "train_meta.json").read_text(encoding="utf-8"))
    imu = np.load(SI / "cache" / "imu_train.npz")
    X_sk = imu["X"].astype(np.float32)
    has_sk = imu["has_imu"].astype(np.uint8)
    sk_map = {key_meta(m): i for i, m in enumerate(sk_meta)}
    N = len(ir_meta)
    T = X_sk.shape[1]
    X = np.zeros((N, T, IMU_CH), np.float32)
    has = np.zeros((N,), np.uint8)
    hit = 0
    for i, m in enumerate(ir_meta):
        j = sk_map.get(key_meta(m))
        if j is None:
            continue
        X[i] = X_sk[j]
        has[i] = has_sk[j]
        hit += 1
    y = np.load(CACHE_IR / "train_y.npy")
    users = np.load(CACHE_IR / "train_users.npy")
    assert len(y) == N
    hold_idx = np.where(np.isin(users, list(DEFAULT_HOLD_OUT_USERS)))[0]
    print(f"[align] IR={N} IMU-hit={hit} has_imu={int(has.sum())} hold={len(hold_idx)}", flush=True)
    return X, has, y, users, hold_idx


class IMUSpecDataset(Dataset):
    def __init__(self, X, y, indices, has_imu=None, augment=False, seed=42):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, np.int64)
        self.has = has_imu if has_imu is not None else np.ones(len(y), np.uint8)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = np.asarray(self.X[idx], np.float32).copy()  # (T, 30)
        if self.augment:
            if self.rng.rand() < 0.5:
                x += self.rng.randn(*x.shape).astype(np.float32) * 0.05
            if self.rng.rand() < 0.5:
                x = np.roll(x, self.rng.randint(-6, 7), axis=0)
            if self.rng.rand() < 0.3:
                t0 = self.rng.randint(0, x.shape[0])
                w = self.rng.randint(1, max(2, x.shape[0] // 8))
                x[t0 : t0 + w] = 0.0
            if self.rng.rand() < 0.3:
                # channel dropout
                ch = self.rng.randint(0, x.shape[1])
                x[:, ch] = 0.0
        return torch.from_numpy(x), int(self.y[idx]), torch.tensor(float(self.has[idx]), dtype=torch.float32)


class IMUSpecNet(nn.Module):
    """STFT mag spectrogram -> tiny 2D CNN. Input (B, T, 30)."""

    def __init__(self, n_classes=40, n_ch=30, n_fft=N_FFT, hop=HOP, base=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.n_ch = n_ch
        # window buffer
        win = torch.hann_window(n_fft, dtype=torch.float32)
        self.register_buffer("win", win)
        self.stem = nn.Sequential(
            nn.Conv2d(n_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )
        self.block = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base * 2, n_classes),
        )

    def stft_mag(self, x):
        # x: (B, T, C) -> (B, C, F, TT)
        B, T, C = x.shape
        xt = x.permute(0, 2, 1).reshape(B * C, T).float()
        # pad to avoid empty
        spec = torch.stft(
            xt,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.win,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()  # (B*C, F, TT)
        mag = torch.log1p(mag)
        Fbins, TT = mag.shape[-2], mag.shape[-1]
        mag = mag.reshape(B, C, Fbins, TT)
        # per-sample normalize
        mu = mag.mean(dim=(2, 3), keepdim=True)
        sd = mag.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        return (mag - mu) / sd

    def forward(self, x, has_imu=None):
        # x (B,T,C)
        s = self.stft_mag(x)
        h = self.stem(s)
        h = self.block(h).flatten(1)
        logits = self.fc(h)
        if has_imu is not None:
            # zero-ish prior when missing: shrink logits
            m = has_imu.view(-1, 1).to(dtype=logits.dtype)
            logits = logits * m
        return logits.float()


def class_weights(labels, n=40):
    cnt = np.bincount(labels, minlength=n).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    w = cnt.sum() / (n * cnt)
    return torch.tensor(w, dtype=torch.float32).float()


@torch.no_grad()
def predict_logits(model, X, has, indices, device, bs=128):
    model.eval()
    out = np.zeros((len(indices), N_CLASS), np.float32)
    for i0 in range(0, len(indices), bs):
        sl = indices[i0 : i0 + bs]
        xb = torch.from_numpy(X[sl]).to(device)
        hb = torch.from_numpy(has[sl].astype(np.float32)).to(device)
        out[i0 : i0 + len(sl)] = model(xb, hb).cpu().numpy()
    return out


def train_one_seed(X, has, y, users, seed, device, epochs=50, patience=12, lr=1e-3):
    hold_set = set(DEFAULT_HOLD_OUT_USERS)
    tr_idx = np.where(~np.isin(users, list(hold_set)) & (has > 0))[0]
    va_idx = np.where(np.isin(users, list(hold_set)))[0]
    # keep val even if missing imu (rare on hold)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_tr = IMUSpecDataset(X, y, tr_idx, has, augment=True, seed=seed)
    ds_va = IMUSpecDataset(X, y, va_idx, has, augment=False, seed=seed)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=0)
    model = IMUSpecNet().to(device)
    cw = class_weights(y[tr_idx]).to(device=device, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state, bad = -1.0, None, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb, hb in dl_tr:
            xb, yb, hb = xb.to(device), yb.to(device), hb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(yb)
            n += len(yb)
        sched.step()
        # val
        model.eval()
        correct, vn = 0, 0
        with torch.no_grad():
            for xb, yb, hb in dl_va:
                xb, yb, hb = xb.to(device), yb.to(device), hb.to(device)
                pred = model(xb, hb).argmax(1)
                correct += int((pred == yb).sum().item())
                vn += len(yb)
        vac = correct / max(vn, 1)
        hist.append({"ep": ep, "loss": tot / max(n, 1), "val_acc": vac})
        print(f"  seed{seed} ep{ep:02d} loss={tot/max(n,1):.4f} hold={vac:.4f}", flush=True)
        if vac > best_acc + 1e-6:
            best_acc = vac
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop ep{ep} best={best_acc:.4f}", flush=True)
                break
    model.load_state_dict(best_state)
    # full-train logits for IR order (honest hold = never trained on hold)
    all_idx = np.arange(len(y))
    logits_full = predict_logits(model, X, has, all_idx, device)
    return model, best_acc, logits_full, hist


def complementarity(imu_h, mid_h, yt, mask):
    ip = imu_h.argmax(1)
    mp = mid_h.argmax(1)
    both = mask & (imu_h.any(1))
    dis = int(((ip != mp) & both).sum())
    imu_ok = (ip == yt) & both
    mid_ok = (mp == yt) & both
    imu_only = int((imu_ok & ~mid_ok).sum())
    mid_only = int((mid_ok & ~imu_ok).sum())
    agree_both = int((imu_ok & mid_ok).sum())
    imu_acc = float((ip[both] == yt[both]).mean()) if both.any() else 0.0
    mid_acc = float((mp[both] == yt[both]).mean()) if both.any() else 0.0
    return {
        "disagree_vs_mid": dis,
        "imu_correct_mid_wrong": imu_only,
        "mid_correct_imu_wrong": mid_only,
        "both_correct": agree_both,
        "imu_hold_acc": imu_acc,
        "mid_hold_acc": mid_acc,
        "n": int(both.sum()),
    }


def nested_conf_gate4(ir, th, md, imu, y, users, mask, T=1.5):
    """Primary = sameT3 base 0.4/0.3/0.3; when low conf, mix IMU."""
    pi, pt, pm, pu = [softmax_np(z, T) for z in (ir, th, md, imu)]
    base = 0.4 * pi + 0.3 * pt + 0.3 * pm
    mx = base.max(1)
    thrs = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    ws = [0.1, 0.2, 0.35, 0.5, 0.65]
    folds, oof = [], np.full(len(y), -1, np.int64)
    for leave in LEAVE:
        te = mask & (users == leave)
        tr = mask & (users != leave)
        best = (-1.0, None)
        for thr in thrs:
            for w in ws:
                out = base[tr].copy()
                low = mx[tr] < thr
                if low.any():
                    out[low] = (1 - w) * base[tr][low] + w * pu[tr][low]
                acc = float((out.argmax(1) == y[tr]).mean())
                if acc > best[0]:
                    best = (acc, (thr, w))
        thr, w = best[1]
        out_te = base[te].copy()
        low = mx[te] < thr
        if low.any():
            out_te[low] = (1 - w) * base[te][low] + w * pu[te][low]
        pred = out_te.argmax(1)
        oof[te] = pred
        folds.append({"leave": int(leave), "te_acc": float((pred == y[te]).mean()),
                      "n": int(te.sum()), "cfg": {"thr": thr, "w": w, "T": T}})
    # full
    best = (-1.0, None)
    for thr in thrs:
        for w in ws:
            out = base[mask].copy()
            low = mx[mask] < thr
            if low.any():
                out[low] = (1 - w) * base[mask][low] + w * pu[mask][low]
            acc = float((out.argmax(1) == y[mask]).mean())
            if acc > best[0]:
                best = (acc, (thr, w, out.argmax(1)))
    return {
        "nested": float(np.mean([f["te_acc"] for f in folds])) if folds else 0.0,
        "full": float(best[0]),
        "folds": folds,
        "oof": oof,
        "cfg": {"thr": best[1][0], "w": best[1][1], "T": T, "mode": "conf_gate_imu"},
    }


def main():
    t0 = time.time()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ir_v23 IMU spectrogram 4th stream | device={device} | {now_pt()}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    X, has, y_all, users_all, hold_idx = align_imu_to_ir()

    # load fusion members (hold-ordered)
    members, yt, yu = load_members()
    assert len(hold_idx) == len(yt)
    ir_base = np.mean([m["base"] for m in members if m["tag"] != "pool_seed55"], 0).astype(np.float32)
    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid_full = np.load(CACHE_IR / "midfuse_aligned_train_logits_ens4_bonetcn.npy").astype(np.float32)
    mid_h = mid_full[hold_idx]
    mask = th.any(1) & mid_h.any(1)
    print(f"mask={int(mask.sum())}/{len(yt)} ir={(ir_base.argmax(1)==yt).mean():.4f} "
          f"th={(th.argmax(1)==yt).mean():.4f} mid={(mid_h.argmax(1)==yt).mean():.4f}", flush=True)

    # v7 reproduce
    mid_v4 = np.load(CACHE_IR / "midfuse_aligned_train_logits.npy")[hold_idx].astype(np.float32)
    v7_full, _ = apply_cfg(ir_base, th, mid_v4, yt, mask, V7_CFG)
    v7_nest = nested_fixed(ir_base, th, mid_v4, yt, yu, mask, V7_CFG)
    v7_preds = preds_full(ir_base, th, mid_v4, mask, V7_CFG)
    print(f"v7 reproduce full={v7_full:.6f} nested={v7_nest['mean']:.6f}", flush=True)

    # --- train seed 42 ---
    seeds_plan = [42]
    seed_results = {}
    logits_by_seed = {}
    print("=== train IMUSpec seed42 ===", flush=True)
    model, acc42, logits42, hist42 = train_one_seed(X, has, y_all, users_all, 42, device)
    torch.save({"model": model.state_dict(), "hold_acc": acc42, "seed": 42, "hist": hist42},
               CKPT_DIR / "imu_spec_seed42.pt")
    np.save(CKPT_DIR / "train_logits_seed42.npy", logits42)
    seed_results[42] = {"hold_acc": float(acc42), "hist_len": len(hist42)}
    logits_by_seed[42] = logits42
    print(f"seed42 hold_acc={acc42:.4f}", flush=True)

    imu_h = logits42[hold_idx]
    comp = complementarity(imu_h, mid_h, yt, mask)
    print(f"complementarity: {comp}", flush=True)

    multi_seed = acc42 >= 0.45 and (comp["imu_correct_mid_wrong"] >= 15 or comp["disagree_vs_mid"] >= 80)
    if multi_seed:
        seeds_plan = [42, 7, 123, 99]
        print(f"multi-seed GREEN (hold>={0.45}, complementary) -> {seeds_plan[1:]}", flush=True)
        for sd in seeds_plan[1:]:
            print(f"=== train IMUSpec seed{sd} ===", flush=True)
            m, acc, lg, hist = train_one_seed(X, has, y_all, users_all, sd, device)
            torch.save({"model": m.state_dict(), "hold_acc": acc, "seed": sd, "hist": hist},
                       CKPT_DIR / f"imu_spec_seed{sd}.pt")
            np.save(CKPT_DIR / f"train_logits_seed{sd}.npy", lg)
            seed_results[sd] = {"hold_acc": float(acc), "hist_len": len(hist)}
            logits_by_seed[sd] = lg
            print(f"seed{sd} hold_acc={acc:.4f}", flush=True)
    else:
        print(f"multi-seed SKIP (hold={acc42:.4f} comp={comp['imu_correct_mid_wrong']})", flush=True)

    # ensemble IMU hold logits (mean)
    stack = np.stack([logits_by_seed[s][hold_idx] for s in sorted(logits_by_seed)], 0)
    imu_ens = stack.mean(0).astype(np.float32)
    # also soft mean
    sm = np.mean([softmax_np(logits_by_seed[s][hold_idx]) for s in sorted(logits_by_seed)], 0)
    imu_sm = np.log(np.clip(sm, 1e-8, 1)).astype(np.float32)
    imu_variants = {
        "imu_spec_s42": logits_by_seed[42][hold_idx].astype(np.float32),
        "imu_spec_ens": imu_ens,
        "imu_spec_sm": imu_sm,
    }
    for k, v in imu_variants.items():
        acc = float((v.argmax(1) == yt)[mask].mean())
        print(f"  {k} hold_acc={acc:.4f}", flush=True)

    # baseline 3-way (no IMU) — ir_v22 best recipe
    acc3, cfg3 = fuse3_sameT(ir_base, th, mid_h, yt, mask, [1.0, 1.5, 2.0, 2.5], ngrid=21)
    from probe_ir_v18 import nested_retune
    nest3 = nested_retune(ir_base, th, mid_h, yt, yu, mask, [1.0, 1.5, 2.0, 2.5], ngrid=21)
    print(f"3way baseline full={acc3:.4f} nested={nest3['mean']:.4f} cfg={cfg3}", flush=True)

    # 4-way grid
    results = []
    Ts = [1.0, 1.5, 2.0, 2.5]
    for imu_name, imu_h_v in imu_variants.items():
        # mask require imu non-zero OR allow zeros (has missing -> zeros already)
        mask4 = mask.copy()
        acc4, cfg4 = fuse4_sameT(ir_base, th, mid_h, imu_h_v, yt, mask4, Ts, ngrid=17)
        nest_fix = nested_fixed4(ir_base, th, mid_h, imu_h_v, yt, yu, mask4, cfg4)
        nest_rt = nested_retune4(ir_base, th, mid_h, imu_h_v, yt, yu, mask4, Ts, ngrid=13)
        # disagree vs v7 using nested-retune full cfg preds
        T = cfg4["T"]
        preds = (cfg4["wa"] * softmax_np(ir_base, T) + cfg4["wb"] * softmax_np(th, T)
                 + cfg4["wc"] * softmax_np(mid_h, T) + cfg4["wd"] * softmax_np(imu_h_v, T)).argmax(1)
        dis = int(((preds != v7_preds) & mask4 & (v7_preds >= 0)).sum())
        clears = (acc4 >= GATE and nest_rt["mean"] >= GATE and dis >= MIN_DISAGREE)
        row = {
            "kind": "sameT4",
            "imu": imu_name,
            "full": float(acc4),
            "nested_fixed": float(nest_fix),
            "honest_nested": float(nest_rt["mean"]),
            "folds": nest_rt["folds"],
            "cfg": cfg4,
            "disagree_vs_v7": dis,
            "clears": bool(clears),
            "imu_solo": float((imu_h_v.argmax(1) == yt)[mask4].mean()),
            "wd": float(cfg4["wd"]),
        }
        results.append(row)
        print(f"4way {imu_name}: full={acc4:.4f} nest_fix={nest_fix:.4f} nest_rt={nest_rt['mean']:.4f} "
              f"wd={cfg4['wd']:.3f} dis={dis} clears={clears}", flush=True)

        # conf-gate
        for Tcg in [1.0, 1.5, 2.0]:
            cg = nested_conf_gate4(ir_base, th, mid_h, imu_h_v, yt, yu, mask4, T=Tcg)
            dis_cg = int(((cg["oof"] != v7_preds) & mask4 & (cg["oof"] >= 0) & (v7_preds >= 0)).sum())
            clears_cg = (cg["full"] >= GATE and cg["nested"] >= GATE and dis_cg >= MIN_DISAGREE)
            row_cg = {
                "kind": "conf_gate_imu",
                "imu": imu_name,
                "full": float(cg["full"]),
                "honest_nested": float(cg["nested"]),
                "folds": cg["folds"],
                "cfg": cg["cfg"],
                "disagree_vs_v7": dis_cg,
                "clears": bool(clears_cg),
                "imu_solo": float((imu_h_v.argmax(1) == yt)[mask4].mean()),
            }
            results.append(row_cg)
            print(f"  conf_gate T={Tcg} full={cg['full']:.4f} nested={cg['nested']:.4f} "
                  f"dis={dis_cg} clears={clears_cg}", flush=True)

    results_sorted = sorted(results, key=lambda r: (-r["honest_nested"], -r["full"]))
    best = results_sorted[0]

    # decide miss / win
    imu_useless = (
        acc42 < 0.35
        or all(r.get("wd", 1.0) is not None and abs(r.get("wd", 1)) < 1e-6
               for r in results if r["kind"] == "sameT4")
        or best["honest_nested"] <= nest3["mean"] + 0.001
    )
    # refine: weight->0 check
    wd_max = max((r.get("wd", 0.0) or 0.0) for r in results if r["kind"] == "sameT4")
    nested_lift = best["honest_nested"] - nest3["mean"]

    wrote_csv = False
    csv_path = None
    if best["clears"]:
        # build test logits + submission
        print("GATE CLEARED — building test submission", flush=True)
        # infer test
        imu_te = np.load(SI / "cache" / "imu_test.npz", allow_pickle=True)
        Xte = imu_te["X"].astype(np.float32)
        has_te = imu_te["has_imu"].astype(np.float32)
        test_logits = []
        for sd in sorted(logits_by_seed):
            blob = torch.load(CKPT_DIR / f"imu_spec_seed{sd}.pt", map_location="cpu", weights_only=False)
            m = IMUSpecNet().to(device)
            m.load_state_dict(blob["model"])
            m.eval()
            lg = predict_logits(m, Xte, has_te, np.arange(len(Xte)), device)
            test_logits.append(lg)
            np.save(CKPT_DIR / f"test_logits_seed{sd}.npy", lg)
        imu_test = np.mean(test_logits, 0).astype(np.float32)
        # IR/TH/MID test
        ir_test = None
        for p in [
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "test_logits_classic9_base.npy",
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "test_logits_classic9.npy",
        ]:
            if p.exists():
                ir_test = np.load(p).astype(np.float32)
                break
        if ir_test is None:
            # mean of member test base
            ir_test = np.mean([m["test_logits"] for m in members if m["tag"] != "pool_seed55"], 0).astype(np.float32)
        th_test = None
        for p in [
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "test_thermal_v6.npy",
            ROOT / "checkpoints" / "thermal_v6" / "test_logits.npy",
        ]:
            if p.exists():
                th_test = np.load(p).astype(np.float32)
                break
        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
        # prefer ens4 test if exists
        ens4_te = ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_test_logits_ens4_bonetcn.npy"
        if ens4_te.exists():
            mid_test = np.load(ens4_te).astype(np.float32)

        cfg = best["cfg"]
        if best["kind"] == "sameT4":
            T = cfg["T"]
            preds = (cfg["wa"] * softmax_np(ir_test, T) + cfg["wb"] * softmax_np(th_test, T)
                     + cfg["wc"] * softmax_np(mid_test, T) + cfg["wd"] * softmax_np(imu_test, T)).argmax(1)
        else:
            # conf gate on test
            T = cfg["T"]
            pi, pt, pm, pu = [softmax_np(z, T) for z in (ir_test, th_test, mid_test, imu_test)]
            base = 0.4 * pi + 0.3 * pt + 0.3 * pm
            out = base.copy()
            low = base.max(1) < cfg["thr"]
            out[low] = (1 - cfg["w"]) * base[low] + cfg["w"] * pu[low]
            preds = out.argmax(1)

        import csv
        csv_path = ROOT / "submission_ir_v23.csv"
        meta = json.loads((CACHE_IR / "test_meta.json").read_text(encoding="utf-8"))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "prediction"])
            for i, m in enumerate(meta):
                p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
                w.writerow([p, int(preds[i])])
        wrote_csv = True
        print(f"wrote {csv_path}", flush=True)

    # ceiling / miss status
    outcome = "WIN" if best["clears"] else "MISS"
    ceiling = not best["clears"]
    gap = GATE - best["honest_nested"]

    status = {
        "tag": "ir_v23_imu_spec_4th",
        "outcome": outcome,
        "keep_ir_v7": not best["clears"],
        "finished_at": now_pt(),
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "inventory_imu": {
            "imu_train_npz": str(SI / "cache" / "imu_train.npz"),
            "imu_test_npz": str(SI / "cache" / "imu_test.npz"),
            "shape_train": list(X.shape),
            "has_imu_train": int(has.sum()),
            "aligned_to_ir": "action_name,user_id,trial",
            "midfuse_loaders": str(SI / "dataset.py"),
            "note": "skeleton_imu_v2 midfuse already uses raw IMU Conv; this stream is STFT spectrogram CNN complementary",
        },
        "imu_spec_train": {
            "seeds": seed_results,
            "multi_seed": multi_seed,
            "seed42_hold": float(acc42),
            "complementarity_vs_mid": comp,
            "ckpt_dir": str(CKPT_DIR),
        },
        "baseline_3way": {
            "full": float(acc3),
            "honest_nested": float(nest3["mean"]),
            "cfg": cfg3,
            "ir": "classic9_base",
            "th": "th_v6_v2trio",
            "mid": "mid_ens4_bonetcn",
        },
        "best": best,
        "top8": results_sorted[:8],
        "wd_max": float(wd_max),
        "nested_lift_vs_3way": float(nested_lift),
        "imu_useless_or_no_lift": bool(imu_useless or wd_max < 1e-6 or nested_lift < 0.002),
        "wrote_csv": wrote_csv,
        "csv": str(csv_path) if csv_path else None,
        "gap_to_gate_nested": float(gap),
        "ceiling": {
            "verdict": (
                f"IMU spectrogram 4th stream {'CLEARS' if best['clears'] else 'MISS'}: "
                f"best honest_nested={best['honest_nested']:.4f} full={best['full']:.4f} "
                f"wd_max={wd_max:.4f} lift_vs_3way={nested_lift:+.4f} gap_to_gate={gap:.4f}. "
                + (
                    "Keep submission_ir_v7.csv @ public 0.69154. Small-track ceiling near ir_v7."
                    if not best["clears"]
                    else "New submission_ir_v23.csv written."
                )
            ),
            "vs_gate": GATE,
            "delta_vs_gate": float(best["honest_nested"] - GATE),
            "small_track_ceiling": not best["clears"],
            "stop_new_ir_kinetics": True,
            "stop_thermal_seeds": True,
            "stop_mid_arch_churn": True,
            "stop_imu_spec_retries": not best["clears"],
        },
        "next_roi": (
            [
                "WIN: submit submission_ir_v23.csv; verify public LB",
                "Keep monitoring vs ir_v7 @ 0.69154",
            ]
            if best["clears"]
            else [
                "MISS: IMU spectrogram 4th stream did not clear gate",
                "Small-track ceiling: keep submission_ir_v7.csv @ public 0.69154 / hold 0.753",
                "Do NOT train new IR Kinetics-3D / Thermal R2+1D seeds / Mid arch / LOUO / more IMU-spec",
                "Shift effort elsewhere (Large track / other ROI)",
            ]
        ),
        "gpu_handoff": {
            "machine": "MosCraciunXXX",
            "machineId": "4ff6e647-6f8c-4d02-88f0-5dda6684ae36",
            "at_start": "exclusive RTX 3060 (LMT CPU-only overnight)",
            "at_end": "releasing GPU after ir_v23",
            "finished_at": now_pt(),
            "status": "FREE",
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "notes": [
            "Nested sameT4 only (+ conf-gate IMU mix); no perT grid overfit",
            "IMU STFT n_fft=16 hop=4 log1p mag -> tiny Conv2d",
            "classic9_base + th_v6_v2trio + mid_ens4_bonetcn + imu_spec",
        ],
    }

    outp = ROOT / "metrics_ir_v23_status.json"
    outp.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(f"wrote {outp}", flush=True)
    print(json.dumps({
        "outcome": outcome,
        "best_nested": best["honest_nested"],
        "best_full": best["full"],
        "wd_max": wd_max,
        "lift": nested_lift,
        "seed42": acc42,
        "wrote_csv": wrote_csv,
        "gap": gap,
    }, indent=2), flush=True)

    # free GPU
    if device.type == "cuda":
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
    print(f"DONE {now_pt()} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

