"""ir_v29 recipe revisit on classic ir_yolo_v4 (T16): longer freeze, mild mixup, stronger LS, longer cosine.
Aim solo >=0.72 to beat classic9 members (~0.65-0.70). AdamW first; optional SAM.
Do NOT overwrite classic ckpts. Dump hold+test logits for near-v7 classic-mid fuse.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score
from dataset import CachedClipDataset, DEFAULT_HOLD_OUT_USERS, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def build(pretrained=True):
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, preds, logits_all = [], [], []
    for x, y, _u, _i in loader:
        x = normalize(x.to(device)).permute(0, 2, 1, 3, 4).contiguous()
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(preds)
    return {"acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "logits": np.concatenate(logits_all), "y": yt}


class SAM(torch.optim.Optimizer):
    """Lightweight Sharpness-Aware Minimization wrapper around a base optimizer."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self):
        shared = self.param_groups[0]["params"][0].device
        norms = [p.grad.norm(p=2).to(shared) for g in self.param_groups for p in g["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0, device=shared)

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        raise RuntimeError("Use first_step/second_step for SAM")


def make_opt(model, lr, use_sam=False, rho=0.05):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if use_sam:
        return SAM(params, torch.optim.AdamW, rho=rho, lr=lr, weight_decay=1e-4)
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def train_one(X, y, users, pool_idx, hold_idx, device, seed, epochs, batch_size, lr, patience,
              ckpt_path, mixup_alpha=0.1, unfreeze_ep=10, label_smoothing=0.1, use_sam=False, sam_rho=0.05):
    set_seed(seed)
    tag = f"seed{seed}"
    model = build(True).to(device)
    for name, p in model.named_parameters():
        if any(k in name for k in ["stem", "layer1"]):
            p.requires_grad = False
    opt = make_opt(model, lr, use_sam=use_sam, rho=sam_rho)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt.base_optimizer if use_sam else opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda")
    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = CachedClipDataset(X, y, users, pool_idx, train=True, seed=seed)
    val_ds = CachedClipDataset(X, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=WeightedRandomSampler(w, num_samples=len(pool_idx), replacement=True),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            opt = make_opt(model, lr * 0.3, use_sam=use_sam, rho=sam_rho)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt.base_optimizer if use_sam else opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all @ep{ep} lr={lr*0.3:g} sam={use_sam}", flush=True)
        model.train(); loss_sum = correct = n = 0
        for xb, yb, _u, _i in train_loader:
            xb = normalize(xb.to(device)).permute(0, 2, 1, 3, 4).contiguous()
            yb = yb.to(device)
            if mixup_alpha > 0 and xb.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb_m = lam * xb + (1 - lam) * xb[idx]
                y1, y2 = yb, yb[idx]
            else:
                lam, xb_m, y1, y2 = 1.0, xb, yb, yb

            def closure_loss():
                with torch.amp.autocast("cuda"):
                    logits_ = model(xb_m)
                    if lam < 1.0:
                        loss_ = lam * crit(logits_, y1) + (1 - lam) * crit(logits_, y2)
                    else:
                        loss_ = crit(logits_, yb)
                return logits_, loss_

            opt.zero_grad(set_to_none=True)
            if use_sam:
                # SAM needs full-precision path around two steps; keep amp on forward
                logits, loss = closure_loss()
                scaler.scale(loss).backward()
                scaler.unscale_(opt.base_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                # first SAM step in fp32 grads
                opt.first_step(zero_grad=True)
                logits2, loss2 = closure_loss()
                scaler.scale(loss2).backward()
                scaler.unscale_(opt.base_optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.second_step(zero_grad=True)
                scaler.update()
                loss_val = float(loss2.item())
            else:
                logits, loss = closure_loss()
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                loss_val = float(loss.item())
            loss_sum += loss_val * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item()); n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": ep, "tr_loss": loss_sum / max(n, 1), "tr_acc": correct / max(n, 1),
               "val_acc": metrics["acc"], "val_f1": metrics["macro_f1"]}
        history.append(row)
        print(f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
              f"val={row['val_acc']:.4f} f1={row['val_f1']:.4f}", flush=True)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save({"model": best_state, "model_fp16": sd16, "val_acc": best_acc,
                        "epoch": ep, "seed": seed, "history": history, "hold_logits": metrics["logits"],
                        "hold_y": metrics["y"], "recipe": {
                            "mixup": mixup_alpha, "unfreeze_ep": unfreeze_ep,
                            "label_smoothing": label_smoothing, "sam": use_sam,
                            "epochs": epochs, "lr": lr}}, ckpt_path)
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state); model.to(device)
    hold_m = evaluate(model, val_loader, device)
    del model; torch.cuda.empty_cache()
    return best_acc, best_state, history, hold_m["logits"], hold_m["y"]


@torch.no_grad()
def infer_cache(model, Xt, device, bs=8):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i + len(x)] = model(x).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v29_recipe"))
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 4096])
    ap.add_argument("--mixup", type=float, default=0.1)
    ap.add_argument("--unfreeze-ep", type=int, default=10)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--sam", action="store_true")
    ap.add_argument("--sam-rho", type=float, default=0.05)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = np.load(cache / "train_y.npy"); users = np.load(cache / "train_users.npy")
    X = np.memmap(cache / f"train_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                  shape=(len(y), args.t, args.size, args.size, 3))
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device} cache={cache}", flush=True)
    print(f"recipe mixup={args.mixup} ls={args.label_smoothing} unfreeze@{args.unfreeze_ep} "
          f"ep={args.epochs} lr={args.lr} sam={args.sam} seeds={args.seeds}", flush=True)

    members = []
    hold_pack = {}
    for seed in args.seeds:
        ck = ckpt_dir / f"pool_seed{seed}.pt"
        if args.skip_train and ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            acc = float(blob.get("val_acc", -1))
            print(f"reuse {ck.name} val={acc}", flush=True)
            members.append({"seed": seed, "acc": acc, "state": blob["model"], "path": ck})
            if "hold_logits" in blob:
                hold_pack[f"s{seed}"] = blob["hold_logits"]
                np.save(ckpt_dir / f"hold_logits_seed{seed}.npy", blob["hold_logits"])
            continue
        acc, state, hist, hlog, hy = train_one(
            X, y, users, pool_idx, hold_idx, device, seed, args.epochs, args.batch_size,
            args.lr, args.patience, ck, mixup_alpha=args.mixup, unfreeze_ep=args.unfreeze_ep,
            label_smoothing=args.label_smoothing, use_sam=args.sam, sam_rho=args.sam_rho)
        members.append({"seed": seed, "acc": acc, "state": state, "path": ck})
        hold_pack[f"s{seed}"] = hlog
        np.save(ckpt_dir / f"hold_logits_seed{seed}.npy", hlog)
        if "y" not in hold_pack:
            hold_pack["y"] = hy

    # re-eval hold for consistency
    val_loader = DataLoader(CachedClipDataset(X, y, users, hold_idx, train=False, seed=0),
                            batch_size=12, shuffle=False, num_workers=0)
    holds = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"]); model.eval()
        met = evaluate(model, val_loader, device)
        print(f"eval seed{m['seed']} hold={met['acc']:.4f}", flush=True)
        holds.append(met["logits"])
        np.save(ckpt_dir / f"hold_logits_seed{m['seed']}.npy", met["logits"])
        m["acc"] = met["acc"]
        del model; torch.cuda.empty_cache()
    ens = np.mean(holds, 0).astype(np.float32)
    np.savez_compressed(ckpt_dir / "hold_logits_v29.npz", ens=ens, **{f"s{m['seed']}": h for m, h in zip(members, holds)},
                        y=y[hold_idx])
    print(f"ens hold={(ens.argmax(1)==y[hold_idx]).mean():.4f}", flush=True)

    # test infer
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s{args.size}.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), args.t, args.size, args.size, 3))
    test_logs = []
    for m in members:
        model = build(False).to(device); model.load_state_dict(m["state"]); model.eval()
        tl = infer_cache(model, Xt, device, bs=8)
        np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", tl)
        test_logs.append(tl)
        print(f"inferred test seed{m['seed']}", flush=True)
        del model; torch.cuda.empty_cache()
    tens = np.mean(test_logs, 0).astype(np.float32)
    np.save(ckpt_dir / "test_logits_ens.npy", tens)

    report = {
        "tag": "ir_v29_recipe",
        "cache": str(cache),
        "mixup": args.mixup,
        "label_smoothing": args.label_smoothing,
        "unfreeze_ep": args.unfreeze_ep,
        "epochs": args.epochs,
        "lr": args.lr,
        "sam": args.sam,
        "members": {f"s{m['seed']}": m["acc"] for m in members},
        "ens_hold": float((ens.argmax(1) == y[hold_idx]).mean()),
        "best_solo": float(max(m["acc"] for m in members)),
    }
    (ckpt_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "metrics_ir_v29_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()