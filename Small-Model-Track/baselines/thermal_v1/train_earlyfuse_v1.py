"""Early-fuse IR+Depth: dual / ch6 / shared(siamese) / hybrid(IR+compact-depth).
Reuses cache/ir_yolo_v4 + cache/depth_color_yolo_v4_irbox (index-aligned).
Size: dual~120MB ILLEGAL; shared/hybrid/ch6 fp16 ~60MB (+yolo~66) LEGAL.
Gate vs ir_v7: hold AND nested >= 0.763 and >=20 disagrees; else status JSON only.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score

from dataset import DEFAULT_HOLD_OUT_USERS, NUM_CLASSES
from model_r2p1d import CompactR2Plus1D

ROOT = Path(__file__).resolve().parent
HOLD = set(DEFAULT_HOLD_OUT_USERS)
IR_V7_HOLD = 0.7530364372469636
GATE_DELTA = 0.01
GATE_MIN_HOLD = IR_V7_HOLD + GATE_DELTA  # ~0.763
GATE_MIN_DISAGREE = 20

K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x: (B,T,C,H,W) with C=3."""
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


class DualCachedClipDataset(Dataset):
    """Paired IR+Depth caches with synchronized spatial/temporal aug."""

    def __init__(
        self,
        X_ir: np.ndarray,
        X_dp: np.ndarray,
        labels: np.ndarray,
        users: np.ndarray,
        indices: np.ndarray | list[int],
        train: bool = False,
        seed: int = 0,
    ):
        assert len(X_ir) == len(X_dp)
        self.X_ir = X_ir
        self.X_dp = X_dp
        self.labels = labels
        self.users = users
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def _aug_pair(self, a: np.ndarray, b: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        # a,b: T,H,W,C uint8
        if rng.random() < 0.15:
            a = a[::-1].copy()
            b = b[::-1].copy()
        do_flip = rng.random() < 0.5
        t, h, w, c = a.shape
        if rng.random() < 0.7:
            ch = int(h * (0.8 + 0.2 * rng.random()))
            cw = int(w * (0.8 + 0.2 * rng.random()))
            top = rng.randint(0, max(h - ch, 0))
            left = rng.randint(0, max(w - cw, 0))
            a = a[:, top : top + ch, left : left + cw, :]
            b = b[:, top : top + ch, left : left + cw, :]
            ys = (np.linspace(0, a.shape[1] - 1, h)).astype(np.int64)
            xs = (np.linspace(0, a.shape[2] - 1, w)).astype(np.int64)
            a = a[:, ys][:, :, xs]
            b = b[:, ys][:, :, xs]
        if do_flip:
            a = a[:, :, ::-1, :].copy()
            b = b[:, :, ::-1, :].copy()
        xa = a.astype(np.float32) / 255.0
        xb = b.astype(np.float32) / 255.0
        # mild photometric, independent per modality
        for x in (xa, xb):
            bri = 0.85 + 0.3 * rng.random()
            con = 0.85 + 0.3 * rng.random()
            x[:] = np.clip((x - 0.5) * con + 0.5 * bri, 0.0, 1.0)
        return xa, xb

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        a = self.X_ir[idx]
        b = self.X_dp[idx]
        if self.train:
            rng = random.Random(self.seed + idx * 13 + random.randint(0, 10**9))
            xa, xb = self._aug_pair(a, b, rng)
        else:
            xa = a.astype(np.float32) / 255.0
            xb = b.astype(np.float32) / 255.0
        xa = torch.from_numpy(np.ascontiguousarray(xa.transpose(0, 3, 1, 2)))  # T,C,H,W
        xb = torch.from_numpy(np.ascontiguousarray(xb.transpose(0, 3, 1, 2)))
        y = int(self.labels[idx])
        uid = int(self.users[idx])
        return xa, xb, y, uid, idx


class DualStemR2P1D(nn.Module):
    """Two Kinetics R(2+1)D-18 stems -> concat 512+512 -> classifier."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.ir = r2plus1d_18(weights=w)
        self.dp = r2plus1d_18(weights=w)
        feat = self.ir.fc.in_features
        self.ir.fc = nn.Identity()
        self.dp.fc = nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat * 2, NUM_CLASSES)

    def forward(self, x_ir: torch.Tensor, x_dp: torch.Tensor) -> torch.Tensor:
        # expect B,C,T,H,W
        f = torch.cat([self.ir(x_ir), self.dp(x_dp)], dim=1)
        return self.head(self.drop(f))

    def freeze_early(self, which: str = "stem_l1") -> None:
        for stem in (self.ir, self.dp):
            for name, p in stem.named_parameters():
                if which == "all_but_head":
                    p.requires_grad = False
                elif which == "stem_l1":
                    p.requires_grad = not any(k in name for k in ["stem", "layer1"])
                elif which == "stem_l1_l2":
                    p.requires_grad = not any(k in name for k in ["stem", "layer1", "layer2"])
                else:
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


class Channel6R2P1D(nn.Module):
    """Single R(2+1)D-18 with 6-channel stem (IR||Depth)."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=w)
        old = self.backbone.stem[0]
        new = nn.Conv3d(
            6,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                new.weight[:, :3] = old.weight
                new.weight[:, 3:] = old.weight
            else:
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.stem[0] = new
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.fc.in_features, NUM_CLASSES),
        )

    def forward(self, x_ir: torch.Tensor, x_dp: torch.Tensor) -> torch.Tensor:
        # B,C,T,H,W each C=3 -> concat channel -> B,6,T,H,W
        x = torch.cat([x_ir, x_dp], dim=1)
        return self.backbone(x)

    def freeze_early(self, which: str = "stem_l1") -> None:
        for name, p in self.backbone.named_parameters():
            if "fc" in name:
                p.requires_grad = True
                continue
            if which == "all_but_head":
                p.requires_grad = False
            elif which == "stem_l1":
                p.requires_grad = not any(k in name for k in ["stem", "layer1"])
            else:
                p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True



class SharedStemR2P1D(nn.Module):
    """Weight-tied R(2+1)D-18: same stem for IR+Depth, concat feats -> head.
    fp16 pack ~60MB (legal). Optional 1x1 modality adapters (~few KB).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4, adapters: bool = True):
        super().__init__()
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=w)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.use_adapters = adapters
        if adapters:
            self.ir_adapt = nn.Sequential(nn.Conv3d(3, 3, 1, bias=False), nn.BatchNorm3d(3))
            self.dp_adapt = nn.Sequential(nn.Conv3d(3, 3, 1, bias=False), nn.BatchNorm3d(3))
            with torch.no_grad():
                self.ir_adapt[0].weight.zero_()
                self.dp_adapt[0].weight.zero_()
                for i in range(3):
                    self.ir_adapt[0].weight[i, i, 0, 0, 0] = 1.0
                    self.dp_adapt[0].weight[i, i, 0, 0, 0] = 1.0
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat * 2, NUM_CLASSES)

    def forward(self, x_ir: torch.Tensor, x_dp: torch.Tensor) -> torch.Tensor:
        if self.use_adapters:
            x_ir = self.ir_adapt(x_ir)
            x_dp = self.dp_adapt(x_dp)
        f = torch.cat([self.backbone(x_ir), self.backbone(x_dp)], dim=1)
        return self.head(self.drop(f))

    def freeze_early(self, which: str = "stem_l1") -> None:
        for name, p in self.backbone.named_parameters():
            if which == "all_but_head":
                p.requires_grad = False
            elif which == "stem_l1":
                p.requires_grad = not any(k in name for k in ["stem", "layer1"])
            elif which == "stem_l1_l2":
                p.requires_grad = not any(k in name for k in ["stem", "layer1", "layer2"])
            else:
                p.requires_grad = True
        for mod in (getattr(self, "ir_adapt", None), getattr(self, "dp_adapt", None), self.head):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


class HybridIRCompactDepth(nn.Module):
    """Full Kinetics IR stem + tiny CompactR2Plus1D depth branch; feature concat.
    fp16 pack ~61MB (legal). Depth capacity limited on purpose for size/VRAM.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.4, depth_base: int = 32):
        super().__init__()
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.ir = r2plus1d_18(weights=w)
        feat_ir = self.ir.fc.in_features
        self.ir.fc = nn.Identity()
        self.dp = CompactR2Plus1D(num_classes=NUM_CLASSES, in_ch=3, base=depth_base, dropout=dropout)
        feat_dp = depth_base * 8
        self.dp.head = nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_ir + feat_dp, NUM_CLASSES)

    def _dp_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dp.stem(x)
        x = self.dp.layer1(x)
        x = self.dp.layer2(x)
        x = self.dp.layer3(x)
        return self.dp.pool(x).flatten(1)

    def forward(self, x_ir: torch.Tensor, x_dp: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.ir(x_ir), self._dp_feat(x_dp)], dim=1)
        return self.head(self.drop(f))

    def freeze_early(self, which: str = "stem_l1") -> None:
        for name, p in self.ir.named_parameters():
            if which == "all_but_head":
                p.requires_grad = False
            elif which == "stem_l1":
                p.requires_grad = not any(k in name for k in ["stem", "layer1"])
            elif which == "stem_l1_l2":
                p.requires_grad = not any(k in name for k in ["stem", "layer1", "layer2"])
            else:
                p.requires_grad = True
        for p in self.dp.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def build_model(mode: str, pretrained: bool = True) -> nn.Module:
    if mode == "dual":
        return DualStemR2P1D(pretrained=pretrained)
    if mode == "ch6":
        return Channel6R2P1D(pretrained=pretrained)
    if mode == "shared":
        return SharedStemR2P1D(pretrained=pretrained, adapters=True)
    if mode == "hybrid":
        return HybridIRCompactDepth(pretrained=pretrained, depth_base=32)
    raise ValueError(mode)


def pack_size_mb(model: nn.Module) -> float:
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    # rough: sum nbytes
    n = 0
    for v in sd.values():
        n += v.numel() * v.element_size()
    return n / (1024 * 1024)


def to_bcthw(x: torch.Tensor) -> torch.Tensor:
    # T,C,H,W batch -> B,C,T,H,W after normalize on B,T,C
    x = normalize(x)
    return x.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mode: str) -> dict[str, Any]:
    model.eval()
    ys, preds, logits_all, users = [], [], [], []
    for batch in loader:
        xa, xb, y, u, _i = batch
        xa = to_bcthw(xa.to(device))
        xb = to_bcthw(xb.to(device))
        logits = model(xa, xb)
        logits_all.append(logits.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        users.append(u.numpy())
    yt = np.concatenate(ys)
    yp = np.concatenate(preds)
    yu = np.concatenate(users)
    per_user = {}
    for leave in sorted(set(yu.tolist())):
        m = yu == leave
        per_user[int(leave)] = float((yt[m] == yp[m]).mean()) if m.any() else 0.0
    nested = float(np.mean(list(per_user.values()))) if per_user else 0.0
    return {
        "acc": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "nested_louo": nested,
        "per_user": per_user,
        "logits": np.concatenate(logits_all),
        "y": yt,
        "users": yu,
    }


def train_one(
    X_ir,
    X_dp,
    y,
    users,
    pool_idx,
    hold_idx,
    device,
    seed,
    epochs,
    batch_size,
    lr,
    patience,
    ckpt_path: Path,
    mode: str,
    mixup_alpha: float = 0.2,
    unfreeze_ep: int = 5,
    freeze_which: str = "stem_l1",
):
    set_seed(seed)
    tag = f"{mode}_seed{seed}"
    model = build_model(mode, pretrained=True).to(device)
    model.freeze_early(freeze_which)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_cuda else None

    counts = np.bincount(y[pool_idx], minlength=40)
    w = 1.0 / np.maximum(counts[y[pool_idx]], 1)
    train_ds = DualCachedClipDataset(X_ir, X_dp, y, users, pool_idx, train=True, seed=seed)
    val_ds = DualCachedClipDataset(X_ir, X_dp, y, users, hold_idx, train=False, seed=0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(w, num_samples=len(pool_idx), replacement=True),
        num_workers=0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 4), shuffle=False, num_workers=0, pin_memory=use_cuda)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] trainable={n_train/1e6:.2f}M / {n_all/1e6:.2f}M fp16_pack~{pack_size_mb(model):.1f}MB", flush=True)

    best_acc, best_ep, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if ep == unfreeze_ep:
            model.unfreeze_all()
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.25, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - ep + 1, 1))
            print(f"[{tag}] unfroze all lr={lr*0.25}", flush=True)
        model.train()
        loss_sum = correct = n = 0
        for xa, xb, yb, _u, _i in train_loader:
            xa = to_bcthw(xa.to(device))
            xb = to_bcthw(xb.to(device))
            yb = yb.to(device)
            if mixup_alpha > 0 and xa.size(0) > 1:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(xa.size(0), device=xa.device)
                xa_m = lam * xa + (1 - lam) * xa[idx]
                xb_m = lam * xb + (1 - lam) * xb[idx]
                y1, y2 = yb, yb[idx]
            else:
                lam, xa_m, xb_m, y1, y2 = 1.0, xa, xb, yb, yb
            opt.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast("cuda"):
                    logits = model(xa_m, xb_m)
                    loss = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2) if lam < 1 else crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xa_m, xb_m)
                loss = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2) if lam < 1 else crit(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)
        sched.step()
        metrics = evaluate(model, val_loader, device, mode)
        row = {
            "epoch": ep,
            "tr_loss": loss_sum / max(n, 1),
            "tr_acc": correct / max(n, 1),
            "val_acc": metrics["acc"],
            "val_f1": metrics["macro_f1"],
            "val_nested": metrics["nested_louo"],
            "per_user": metrics["per_user"],
        }
        history.append(row)
        print(
            f"[{tag}] ep{ep:03d} loss={row['tr_loss']:.4f} tr={row['tr_acc']:.3f} "
            f"val={row['val_acc']:.4f} nest={row['val_nested']:.4f} f1={row['val_f1']:.4f}",
            flush=True,
        )
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()}
            torch.save(
                {
                    "model": best_state,
                    "model_fp16": sd16,
                    "val_acc": best_acc,
                    "val_nested": metrics["nested_louo"],
                    "per_user": metrics["per_user"],
                    "epoch": ep,
                    "seed": seed,
                    "mode": mode,
                    "history": history,
                    "hold_logits": metrics["logits"],
                    "hold_y": metrics["y"],
                    "hold_users": metrics["users"],
                    "fp16_mb": pack_size_mb(model),
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path.name} acc={best_acc:.4f} nest={metrics['nested_louo']:.4f}", flush=True)
        if patience > 0 and ep - best_ep >= patience:
            print(f"[{tag}] early stop ep{ep} best={best_acc:.4f}", flush=True)
            break
    print(f"[{tag}] BEST={best_acc:.4f} took={time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    model.to(device)
    hold_m = evaluate(model, val_loader, device, mode)
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return best_acc, best_state, history, hold_m


@torch.no_grad()
def infer_pair(model, X_ir, X_dp, device, bs=4, flip=False):
    model.eval()
    n = len(X_ir)
    out = np.zeros((n, NUM_CLASSES), np.float32)
    for i in range(0, n, bs):
        a = X_ir[i : i + bs].astype(np.float32) / 255.0
        b = X_dp[i : i + bs].astype(np.float32) / 255.0
        if flip:
            a = a[:, :, :, ::-1, :].copy()
            b = b[:, :, :, ::-1, :].copy()
        xa = torch.from_numpy(np.ascontiguousarray(a.transpose(0, 1, 4, 2, 3))).to(device)
        xb = torch.from_numpy(np.ascontiguousarray(b.transpose(0, 1, 4, 2, 3))).to(device)
        xa = to_bcthw(xa)
        xb = to_bcthw(xb)
        out[i : i + len(xa)] = model(xa, xb).float().cpu().numpy()
    return out


def load_v7_hold_preds() -> tuple[np.ndarray, np.ndarray]:
    """Best-effort IR v7 hold predictions for disagree count."""
    # Prefer refined v7 ens if present via submission labels + hold keys
    sub = ROOT / "submission_ir_v7.csv"
    # Use IR ens hold logits if available
    candidates = [
        ROOT / "checkpoints" / "ir_yolo_r2p1d18_v7" / "hold_logits_v7.npz",
        ROOT / "checkpoints" / "ir_yolo_r2p1d18_v6" / "hold_logits_new_seeds.npz",
        ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_logits_v6.npz",
    ]
    for p in candidates:
        if p.exists():
            z = np.load(p, allow_pickle=True)
            if "ens" in z:
                return z["ens"], z["y"]
            if "logits" in z:
                lg = z["logits"]
                if lg.ndim == 3:
                    return lg.mean(0), z["y"]
                return lg, z["y"]
    # fallback from CSV is test-only; return None markers
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-cache", default=str(ROOT / "cache" / "ir_yolo_v4"))
    ap.add_argument("--dp-cache", default=str(ROOT / "cache" / "depth_color_yolo_v4_irbox"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "sharedstem_ir_depth_v1"))
    ap.add_argument("--mode", choices=["dual", "ch6", "shared", "hybrid"], default="shared")
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--unfreeze-ep", type=int, default=5)
    ap.add_argument("--freeze", default="stem_l1", choices=["stem_l1", "stem_l1_l2", "all_but_head", "none"])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (prep/dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="1-batch forward sanity on CPU/GPU")
    ap.add_argument("--infer-test", action="store_true")
    args = ap.parse_args()

    ir_cache = Path(args.ir_cache)
    dp_cache = Path(args.dp_cache)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    y = np.load(ir_cache / "train_y.npy")
    users = np.load(ir_cache / "train_users.npy")
    yd = np.load(dp_cache / "train_y.npy")
    assert np.all(y == yd), "IR/Depth label mismatch"
    X_ir = np.memmap(
        ir_cache / f"train_x_t{args.t}_s{args.size}.npy",
        dtype=np.uint8,
        mode="r",
        shape=(len(y), args.t, args.size, args.size, 3),
    )
    X_dp = np.memmap(
        dp_cache / f"train_x_t{args.t}_s{args.size}.npy",
        dtype=np.uint8,
        mode="r",
        shape=(len(y), args.t, args.size, args.size, 3),
    )
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]
    pool_idx = np.where(~np.isin(users, list(HOLD)))[0]
    print(
        f"n={len(y)} hold={len(hold_idx)} pool={len(pool_idx)} device={device} mode={args.mode}",
        flush=True,
    )

    if args.dry_run:
        print("DRY-RUN: build model + 1 batch", flush=True)
        model = build_model(args.mode, pretrained=True).to(device)
        model.freeze_early(args.freeze if args.freeze != "none" else "stem_l1")
        ds = DualCachedClipDataset(X_ir, X_dp, y, users, pool_idx[: max(args.batch_size, 2)], train=True, seed=0)
        xa, xb, yb, _u, _i = next(iter(DataLoader(ds, batch_size=args.batch_size)))
        xa = to_bcthw(xa.to(device))
        xb = to_bcthw(xb.to(device))
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(xa, xb)
            else:
                out = model(xa, xb)
        print(f"dry-run ok out={tuple(out.shape)} fp16_mb={pack_size_mb(model):.1f}", flush=True)
        status = {
            "tag": "earlyfuse_v1_dryrun",
            "mode": args.mode,
            "device": str(device),
            "out_shape": list(out.shape),
            "fp16_mb": pack_size_mb(model),
            "trainable_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
            "total_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "wrote_csv": False,
        }
        (ROOT / "metrics_earlyfuse_v1_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2), flush=True)
        return

    members = []
    for seed in args.seeds:
        ck = ckpt_dir / f"{args.mode}_pool_seed{seed}.pt"
        if ck.exists():
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            acc = float(blob.get("val_acc", -1))
            print(f"reuse {ck.name} val={acc}", flush=True)
            hold_logits = blob.get("hold_logits")
            hold_users = blob.get("hold_users")
            hold_y = blob.get("hold_y")
            nested = float(blob.get("val_nested", -1))
            if hold_logits is None:
                model = build_model(args.mode, pretrained=False)
                model.load_state_dict(blob["model"])
                model.to(device)
                val_ds = DualCachedClipDataset(X_ir, X_dp, y, users, hold_idx, train=False, seed=0)
                val_loader = DataLoader(val_ds, batch_size=max(args.batch_size, 4), shuffle=False, num_workers=0)
                hold_m = evaluate(model, val_loader, device, args.mode)
                hold_logits, hold_users, hold_y = hold_m["logits"], hold_m["users"], hold_m["y"]
                nested = hold_m["nested_louo"]
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            members.append(
                {
                    "seed": seed,
                    "acc": acc,
                    "nested": nested,
                    "hold_logits": np.asarray(hold_logits),
                    "hold_users": np.asarray(hold_users) if hold_users is not None else users[hold_idx],
                    "hold_y": np.asarray(hold_y) if hold_y is not None else y[hold_idx],
                    "ckpt": ck,
                    "per_user": blob.get("per_user", {}),
                }
            )
            continue
        if args.skip_train:
            print(f"missing {ck}", flush=True)
            continue
        freeze = args.freeze if args.freeze != "none" else "stem_l1"
        acc, state, hist, hold_m = train_one(
            X_ir,
            X_dp,
            y,
            users,
            pool_idx,
            hold_idx,
            device,
            seed,
            args.epochs,
            args.batch_size,
            args.lr,
            args.patience,
            ck,
            args.mode,
            mixup_alpha=args.mixup,
            unfreeze_ep=args.unfreeze_ep,
            freeze_which=freeze,
        )
        members.append(
            {
                "seed": seed,
                "acc": acc,
                "nested": hold_m["nested_louo"],
                "hold_logits": hold_m["logits"],
                "hold_users": hold_m["users"],
                "hold_y": hold_m["y"],
                "ckpt": ck,
                "per_user": hold_m["per_user"],
            }
        )

    if not members:
        status = {"tag": "earlyfuse_v1", "error": "no members", "wrote_csv": False}
        (ROOT / "metrics_earlyfuse_v1_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(status, flush=True)
        return

    hold_stack = np.stack([m["hold_logits"] for m in members], 0)
    hold_ens = hold_stack.mean(0)
    yt = members[0]["hold_y"]
    yu = members[0]["hold_users"]
    ens_pred = hold_ens.argmax(1)
    hold_acc = float((ens_pred == yt).mean())
    per_user = {}
    for leave in sorted(set(yu.tolist())):
        msk = yu == leave
        per_user[int(leave)] = float((ens_pred[msk] == yt[msk]).mean())
    nested = float(np.mean(list(per_user.values())))

    # disagree vs v7
    v7_ens, v7_y = load_v7_hold_preds()
    disagree = -1
    if v7_ens is not None and len(v7_ens) == len(ens_pred):
        v7_pred = v7_ens.argmax(1)
        disagree = int((v7_pred != ens_pred).sum())
        print(f"disagree vs v7 logits ens: {disagree}", flush=True)
    else:
        # try reading submission alignment via IR v7 write artifacts
        print("v7 hold logits not found for disagree; will compute vs IR solo ens if present", flush=True)

    np.savez(
        ckpt_dir / f"hold_logits_{args.mode}.npz",
        logits=hold_stack,
        ens=hold_ens,
        y=yt,
        users=yu,
        hold_idx=hold_idx,
        seeds=np.array([m["seed"] for m in members]),
        scores=np.array([m["acc"] for m in members]),
    )

    if args.infer_test and device.type == "cuda":
        Xt_ir = np.memmap(
            ir_cache / f"test_x_t{args.t}_s{args.size}.npy",
            dtype=np.uint8,
            mode="r",
            shape=(405, args.t, args.size, args.size, 3),
        )
        Xt_dp = np.memmap(
            dp_cache / f"test_x_t{args.t}_s{args.size}.npy",
            dtype=np.uint8,
            mode="r",
            shape=(405, args.t, args.size, args.size, 3),
        )
        test_logs = []
        for m in members:
            blob = torch.load(m["ckpt"], map_location="cpu", weights_only=False)
            model = build_model(args.mode, pretrained=False)
            model.load_state_dict(blob["model"])
            model.to(device)
            base = infer_pair(model, Xt_ir, Xt_dp, device, bs=max(args.batch_size, 4), flip=False)
            flip = infer_pair(model, Xt_ir, Xt_dp, device, bs=max(args.batch_size, 4), flip=True)
            tta = 0.5 * (base + flip)
            np.save(ckpt_dir / f"test_logits_seed{m['seed']}.npy", base)
            np.save(ckpt_dir / f"test_logits_seed{m['seed']}_tta.npy", tta)
            test_logs.append(tta)
            print(f"inferred test seed{m['seed']}", flush=True)
            del model
            torch.cuda.empty_cache()
        test_ens = np.mean(test_logs, 0)
        np.save(ckpt_dir / "test_logits_ens.npy", test_ens)

    clear_win = hold_acc >= GATE_MIN_HOLD and nested >= GATE_MIN_HOLD and disagree >= GATE_MIN_DISAGREE
    metrics = {
        "tag": "earlyfuse_v1",
        "mode": args.mode,
        "member_scores": {f"seed{m['seed']}": {"hold": m["acc"], "nested": m["nested"], "per_user": m["per_user"]} for m in members},
        "holdout_acc_ensemble": hold_acc,
        "nested_louo_ensemble": nested,
        "per_user": per_user,
        "disagree_vs_v7": disagree,
        "gate": {
            "min_hold": GATE_MIN_HOLD,
            "min_nested": GATE_MIN_HOLD,
            "min_disagree": GATE_MIN_DISAGREE,
            "clear_win": clear_win,
        },
        "ir_v7_hold": IR_V7_HOLD,
        "delta_vs_v7_hold": hold_acc - IR_V7_HOLD,
        "wrote_csv": False,
        "submit": "DO NOT - keep ir_v7" if not clear_win else "CANDIDATE - write CSV",
        "ckpt_dir": str(ckpt_dir),
        "fp16_pack_note": "shared/hybrid/ch6 ~60MB fp16 legal; dual~120MB illegal — prefer shared",
    }
    (ROOT / "metrics_earlyfuse_v1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "metrics_earlyfuse_v1_status.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    if clear_win:
        print("GATE CLEARED — parent/write step should emit submission CSV", flush=True)
    else:
        print("GATE NOT CLEARED — status only, no CSV promotion", flush=True)


if __name__ == "__main__":
    main()
