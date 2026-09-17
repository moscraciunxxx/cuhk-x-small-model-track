"""ir_v22: temporal-stride / multi-clip TTA on EXISTING classic9 R2+1D ckpts only.
No new train. Prefer nested sameT fuse with th_v6_v2trio + mid_ens4 (ir_v21b recipe).
Avoid perT / per-member overfit. Write metrics_ir_v22_status.json; CSV only if gate clears.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from torchvision.models.video import r2plus1d_18

from dataset import DEFAULT_HOLD_OUT_USERS, NUM_CLASSES, list_frame_paths
from fuse_ir_v9 import load_members, softmax_np, nested_fixed, fuse3_sameT
from probe_ir_v18 import (
    apply_cfg,
    preds_full,
    nested_retune,
    load_ir_pools,
    V7_CFG,
    V7_HOLD,
    GATE,
    MIN_DISAGREE,
)

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
PT = timezone(timedelta(hours=-7))
HOLD = set(DEFAULT_HOLD_OUT_USERS)
T_FRAMES = 16
SIZE = 112
CONF = 0.10

K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

CLASSIC9 = [
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed42.pt", "pool_seed42"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed2024.pt", "pool_seed2024"),
    ("checkpoints/ir_yolo_r2p1d18_v6/pool_seed777.pt", "pool_seed777"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed11.pt", "pool_seed11"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed99.pt", "pool_seed99"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed123.pt", "pool_seed123"),
    ("checkpoints/ir_yolo_r2p1d18_v5/pool_seed7.pt", "pool_seed7"),
    ("checkpoints/ir_yolo_r2p1d18_v6/pool_seed333.pt", "pool_seed333"),
    ("checkpoints/ir_yolo_r2p1d18_v6/pool_seed1.pt", "pool_seed1"),
]

# temporal windows: start_frac, end_frac (inclusive span of timeline)
TEMP_VIEWS = {
    "full": (0.00, 1.00),
    "early": (0.00, 0.82),
    "late": (0.18, 1.00),
    "center": (0.10, 0.90),
    "stride0": ("stride", 0),
    "stride1": ("stride", 1),
}


def now_pt() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")


def build(pretrained: bool = False):
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)


def load_state(ckpt_path: Path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"], float(blob.get("val_acc", -1))
    return blob, -1.0


def open_rgb(path: Path, modality: str = "IR") -> Image.Image:
    img = Image.open(path)
    if modality.lower() == "ir" or img.mode == "L":
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return img.convert("RGB")


def motion_box(frame_paths: list[Path], idxs: list[int]):
    try:
        a = np.asarray(open_rgb(frame_paths[idxs[0]]).convert("L"), dtype=np.float32)
        b = np.asarray(open_rgb(frame_paths[idxs[-1]]).convert("L"), dtype=np.float32)
        if a.shape != b.shape:
            return None
        d = np.abs(a - b)
        if float(d.mean()) < 1.5:
            return None
        thr = np.percentile(d, 85)
        ys, xs = np.where(d >= thr)
        if len(xs) < 50:
            return None
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        h, w = a.shape
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - 0.2 * bw)
        y1 = max(0, y1 - 0.2 * bh)
        x2 = min(w - 1, x2 + 0.2 * bw)
        y2 = min(h - 1, y2 + 0.2 * bh)
        return x1 / w, y1 / h, x2 / w, y2 / h
    except Exception:
        return None


def detect_norm_box(frame_paths: list[Path], model: YOLO, conf: float):
    n = len(frame_paths)
    if n == 0:
        return (0.075, 0.075, 0.925, 0.925), "empty"
    idxs = sorted(set([n // 4, n // 2, (3 * n) // 4, max(0, n - 1)]))
    best = None
    for i in idxs:
        try:
            img = open_rgb(frame_paths[i])
        except Exception:
            continue
        w, h = img.size
        res = model.predict(img, classes=[0], verbose=False, conf=conf)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        score = confs * np.sqrt(np.maximum(areas, 1.0))
        j = int(score.argmax())
        x1, y1, x2, y2 = xyxy[j]
        cand = (float(confs[j]), x1 / w, y1 / h, x2 / w, y2 / h)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is not None:
        return best[1:], "yolo"
    mb = motion_box(frame_paths, idxs)
    if mb is not None:
        return mb, "motion"
    return (0.075, 0.075, 0.925, 0.925), "center"


def apply_box(img: Image.Image, nx1, ny1, nx2, ny2, pad: float = 0.15) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = nx1 * w, ny1 * h, nx2 * w, ny2 * h
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    x1 = max(0, x1 - pad * bw)
    y1 = max(0, y1 - pad * bh)
    x2 = min(w, x2 + pad * bw)
    y2 = min(h, y2 + pad * bh)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 16)
    half = side / 2
    left = int(max(0, min(w - side, cx - half)))
    top = int(max(0, min(h - side, cy - half)))
    side = int(min(side, w - left, h - top))
    return img.crop((left, top, left + side, top + side))


def temporal_indices(n: int, t: int, mode: str) -> np.ndarray:
    if n <= 0:
        return np.zeros(t, dtype=np.int64)
    if n == 1:
        return np.zeros(t, dtype=np.int64)
    spec = TEMP_VIEWS[mode]
    if spec[0] == "stride":
        offset = int(spec[1])
        if n >= t * 2:
            idx = np.arange(offset, offset + t * 2, 2)[:t]
            return np.clip(idx, 0, n - 1).astype(np.int64)
        # short clip: shift linspace start
        start = 0.0 if offset == 0 else 0.12
        end = 0.88 if offset == 0 else 1.0
        return (np.linspace(start, end, t) * (n - 1)).astype(np.int64)
    start, end = float(spec[0]), float(spec[1])
    if end <= start + 0.05:
        start, end = 0.0, 1.0
    return (np.linspace(start, end, t) * (n - 1)).astype(np.int64)


def render_clip(frame_paths: list[Path], box, mode: str, t: int = T_FRAMES, size: int = SIZE) -> np.ndarray:
    out = np.zeros((t, size, size, 3), dtype=np.uint8)
    n = len(frame_paths)
    if n == 0:
        return out
    idxs = temporal_indices(n, t, mode)
    nx1, ny1, nx2, ny2 = box
    for i, fi in enumerate(idxs):
        try:
            img = open_rgb(frame_paths[int(fi)])
            crop = apply_box(img, nx1, ny1, nx2, ny2).resize((size, size), Image.BILINEAR)
            out[i] = np.asarray(crop, dtype=np.uint8)
        except Exception:
            continue
    return out


def inventory_ckpts() -> list[dict[str, Any]]:
    rows = []
    for rel, tag in CLASSIC9:
        p = ROOT / rel
        rows.append({
            "tag": tag,
            "path": str(p),
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None,
            "infer_frames": T_FRAMES,
            "infer_size": SIZE,
            "train_sample": "linspace_full_clip_yolo_crop_cached_ir_yolo_v4",
        })
    return rows


def ensure_boxes(cache: Path, out_dir: Path, force: bool = False) -> dict[str, Any]:
    box_path = out_dir / "boxes_hold_test.npz"
    if box_path.exists() and not force:
        z = np.load(box_path, allow_pickle=True)
        print(f"[boxes] reuse {box_path}", flush=True)
        return {
            "hold_boxes": z["hold_boxes"],
            "hold_src": z["hold_src"],
            "test_boxes": z["test_boxes"],
            "test_src": z["test_src"],
            "hold_idx": z["hold_idx"],
        }

    train_meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    test_meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    users = np.load(cache / "train_users.npy")
    hold_idx = np.where(np.isin(users, list(HOLD)))[0]

    yolo = YOLO(str(ROOT / "yolov8n.pt"))
    yolo.predict(np.zeros((112, 112, 3), dtype=np.uint8), classes=[0], verbose=False, device=0)

    hold_boxes = np.zeros((len(hold_idx), 4), np.float32)
    hold_src = []
    print(f"[boxes] detecting hold n={len(hold_idx)} ...", flush=True)
    t0 = time.time()
    for j, gi in enumerate(hold_idx):
        frames = list_frame_paths(Path(train_meta[int(gi)]["clip_dir"]), "IR")
        box, src = detect_norm_box(frames, yolo, CONF)
        hold_boxes[j] = box
        hold_src.append(src)
        if (j + 1) % 50 == 0:
            print(f"  hold {j+1}/{len(hold_idx)} elapsed={time.time()-t0:.0f}s", flush=True)

    test_boxes = np.zeros((len(test_meta), 4), np.float32)
    test_src = []
    print(f"[boxes] detecting test n={len(test_meta)} ...", flush=True)
    for j, m in enumerate(test_meta):
        frames = list_frame_paths(Path(m["clip_dir"]), "IR") if not m.get("empty") else []
        box, src = detect_norm_box(frames, yolo, CONF)
        test_boxes[j] = box
        test_src.append(src)
        if (j + 1) % 50 == 0:
            print(f"  test {j+1}/{len(test_meta)} elapsed={time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(
        box_path,
        hold_boxes=hold_boxes,
        hold_src=np.array(hold_src),
        test_boxes=test_boxes,
        test_src=np.array(test_src),
        hold_idx=hold_idx,
    )
    print(f"[boxes] saved {box_path} in {time.time()-t0:.1f}s src_hold={ {k:hold_src.count(k) for k in set(hold_src)} }", flush=True)
    del yolo
    torch.cuda.empty_cache()
    return {
        "hold_boxes": hold_boxes,
        "hold_src": np.array(hold_src),
        "test_boxes": test_boxes,
        "test_src": np.array(test_src),
        "hold_idx": hold_idx,
    }


def build_view_arrays(cache: Path, out_dir: Path, boxes: dict, view_names: list[str], force: bool = False):
    """Build uint8 memmaps: hold_x_{view}.npy / test_x_{view}.npy"""
    train_meta = json.loads((cache / "train_meta.json").read_text(encoding="utf-8"))
    test_meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    hold_idx = boxes["hold_idx"]
    n_h, n_t = len(hold_idx), len(test_meta)

    # reuse official cache as 'full' (exact train-time sampling)
    X_full = np.memmap(cache / f"train_x_t{T_FRAMES}_s{SIZE}.npy", dtype=np.uint8, mode="r",
                       shape=(len(train_meta), T_FRAMES, SIZE, SIZE, 3))
    Xt_full = np.memmap(cache / f"test_x_t{T_FRAMES}_s{SIZE}.npy", dtype=np.uint8, mode="r",
                        shape=(n_t, T_FRAMES, SIZE, SIZE, 3))

    built = {}
    for view in view_names:
        hp = out_dir / f"hold_x_{view}.npy"
        tp = out_dir / f"test_x_{view}.npy"
        if view == "full":
            if force or not hp.exists():
                arr = np.asarray(X_full[hold_idx])
                np.save(hp, arr)
                print(f"[views] wrote {hp.name} from cache hold {arr.shape}", flush=True)
            if force or not tp.exists():
                # memmap save via copy
                arrt = np.asarray(Xt_full)
                np.save(tp, arrt)
                print(f"[views] wrote {tp.name} from cache test {arrt.shape}", flush=True)
            built[view] = (hp, tp)
            continue

        if hp.exists() and tp.exists() and not force:
            print(f"[views] reuse {view}", flush=True)
            built[view] = (hp, tp)
            continue

        print(f"[views] building temporal view={view} ...", flush=True)
        t0 = time.time()
        Hh = np.lib.format.open_memmap(str(hp), mode="w+", dtype=np.uint8,
                                       shape=(n_h, T_FRAMES, SIZE, SIZE, 3))
        for j, gi in enumerate(hold_idx):
            frames = list_frame_paths(Path(train_meta[int(gi)]["clip_dir"]), "IR")
            Hh[j] = render_clip(frames, boxes["hold_boxes"][j], view)
            if (j + 1) % 50 == 0:
                Hh.flush()
                print(f"  hold {view} {j+1}/{n_h} {time.time()-t0:.0f}s", flush=True)
        Hh.flush()
        Ht = np.lib.format.open_memmap(str(tp), mode="w+", dtype=np.uint8,
                                       shape=(n_t, T_FRAMES, SIZE, SIZE, 3))
        for j, m in enumerate(test_meta):
            frames = list_frame_paths(Path(m["clip_dir"]), "IR") if not m.get("empty") else []
            Ht[j] = render_clip(frames, boxes["test_boxes"][j], view)
            if (j + 1) % 50 == 0:
                Ht.flush()
                print(f"  test {view} {j+1}/{n_t} {time.time()-t0:.0f}s", flush=True)
        Ht.flush()
        del Hh, Ht
        print(f"[views] done {view} in {time.time()-t0:.1f}s", flush=True)
        built[view] = (hp, tp)
    return built


@torch.no_grad()
def infer_uint8(model, X_uint8, device, bs=3, do_flip=False) -> np.ndarray:
    n = len(X_uint8)
    out = np.zeros((n, NUM_CLASSES), np.float32)
    for i0 in range(0, n, bs):
        arr = np.asarray(X_uint8[i0:i0 + bs]).astype(np.float32) / 255.0  # B,T,H,W,C
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)  # B,T,C,H,W
        xn = normalize(x).permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,H,W
        logits = model(xn).float()
        if do_flip:
            xf = torch.flip(x, dims=[-1])
            xfn = normalize(xf).permute(0, 2, 1, 3, 4).contiguous()
            logits = 0.5 * (logits + model(xfn).float())
        out[i0:i0 + len(x)] = logits.cpu().numpy()
    return out


def run_infer(device, view_paths: dict, out_dir: Path, yt: np.ndarray, force: bool = False):
    """Per-seed: average logits across views (+flip). Return members list."""
    members = []
    view_names = list(view_paths.keys())
    # materialize hold views once
    hold_Xs = {v: np.load(view_paths[v][0], mmap_mode="r") for v in view_names}
    test_Xs = {v: np.load(view_paths[v][1], mmap_mode="r") for v in view_names}

    for rel, tag in CLASSIC9:
        hold_p = out_dir / f"hold_temp_{tag}.npy"
        test_p = out_dir / f"test_temp_{tag}.npy"
        meta_p = out_dir / f"meta_{tag}.json"
        if hold_p.exists() and test_p.exists() and not force:
            hl = np.load(hold_p)
            acc = float((hl.argmax(1) == yt).mean())
            print(f"[infer] reuse {tag} acc={acc:.4f}", flush=True)
            members.append({
                "tag": tag, "logits": hl, "base": hl, "tta": hl,
                "acc": acc, "acc_base": acc, "use_tta": True,
                "test_logits": np.load(test_p),
                "hold_path": str(hold_p), "test_path": str(test_p),
            })
            continue
        ck = ROOT / rel
        if not ck.exists():
            print(f"[infer] MISSING {ck}", flush=True)
            continue
        state, vacc = load_state(ck)
        model = build(False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        t0 = time.time()
        print(f"[infer] {tag} ckpt_val={vacc:.4f} views={view_names} flip=1 ...", flush=True)

        hold_accums = []
        test_accums = []
        per_view = {}
        for v in view_names:
            hv = infer_uint8(model, hold_Xs[v], device, bs=3, do_flip=True)
            tv = infer_uint8(model, test_Xs[v], device, bs=3, do_flip=True)
            hold_accums.append(hv)
            test_accums.append(tv)
            per_view[v] = float((hv.argmax(1) == yt).mean())
            print(f"  view {v} hold_acc={per_view[v]:.4f}", flush=True)

        # also identity-only on full (no flip) for sanity
        h_full = infer_uint8(model, hold_Xs["full"], device, bs=3, do_flip=False)
        full_acc = float((h_full.argmax(1) == yt).mean())

        hl = np.mean(hold_accums, 0).astype(np.float32)
        tl = np.mean(test_accums, 0).astype(np.float32)
        acc = float((hl.argmax(1) == yt).mean())
        np.save(hold_p, hl)
        np.save(test_p, tl)
        meta = {"tag": tag, "acc_temp_mean": acc, "acc_full_nof lip": full_acc, "per_view_flip": per_view,
                "n_views": len(view_names), "took_sec": time.time() - t0}
        # fix typo key
        meta = {"tag": tag, "acc_temp_mean": acc, "acc_full_noflip": full_acc, "per_view_flip": per_view,
                "n_views": len(view_names), "took_sec": time.time() - t0}
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[infer] {tag} temp_mean={acc:.4f} full_noflip={full_acc:.4f} took={meta['took_sec']:.1f}s", flush=True)
        members.append({
            "tag": tag, "logits": hl, "base": h_full, "tta": hl,
            "acc": acc, "acc_base": full_acc, "use_tta": True,
            "test_logits": tl, "hold_path": str(hold_p), "test_path": str(test_p),
            "per_view": per_view,
        })
        del model
        torch.cuda.empty_cache()
        gc.collect()
    return members


def fuse_and_report(temp_members, inventory, t0, gpu_note):
    # baseline classic9 from existing selective members
    members_v7, yt, yu = load_members()
    pools_base, c9 = load_ir_pools(members_v7, yt, yu)
    ir_base = pools_base["classic9_base"]

    # temporal pools
    temp_sorted = sorted(temp_members, key=lambda d: -d["acc"])
    ir_temp = np.mean([m["logits"] for m in temp_sorted], 0).astype(np.float32)
    # mix: 0.5 base + 0.5 temp (stabilize)
    ir_mix = (0.5 * ir_base + 0.5 * ir_temp).astype(np.float32)
    # only top6 temp
    ir_temp6 = np.mean([m["logits"] for m in temp_sorted[:6]], 0).astype(np.float32)

    th = np.load(ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "hold_thermal_v6.npy").astype(np.float32)
    mid = np.load(ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits_ens4_bonetcn.npy").astype(np.float32)
    tu = np.load(ROOT / "cache" / "ir_yolo_v4" / "train_users.npy")
    hold_idx = np.where(np.isin(tu, list(HOLD)))[0]
    assert len(hold_idx) == len(yt)
    mid_h = mid[hold_idx]
    mask = th.any(1) & mid_h.any(1)

    ir_pools = {
        "classic9_base": ir_base,
        "classic9_temp": ir_temp,
        "classic9_mix50": ir_mix,
        "classic9_temp_top6": ir_temp6,
    }
    # also soft-avg temp members
    sm = np.mean([softmax_np(m["logits"]) for m in temp_sorted], 0)
    ir_pools["classic9_temp_sm"] = np.log(np.clip(sm, 1e-8, 1)).astype(np.float32)

    th_pools = {
        "th_v6_v2trio": th,
        "th_v6_soft": (th / 1.8).astype(np.float32),
    }
    mid_pools = {
        "mid_ens4_bonetcn": mid_h,
    }
    mid_v4 = ROOT / "cache" / "ir_yolo_v4" / "midfuse_aligned_train_logits.npy"
    if mid_v4.exists():
        mid_pools["mid_ir_v4"] = np.load(mid_v4)[hold_idx].astype(np.float32)

    Ts = [1.0, 1.5, 2.0, 2.5, 3.0]
    rows = []
    focus = [
        ("classic9_temp", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_mix50", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_temp_sm", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_temp_top6", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_base", "th_v6_v2trio", "mid_ens4_bonetcn"),
        ("classic9_temp", "th_v6_soft", "mid_ens4_bonetcn"),
        ("classic9_mix50", "th_v6_soft", "mid_ens4_bonetcn"),
    ]
    if "mid_ir_v4" in mid_pools:
        focus.append(("classic9_temp", "th_v6_v2trio", "mid_ir_v4"))
        focus.append(("classic9_mix50", "th_v6_v2trio", "mid_ir_v4"))

    v7_full, _ = apply_cfg(ir_base, th, mid_pools.get("mid_ir_v4", mid_h), yt, mask, V7_CFG)
    v7_nest = nested_fixed(ir_base, th, mid_pools.get("mid_ir_v4", mid_h), yt, yu, mask, V7_CFG)
    v7_preds = preds_full(ir_base, th, mid_pools.get("mid_ir_v4", mid_h), mask, V7_CFG)

    for ir_n, th_n, md_n in focus:
        if ir_n not in ir_pools or th_n not in th_pools or md_n not in mid_pools:
            continue
        a, b, c = ir_pools[ir_n], th_pools[th_n], mid_pools[md_n]
        full_acc, cfg = fuse3_sameT(a, b, c, yt, mask, Ts, ngrid=21)
        nest = nested_retune(a, b, c, yt, yu, mask, Ts, ngrid=21)
        preds = preds_full(a, b, c, mask, cfg)
        dis = int(((preds != v7_preds) & mask & (v7_preds >= 0)).sum())
        row = {
            "ir": ir_n, "th": th_n, "mid": md_n, "mode": "sameT",
            "full": float(full_acc), "honest_nested": float(nest["mean"]),
            "folds": nest["folds"], "cfg": cfg, "disagree_vs_v7": dis,
            "clears": bool(full_acc >= GATE and nest["mean"] >= GATE and dis >= MIN_DISAGREE),
            "ir_solo": float((a.argmax(1) == yt).mean()),
        }
        rows.append(row)
        print(f"[fuse] {ir_n}|{th_n}|{md_n} full={row['full']:.4f} nested={row['honest_nested']:.4f} dis={dis} soloIR={row['ir_solo']:.4f}", flush=True)

    rows = sorted(rows, key=lambda d: (-d["honest_nested"], -d["full"]))
    best = rows[0] if rows else None

    # solo IR comparison
    solo = {
        "classic9_base": float((ir_base.argmax(1) == yt).mean()),
        "classic9_temp": float((ir_temp.argmax(1) == yt).mean()),
        "classic9_mix50": float((ir_mix.argmax(1) == yt).mean()),
        "per_seed_temp": {m["tag"]: float(m["acc"]) for m in temp_sorted},
        "per_seed_full_noflip": {m["tag"]: float(m["acc_base"]) for m in temp_sorted},
    }

    gap = None if best is None else float(GATE - best["honest_nested"])
    clears = bool(best and best["clears"])
    wrote_csv = False
    csv_path = None

    if clears:
        # build test submission
        te_ir = np.mean([m["test_logits"] for m in temp_sorted], 0).astype(np.float32)
        # map th/mid test
        th_test_cands = [
            ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy",
            ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy",
            ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5" / "test_logits_ens_v6.npy",
        ]
        # Use v7 recipe thermal test if available via prior submissions tooling — prefer thermal ens
        th_test = None
        for p in th_test_cands:
            if p.exists() and "thermal" in str(p):
                th_test = np.load(p).astype(np.float32)
                break
        mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy").astype(np.float32)
        # fallback thermal from v6 ens path used historically
        if th_test is None:
            # reconstruct from hold-trained trio not available — skip CSV
            print("[csv] missing thermal test logits; skip write despite gate", flush=True)
        else:
            cfg = best["cfg"]
            T = cfg["T"]
            fused = (cfg["wa"] * softmax_np(te_ir, T) + cfg["wb"] * softmax_np(th_test, T)
                     + cfg["wc"] * softmax_np(mid_test, T)).argmax(1)
            cache = ROOT / "cache" / "ir_yolo_v4"
            meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
            empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
            csv_path = ROOT / "submission_ir_v22.csv"
            import csv as csvmod
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csvmod.writer(f)
                w.writerow(["id", "action"])
                for i, m in enumerate(meta):
                    sid = m["sample_id"]
                    pred = int(fused[i])
                    if sid in empty or m.get("empty"):
                        pred = 0
                    w.writerow([sid, pred])
            wrote_csv = True
            print(f"[csv] wrote {csv_path}", flush=True)

    outcome = "WIN" if clears else "MISS"
    next_roi = []
    if clears:
        next_roi = [
            "Submit submission_ir_v22.csv; verify public vs ir_v7 0.69154",
            "If public lifts, freeze recipe; else investigate hold/public gap",
        ]
    else:
        next_roi = [
            f"Temporal TTA miss: best honest_nested={best['honest_nested'] if best else None} vs gate {GATE:.4f} (gap={gap})",
            "IR temporal multi-clip on classic9 appears near ceiling (solo IR ~0.70-0.71)",
            "Optional last Small-track shot: tiny IMU spectrogram as 4th soft stream (skeleton_imu already midfused; spectrogram-only branch not trained)",
            "Else accept Small-track ceiling; keep ir_v7 @ 0.69154; shift ROI to Large track / non-IR modalities with honest nested discipline",
            "STOP: no new IR Kinetics-3D, no Thermal R2+1D seeds, no Mid arch churn, no LOUO class stacks",
        ]

    status = {
        "tag": "ir_v22_temporal_tta",
        "outcome": outcome,
        "keep_ir_v7": not clears,
        "finished_at": now_pt(),
        "gate": {"hold_min": GATE, "nested_min": GATE, "min_disagree": MIN_DISAGREE},
        "v7_reproduce": {"full": float(v7_full), "nested": float(v7_nest["mean"])},
        "best_public": {"csv": "submission_ir_v7.csv", "public": 0.69154, "hold": V7_HOLD},
        "inventory_classic9": inventory,
        "current_infer": {
            "frames": T_FRAMES,
            "size": SIZE,
            "cache": "cache/ir_yolo_v4 train_x_t16_s112 (linspace full-clip YOLO crop)",
            "prior_tta": "v15 spatial mild4/flip on cached T=16 (NOT temporal)",
            "this_tta": {
                "views": list(TEMP_VIEWS.keys()),
                "flip": True,
                "aggregate": "mean logits across views",
                "full_view_source": "existing ir_yolo_v4 cache (exact)",
                "other_views": "YOLO-box cached re-sample with start-offset / stride windows",
            },
        },
        "solo_ir": solo,
        "best": best,
        "top8": rows[:8],
        "wrote_csv": wrote_csv,
        "csv": str(csv_path) if csv_path else None,
        "gap_to_gate_nested": gap,
        "imu_note": (
            "IMU CSVs present; skeleton_imu_v2 already in mid_ens*; "
            "spectrogram-only 4th soft stream NOT trained this turn "
            f"(gap_to_gate={gap}). Train tiny IMU-spec CNN only if parent greenlights."
        ),
        "next_roi": next_roi,
        "gpu_handoff": gpu_note,
        "elapsed_sec": round(time.time() - t0, 1),
        "notes": [
            "Nested sameT only (no perT grid) to avoid ir_v9-style overfit",
            "No new Kinetics IR 3D train; classic9 ckpts only",
            "Prefer th_v6_v2trio + mid_ens4_bonetcn per ir_v21b",
        ],
    }
    outp = ROOT / "metrics_ir_v22_status.json"
    outp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ["tag", "outcome", "best", "gap_to_gate_nested", "next_roi", "gpu_handoff", "elapsed_sec"]}, indent=2), flush=True)
    print(f"wrote {outp}", flush=True)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-boxes", action="store_true")
    ap.add_argument("--force-views", action="store_true")
    ap.add_argument("--force-infer", action="store_true")
    ap.add_argument("--views", nargs="+", default=["full", "early", "late", "center", "stride0", "stride1"])
    ap.add_argument("--fuse-only", action="store_true", help="skip GPU infer; fuse existing temp logits")
    args = ap.parse_args()

    t0 = time.time()
    print(f"ir_v22 temporal-stride TTA start {now_pt()}", flush=True)
    inventory = inventory_ckpts()
    print("inventory:", json.dumps(inventory, indent=2), flush=True)

    cache = ROOT / "cache" / "ir_yolo_v4"
    out_dir = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v22_temp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels
    members_v7, yt, yu = load_members()
    print(f"hold labels n={len(yt)} classic9 inventory ok={all(r['exists'] for r in inventory)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"gpu mem free={free/1e9:.2f}G total={total/1e9:.2f}G", flush=True)

    if not args.fuse_only:
        boxes = ensure_boxes(cache, out_dir, force=args.force_boxes)
        # sanity hold_idx alignment
        assert len(boxes["hold_idx"]) == len(yt)
        view_paths = build_view_arrays(cache, out_dir, boxes, args.views, force=args.force_views)
        temp_members = run_infer(device, view_paths, out_dir, yt, force=args.force_infer)
    else:
        temp_members = []
        for rel, tag in CLASSIC9:
            hp = out_dir / f"hold_temp_{tag}.npy"
            tp = out_dir / f"test_temp_{tag}.npy"
            if hp.exists() and tp.exists():
                hl = np.load(hp)
                temp_members.append({
                    "tag": tag, "logits": hl, "base": hl, "tta": hl,
                    "acc": float((hl.argmax(1) == yt).mean()),
                    "acc_base": float((hl.argmax(1) == yt).mean()),
                    "use_tta": True, "test_logits": np.load(tp),
                })
        print(f"fuse-only loaded {len(temp_members)} seeds", flush=True)

    if len(temp_members) < 5:
        raise RuntimeError(f"too few temp members: {len(temp_members)}")

    gpu_note = {
        "machine": "MosCraciunXXX",
        "machineId": "4ff6e647-6f8c-4d02-88f0-5dda6684ae36",
        "at_start": "idle RTX 3060 6GB taken for TTA infer",
        "at_end": "releasing GPU after ir_v22; yield to LMT if needed",
        "finished_at": now_pt(),
    }
    fuse_and_report(temp_members, inventory, t0, gpu_note)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"DONE {now_pt()} elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
