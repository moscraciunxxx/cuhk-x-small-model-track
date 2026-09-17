"""Infer test logits for T24 focal_ft pool seeds. GPU required."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from dataset import NUM_CLASSES

ROOT = Path(__file__).resolve().parent
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)

def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

@torch.no_grad()
def infer_array(model, Xt, device, bs=4):
    model.eval()
    n = len(Xt)
    o = np.zeros((n, NUM_CLASSES), np.float32)
    for i in range(0, n, bs):
        arr = Xt[i:i + bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(device)  # B,T,H,W,C
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # B,T,C,H,W
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        o[i:i + x.size(0)] = model(x).float().cpu().numpy()
        if (i // bs) % 20 == 0:
            print(f"  infer {i}/{n}", flush=True)
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, required=True)
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--t", type=int, default=24)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    cache = Path(args.cache_dir); ckpt = Path(args.ckpt_dir)
    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    Xt = np.memmap(cache / f"test_x_t{args.t}_s112.npy", dtype=np.uint8, mode="r",
                   shape=(len(meta), args.t, 112, 112, 3))
    device = torch.device("cuda")
    seeds = []
    holds = []
    for p in sorted(ckpt.glob("pool_seed*.pt")):
        seed = int(p.stem.replace("pool_seed", ""))
        out_p = ckpt / f"test_logits_seed{seed}.npy"
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if out_p.exists():
            print("reuse test", out_p.name, flush=True)
            seeds.append(seed)
            holds.append(np.load(out_p))
            continue
        print(f"infer seed{seed} val_acc={blob.get('val_acc')}", flush=True)
        model = build().to(device)
        model.load_state_dict(blob["model"])
        o = infer_array(model, Xt, device, bs=args.bs)
        np.save(out_p, o)
        seeds.append(seed); holds.append(o)
        del model; torch.cuda.empty_cache()
        print(f"saved {out_p.name}", flush=True)
    if not holds:
        raise SystemExit("no pool seeds")
    ens = np.mean(holds, 0).astype(np.float32)
    np.save(ckpt / "test_logits_ens.npy", ens)
    print("ENSEMBLE_TEST seeds", seeds, "shape", ens.shape, flush=True)
    (ckpt / "test_infer_report.json").write_text(
        json.dumps({"seeds": seeds, "n_test": int(ens.shape[0])}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
