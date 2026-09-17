"""CPU infer test logits for thermal v6 seed3141 while GPU trains other seeds."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from dataset import NUM_CLASSES

ROOT = Path(__file__).resolve().parent
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)

def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

def normalize(x):
    return (x - K_MEAN.to(x.device)) / K_STD.to(x.device)

@torch.no_grad()
def infer(model, Xt, bs=4):
    model.eval()
    out = np.zeros((len(Xt), NUM_CLASSES), np.float32)
    for i in range(0, len(Xt), bs):
        arr = Xt[i:i+bs].astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3)))
        x = normalize(x).permute(0, 2, 1, 3, 4).contiguous()
        out[i:i+len(x)] = model(x).float().numpy()
        if i % 40 == 0:
            print(f"infer {i}/{len(Xt)}", flush=True)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3141)
    args = ap.parse_args()
    ck = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v6_lift" / f"pool_seed{args.seed}.pt"
    outp = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v6_lift" / f"test_logits_seed{args.seed}.npy"
    if outp.exists():
        print(f"exists {outp}", flush=True)
        return
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model = build()
    model.load_state_dict(blob["model"])
    model.to("cpu")
    Xt = np.memmap(ROOT / "cache" / "thermal_yolo" / "test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(405, 16, 112, 112, 3))
    print(f"seed{args.seed} val={blob['val_acc']:.4f} CPU infer", flush=True)
    test = infer(model, Xt, bs=4)
    np.save(outp, test)
    print(f"saved {outp} {test.shape}", flush=True)

if __name__ == "__main__":
    main()
