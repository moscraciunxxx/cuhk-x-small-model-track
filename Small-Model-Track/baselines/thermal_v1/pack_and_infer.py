"""Pack fp16 ckpt <100MB and write submission from thermal_yolo_r2p1d18 holdout_train.pt."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
K_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1,1,3,1,1)
K_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1,1,3,1,1)

def build():
    m = r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 40)
    return m

def main():
    ckpt_dir = ROOT / "checkpoints" / "thermal_yolo_r2p1d18"
    ckpt = ckpt_dir / "holdout_train.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("val_acc", blob.get("val_acc"), "epoch", blob.get("epoch"))
    sd = blob["model"]
    sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    pack = ckpt_dir / "model_fp16.pt"
    torch.save({"model_fp16": sd16, "val_acc": blob.get("val_acc"), "epoch": blob.get("epoch"), "num_classes": 40}, pack)
    print("fp16 pack mb", pack.stat().st_size / (1024*1024))
    # also yolo weights size
    yolo = ROOT / "yolov8n.pt"
    print("yolo mb", yolo.stat().st_size/(1024*1024) if yolo.exists() else None)
    print("total approx", pack.stat().st_size/(1024*1024) + (yolo.stat().st_size/(1024*1024) if yolo.exists() else 0))

    cache = ROOT / "cache" / "thermal_yolo"
    meta = json.loads((cache/"test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache/"test_empty.json").read_text(encoding="utf-8")))
    Xt = np.memmap(cache/"test_x_t16_s112.npy", dtype=np.uint8, mode="r", shape=(len(meta),16,112,112,3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build().to(device)
    model.load_state_dict(sd)
    model.eval()
    logits = np.zeros((len(meta),40), np.float32)
    with torch.no_grad():
        for i in range(0, len(meta), 8):
            arr = Xt[i:i+8].astype(np.float32)/255.0
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0,1,4,2,3))).to(device)
            x = (x - K_MEAN.to(device))/K_STD.to(device)
            x = x.permute(0,2,1,3,4).contiguous()
            logits[i:i+len(x)] = model(x).float().cpu().numpy()
    preds = logits.argmax(1)
    fb={}
    with open(TRACK/"baselines"/"skeleton_imu_v2"/"submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/")+"/"]=int(row["prediction"])
    out = ROOT/"submission_thermal_v1.csv"
    nfb=0
    with out.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","prediction"])
        for i,m in enumerate(meta):
            path=m["path"] if m["path"].endswith("/") else m["path"]+"/"
            if m.get("empty") or m["sample_id"] in empty:
                pred=fb.get(path,int(preds[i])); nfb+=1
            else:
                pred=int(preds[i])
            w.writerow([path,pred])
    np.save(ckpt_dir/"test_logits.npy", logits)
    print("wrote", out, "fallback", nfb, "val_acc", blob.get("val_acc"))
    # promote
    if float(blob.get("val_acc") or 0) >= 0.55:
        dest = TRACK/"submission.csv"
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print("PROMOTED", dest)
    # blend with midfuse
    mid = np.load(TRACK/"baselines"/"depth_color_v1"/"cache"/"midfuse_test_logits.npy")
    def sm(z,T=1.0):
        z=z/T; z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
    # use holdout-tuned w from video strength — prefer video since 0.57 > mid 0.52
    for w,T,name in [(0.7,1.0,"w07"), (0.85,1.0,"w085"), (1.0,1.0,"video")]:
        fused=(w*sm(logits,T)+(1-w)*sm(mid,T)).argmax(1)
        p=ROOT/f"submission_thermal_midfuse_{name}.csv"
        with p.open("w",newline="",encoding="utf-8") as f:
            wr=csv.writer(f); wr.writerow(["path","prediction"])
            for i,m in enumerate(meta):
                path=m["path"] if m["path"].endswith("/") else m["path"]+"/"
                if m.get("empty") or m["sample_id"] in empty:
                    pred=fb.get(path,int(fused[i]))
                else:
                    pred=int(fused[i])
                wr.writerow([path,pred])
        print("wrote", p)

if __name__=="__main__":
    main()
