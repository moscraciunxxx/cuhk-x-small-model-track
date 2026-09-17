import csv, json
from pathlib import Path
import numpy as np

DC = Path(r"D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1")
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")

def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

mid_test = np.load(DC / "cache" / "midfuse_test_logits.npy")
vid_test = np.load(DC / "checkpoints" / "depth_color" / "test_logits.npy")
w, T = 0.6, 2.0
fused = (w * softmax(vid_test, T) + (1 - w) * softmax(mid_test, T)).argmax(1)
mid_only = mid_test.argmax(1)
pm, pv = softmax(mid_test, 1.0), softmax(vid_test, 1.0)
conf_gate = []
for i in range(len(fused)):
    if pv[i].max() > 0.35 and pv[i].max() > pm[i].max() + 0.05:
        conf_gate.append(int(pv[i].argmax()))
    else:
        conf_gate.append(int(pm[i].argmax()))
conf_gate = np.array(conf_gate)
test_meta = json.loads((DC / "cache" / "depth_color" / "test_meta.json").read_text(encoding="utf-8"))

def write_csv(path, preds):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "prediction"])
        for meta, pred in zip(test_meta, preds):
            pth = meta["path"] if meta["path"].endswith("/") else meta["path"] + "/"
            wr.writerow([pth, int(pred)])
    print("wrote", path)

write_csv(DC / "submission_blend_depth_midfuse.csv", fused)
write_csv(DC / "submission_midfuse_testlogits.csv", mid_only)
write_csv(DC / "submission_confgate_depth_midfuse.csv", conf_gate)

ens = {}
with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
    for row in csv.DictReader(f):
        ens[row["path"].rstrip("/") + "/"] = int(row["prediction"])
for name, preds in [("blend", fused), ("mid", mid_only), ("gate", conf_gate)]:
    agree = sum(
        1
        for meta, pred in zip(test_meta, preds)
        if ens.get(meta["path"] if meta["path"].endswith("/") else meta["path"] + "/") == int(pred)
    )
    print(name, "agree ensemble", agree, "/", 405)
