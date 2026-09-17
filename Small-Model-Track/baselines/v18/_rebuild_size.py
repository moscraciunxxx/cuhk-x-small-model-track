"""Rebuild v18 submission for size-constrained clear-win selection (<=100MB fold ckpts)."""
from __future__ import annotations
import json, sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "skeleton_imu_v2"
V11 = ROOT.parent / "v11"
TRACK = ROOT.parent.parent
sys.path.insert(0, str(V2))
from dataset import load_skel_test_cache

NUM_CLASSES = 40
KEYS = ["kd_mf_v13", "kd_T4_a02", "kd_eq_a02"]
FAMILY = "pow"
# power fit on full non-holdout already stored in metrics params
P_VAL = None  # filled from metrics


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def apply_power_mean(probs_list, p):
    stacked = np.stack(probs_list, 0)
    if abs(p) < 1e-8:
        out = np.exp(np.mean(np.log(np.clip(stacked, 1e-12, 1)), 0))
    else:
        out = np.mean(stacked ** p, 0) ** (1.0 / p)
    out = out / np.maximum(out.sum(1, keepdims=True), 1e-12)
    return out.astype(np.float32)


class ArrayDual(Dataset):
    def __init__(self, xs, xi, flag):
        self.xs = torch.from_numpy(np.asarray(xs, np.float32))
        self.xi = torch.from_numpy(np.asarray(xi, np.float32))
        self.flag = torch.from_numpy(np.asarray(flag, np.float32))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], self.xi[i], self.flag[i]


@torch.no_grad()
def predict_logits_arr(model, xs, xi, flag, device, batch=32):
    ds = ArrayDual(xs, xi, flag)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    outs = []
    for xb, ib, fb in loader:
        outs.append(model(xb.to(device), ib.to(device), fb.to(device)).float().cpu().numpy())
    return np.concatenate(outs, 0)


def load_v2():
    spec = importlib.util.spec_from_file_location("v2m", V2 / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ckpt(path, device, v2m):
    ck = torch.load(path, map_location=device, weights_only=False)
    name = ck.get("model_name", "compact")
    ncls = ck.get("num_classes", NUM_CLASSES)
    if name in ("compact", "compact_fuse"):
        m = v2m.CompactMidFuse(num_classes=ncls)
    else:
        m = v2m.build_model("midfuse", num_classes=ncls)
    m.load_state_dict(ck["model_state"])
    m.to(device).eval()
    return m


def main():
    metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    # find chosen method params
    name = "pow_kd_mf_v13_kd_T4_a02_kd_eq_a02"
    row = next(r for r in metrics["all_methods"] if r["name"] == name)
    p = float(row["params"]["p"])
    print(f"rebuild {name} p={p} nested={row['nested_oof']:.4f} hold={row['holdout']:.4f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2m = load_v2()
    Xte, _ = load_skel_test_cache(V2 / "cache")
    imu_te = np.load(V2 / "cache" / "imu_test.npz")
    Xte_imu = imu_te["X"]
    te_flag = imu_te["has_imu"].astype(np.float32)

    ckpt_dirs = {
        "kd_mf_v13": ROOT / "checkpoints_kd_mf_v13",
        "kd_T4_a02": ROOT / "checkpoints_kd_T4_a02",
        "kd_eq_a02": ROOT / "checkpoints_kd_eq_a02",
    }
    probs = {}
    for k in KEYS:
        accums = []
        for fi in range(5):
            m = load_ckpt(ckpt_dirs[k] / f"best_fold{fi}.pt", device, v2m)
            accums.append(predict_logits_arr(m, Xte, Xte_imu, te_flag, device))
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        probs[k] = softmax(sum(accums) / float(len(accums)))
        print(f"test {k}", flush=True)

    test_probs = apply_power_mean([probs[k] for k in KEYS], p)
    pred = test_probs.argmax(1).astype(int)
    sub = pd.read_csv(V11 / "submission_v11.csv").copy()
    sub.iloc[:, 1] = pred[: len(sub)]
    sub.to_csv(ROOT / "submission_v18_candidate.csv", index=False)
    sub.to_csv(ROOT / "submission_v18.csv", index=False)
    np.savez_compressed(ROOT / "submission_v18_probs.npz", probs=test_probs, pred=pred)
    sub.to_csv(TRACK / "submission.csv", index=False)

    # update metrics summary
    summary = metrics["summary"]
    summary["selected_unconstrained"] = {
        "name": summary["selected"],
        "nested_oof": summary["selected_nested_oof"],
        "holdout": summary["selected_holdout"],
        "note": "max nested among all blends; fold ckpts ~145MB",
    }
    summary["selected"] = name
    summary["selected_nested_oof"] = row["nested_oof"]
    summary["selected_holdout"] = row["holdout"]
    summary["selected_family"] = "pow"
    summary["selected_params"] = row["params"]
    summary["ckpt_size_mb"] = 83.5
    summary["size_constrained"] = True
    summary["clear_win"] = True
    summary["overwrite_submission"] = True
    summary["ping_disk_saver"] = True
    metrics["summary"] = summary
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (TRACK / "submission_README.txt").write_text(
        (
            f"v18 clear-win (size<=100MB): nested OOF {row['nested_oof']:.4f} holdout {row['holdout']:.4f} "
            f"(gate hold>=0.636, or hold>=0.631 & OOF>=0.670)\n"
            f"method={name} params={json.dumps(row['params'])}\n"
            f"fold_ckpt_size_mb=83.5 (unconstrained max-nested was eq_kd_c_kd_a02_kd_mf_v13_kd_mf_a02_kd_eq_a02 "
            f"nested=0.6764 hold=0.6317 but ~145MB)\n"
            f"ping_disk_saver=true.\n"
        ),
        encoding="utf-8",
    )
    print("OVERWROTE submission with size-constrained selection", flush=True)


if __name__ == "__main__":
    main()
