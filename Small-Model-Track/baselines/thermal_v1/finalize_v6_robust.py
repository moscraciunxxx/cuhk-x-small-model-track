"""Finalize robust ir_v6: reject overfit perT; use finer same-T fuse + top4_base IR ens."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TRACK = Path(r"D:\CUHK-X\Small-Model-Track")
V5 = 0.7388663967611336
NUM_CLASSES = 40


def softmax_np(z, T=1.0):
    z = z / float(T)
    z = z - z.max(1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(1, keepdims=True)


def write_sub(path, meta, preds, empty, fb):
    nfb = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "prediction"])
        for i, m in enumerate(meta):
            p = m["path"] if m["path"].endswith("/") else m["path"] + "/"
            if m.get("empty") or m["sample_id"] in empty:
                pred = fb.get(p, int(preds[i])); nfb += 1
            else:
                pred = int(preds[i])
            w.writerow([p, pred])
    return nfb


def main():
    ckpt = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    cache = ROOT / "cache" / "ir_yolo_v4"
    probe = json.loads((ROOT / "metrics_ir_v6_probe.json").read_text(encoding="utf-8"))

    # Robust primary: top4_base same-T triple (hold 0.7449), NOT perT wa=0.1
    cfg = probe["fuse_report"]["top4_base"]["triple"]
    assert cfg["acc"] >= V5 + 0.002, cfg

    # top4 by base acc from member scores in probe / known order
    # From probe: order by acc_base desc: 2024, 11, 42, 99, 7, 123
    member_scores = {
        "pool_seed2024": 0.6792079207920793,
        "pool_seed11": 0.6772277227722773,
        "pool_seed42": 0.6752475247524753,
        "pool_seed99": 0.6673267326732674,
        "pool_seed7": 0.6633663366336634,
        "pool_seed123": 0.6613861386138614,
    }
    top4 = sorted(member_scores, key=member_scores.get, reverse=True)[:4]
    print("top4", top4, flush=True)

    logs = []
    for tag in top4:
        p = ckpt / f"test_logits_{tag}.npy"
        if not p.exists():
            # fallback naming
            seed = tag.replace("pool_seed", "")
            p = ckpt / f"test_logits_seed{seed}.npy"
        logs.append(np.load(p))
        print("loaded", p.name, flush=True)
    ir_test = np.mean(logs, 0).astype(np.float32)

    # also all6 base and selective for alts
    all_tags = sorted(member_scores.keys())
    all_logs = [np.load(ckpt / f"test_logits_{t}.npy") for t in all_tags]
    ir_all = np.mean(all_logs, 0).astype(np.float32)

    # selective test logits from v6 probe write
    sel_files = [
        ckpt / "test_logits_v6_pool_seed11_base.npy",
        ckpt / "test_logits_v6_pool_seed123_tta.npy",
        ckpt / "test_logits_v6_pool_seed2024_base.npy",
        ckpt / "test_logits_v6_pool_seed42_tta.npy",
        ckpt / "test_logits_v6_pool_seed7_base.npy",
        ckpt / "test_logits_v6_pool_seed99_tta.npy",
    ]
    ir_sel = np.mean([np.load(p) for p in sel_files], 0).astype(np.float32)

    mid_test = np.load(TRACK / "baselines" / "depth_color_v1" / "cache" / "midfuse_test_logits.npy")
    th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits.npy"
    if not th_p.exists():
        th_p = ROOT / "checkpoints" / "thermal_yolo_r2p1d18_v3" / "test_logits_final.npy"
    th_test = np.load(th_p)

    def pred3(ir, cfg3):
        T = cfg3["T"]
        return (cfg3["wa"] * softmax_np(ir, T) + cfg3["wb"] * softmax_np(th_test, T) + cfg3["wc"] * softmax_np(mid_test, T)).argmax(1)

    def pred2(a, b, cfg2):
        return (cfg2["w"] * softmax_np(a, cfg2["T"]) + (1 - cfg2["w"]) * softmax_np(b, cfg2["T"])).argmax(1)

    cfg_top4 = probe["fuse_report"]["top4_base"]["triple"]
    cfg_ens = probe["fuse_report"]["ens_base"]["triple"]
    cfg_sel = probe["fuse_report"]["ens_selective_tta"]["triple"]
    cfg_nested = probe["nested"]["nested_mean_cfg"]
    # compromise: average top4 cfg with a bit more mid (public-gap hedge)
    cfg_comp = {
        "wa": 0.45, "wb": 0.30, "wc": 0.25, "T": 1.5,
        "note": "public-gap hedge more Mid vs holdout-opt",
    }
    # geom from probe
    cfg_geom = probe["geom"]

    pred_primary = pred3(ir_test, cfg_top4)
    pred_ens = pred3(ir_all, cfg_ens)
    pred_sel = pred3(ir_sel, cfg_sel)
    pred_nested = pred3(ir_test, {**cfg_nested, "T": cfg_nested["T"]})
    # geom
    T = cfg_geom["T"]
    la = np.log(np.clip(softmax_np(ir_test, T), 1e-8, 1))
    lb = np.log(np.clip(softmax_np(th_test, T), 1e-8, 1))
    lc = np.log(np.clip(softmax_np(mid_test, T), 1e-8, 1))
    pred_geom = (cfg_geom["wa"] * la + cfg_geom["wb"] * lb + cfg_geom["wc"] * lc).argmax(1)
    pred_comp = pred3(ir_test, cfg_comp)

    meta = json.loads((cache / "test_meta.json").read_text(encoding="utf-8"))
    empty = set(json.loads((cache / "test_empty.json").read_text(encoding="utf-8")))
    fb = {}
    with open(TRACK / "baselines" / "skeleton_imu_v2" / "submission_skeleton_imu_v2_ensemble.csv") as f:
        for row in csv.DictReader(f):
            fb[row["path"].rstrip("/") + "/"] = int(row["prediction"])

    outs = {
        "submission_ir_v6.csv": (pred_primary, cfg_top4, "top4_base+finer_sameT_triple", cfg_top4["acc"]),
        "submission_ir_v6_ensbase.csv": (pred_ens, cfg_ens, "ens_base+finer_sameT", cfg_ens["acc"]),
        "submission_ir_v6_sel.csv": (pred_sel, cfg_sel, "ens_selective_tta+finer", cfg_sel["acc"]),
        "submission_ir_v6_nested.csv": (pred_nested, cfg_nested, "top4+nested_mean_cfg", probe["nested"].get("nested_mean_cfg_full_acc")),
        "submission_ir_v6_geom.csv": (pred_geom, cfg_geom, "top4+geom", cfg_geom["acc"]),
        "submission_ir_v6_compromise.csv": (pred_comp, cfg_comp, "top4+compromise_more_mid", None),
    }
    written = {}
    for name, (preds, cfg, tag, hold) in outs.items():
        nfb = write_sub(ROOT / name, meta, preds, empty, fb)
        written[name] = {"tag": tag, "hold": hold, "cfg": cfg, "empty_fallback": nfb}
        print(f"WROTE {name} tag={tag} hold={hold}", flush=True)

    # compare to v5
    v5 = {r["path"]: int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v5.csv"))}
    v6 = {r["path"]: int(r["prediction"]) for r in csv.DictReader(open(ROOT / "submission_ir_v6.csv"))}
    disagree = sum(1 for k in v5 if v5[k] != v6[k])

    fp16_mb = (ckpt / "model_fp16.pt").stat().st_size / (1024 * 1024)
    yolo_mb = (ROOT / "yolov8n.pt").stat().st_size / (1024 * 1024)

    report = {
        "tag": "ir_v6",
        "primary": "submission_ir_v6.csv",
        "method": "top4_base IR ens (seeds 2024,11,42,99) + Thermal v2_trio + MidFuse; finer same-T grid (incl T=0.75)",
        "holdout_acc": cfg_top4["acc"],
        "cfg": cfg_top4,
        "delta_vs_v5": float(cfg_top4["acc"] - V5),
        "rejected_perT": {
            "reason": "overfit: wa_IR=0.1 despite IR being strongest modality; nested LOUO mean only 0.7195",
            "hold_acc": probe["best_acc"],
            "cfg": probe["best_cfg"],
            "nested_mean": probe["nested"]["nested_mean"],
        },
        "ir_top4": top4,
        "ir_top4_member_acc": {t: member_scores[t] for t in top4},
        "alts": written,
        "disagree_vs_v5": disagree,
        "disagree_frac": disagree / len(v5),
        "fp16_pack_mb": fp16_mb,
        "yolo_mb": yolo_mb,
        "total_approx_mb": fp16_mb + yolo_mb,
        "size_ok_under_100mb": (fp16_mb + yolo_mb) < 100,
        "notes": [
            "PRIMARY beats v5 holdout by ~+0.006 via finer fuse grid + dropping 2 weakest IR seeds",
            "perT 0.747 rejected as holdout-overfit (IR weight 0.1)",
            "More IR seeds (1,777,333) training in checkpoints/ir_yolo_r2p1d18_v6 — re-finalize if they help ens",
            "TTA selective helps IR ens slightly (0.695->0.697) but top4_base fuse won",
            "Do not Kaggle-submit from this script",
            "Depth_Color still skipped (~0.21 holdout)",
        ],
        "recommend_submit_order": [
            f"submission_ir_v6.csv (top4+triple hold {cfg_top4['acc']:.4f}) — PRIMARY",
            f"submission_ir_v6_ensbase.csv (all6 hold {cfg_ens['acc']:.4f})",
            "submission_ir_v6_compromise.csv — public-gap hedge more Mid",
            "submission_ir_v5.csv — previous primary hold 0.7389",
        ],
    }
    (ROOT / "metrics_ir_v6.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
