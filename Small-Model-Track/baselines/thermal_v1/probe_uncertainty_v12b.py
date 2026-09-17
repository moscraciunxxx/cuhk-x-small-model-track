"""v12b: use IR seed disagreement / entropy as uncertainty for gating specialists."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from fuse_ir_v9 import load_members, softmax_np
from write_ir_v11 import align_depth_hold

ROOT = Path(__file__).resolve().parent
V7_HOLD = 0.7530364372469636
V7_NESTED = 0.7519623092355898
V7_CFG = {"wa": 0.56, "wb": 0.35, "wc": 0.09, "T": 2.5}
MIN_DELTA, MIN_DISAGREE = 0.01, 20


def soft(z, T=1.0):
    return softmax_np(z, T)


def fuse_v7(ir, th, mid, cfg=V7_CFG):
    T = cfg["T"]
    return cfg["wa"] * soft(ir, T) + cfg["wb"] * soft(th, T) + cfg["wc"] * soft(mid, T)


def main():
    t0 = time.time()
    members, yt, yu = load_members()
    old = ROOT / "checkpoints" / "ir_yolo_r2p1d18_v5"
    cache = ROOT / "cache" / "ir_yolo_v4"
    th = np.load(old / "hold_thermal_v6.npy")
    mid = np.load(cache / "midfuse_aligned_train_logits.npy")
    tu = np.load(cache / "train_users.npy")
    from dataset import DEFAULT_HOLD_OUT_USERS
    hold_idx = np.where(np.isin(tu, list(DEFAULT_HOLD_OUT_USERS)))[0]
    mid_h = mid[hold_idx] if len(mid) == len(tu) else mid
    if len(yu) != len(yt):
        yu = tu[hold_idx]
    mask = th.any(1) & mid_h.any(1)

    bases = [m["base"] for m in members if m["tag"] != "pool_seed55"]
    ir = np.mean(bases, 0).astype(np.float32)
    stack = np.stack(bases, 0)  # S,N,C
    preds = stack.argmax(-1)  # S,N
    # agreement rate
    mode_pred = ir.argmax(1)
    agree = (preds == mode_pred[None, :]).mean(0)  # fraction seeds agreeing with ens
    # entropy of mean softmax
    p = soft(ir, 1.0)
    ent = -(p * np.log(np.clip(p, 1e-8, 1))).sum(1)
    # pairwise disagree count
    S = len(bases)
    dis_pairs = np.zeros(len(yt))
    for i in range(S):
        for j in range(i + 1, S):
            dis_pairs += (preds[i] != preds[j]).astype(np.float64)
    dis_pairs /= (S * (S - 1) / 2)

    depth_pack = np.load(ROOT / "checkpoints" / "depth_yolo_r2p1d18_v11" / "hold_logits_v11.npz", allow_pickle=True)
    depth_h = align_depth_hold(depth_pack, yt, yu, cache / "train_meta.json",
                               ROOT / "cache" / "depth_color_yolo_v4" / "train_meta.json")
    v7p = fuse_v7(ir, th, mid_h)
    pdep = soft(depth_h, 2.5)
    pth = soft(th, 2.5)
    pmid = soft(mid_h, 2.5)

    print(f"v7={(v7p[mask].argmax(1)==yt[mask]).mean():.4f} agree mean={agree.mean():.3f} ent mean={ent.mean():.3f}", flush=True)

    # bins of agree vs acc
    for lo, hi in [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (agree >= lo) & (agree < hi) & mask
        if m.sum() == 0:
            continue
        print(f"agree[{lo},{hi}) n={m.sum()} v7_acc={(v7p[m].argmax(1)==yt[m]).mean():.3f} "
              f"dep={(pdep[m].argmax(1)==yt[m]).mean():.3f} "
              f"dep_helps_v7wrong={((v7p[m].argmax(1)!=yt[m])&(pdep[m].argmax(1)==yt[m])).sum()}", flush=True)

    attempts = []
    # gate when agree < thr: blend specialist
    for thr in [0.4, 0.5, 0.55, 0.6, 0.67, 0.7, 0.75, 0.8, 0.9]:
        low = agree < thr
        for name, aux in [("depth", pdep), ("th", pth), ("mid", pmid),
                          ("th+dep", 0.7 * pth + 0.3 * pdep),
                          ("eq3", (pth + pdep + pmid) / 3)]:
            for w in [0.3, 0.5, 0.7, 1.0]:
                out = v7p.copy()
                out[low] = (1 - w) * v7p[low] + w * aux[low]
                acc = float((out[mask].argmax(1) == yt[mask]).mean())
                attempts.append({"signal": "agree", "thr": thr, "aux": name, "w": w,
                                 "n_low": int((low & mask).sum()), "acc": acc})

    # gate on high entropy
    for thr in np.percentile(ent[mask], [50, 60, 70, 75, 80, 85, 90]):
        low = ent >= thr
        for name, aux in [("depth", pdep), ("th", pth), ("th+dep", 0.7 * pth + 0.3 * pdep)]:
            for w in [0.3, 0.5, 0.7, 1.0]:
                out = v7p.copy()
                out[low] = (1 - w) * v7p[low] + w * aux[low]
                acc = float((out[mask].argmax(1) == yt[mask]).mean())
                attempts.append({"signal": "entropy", "thr": float(thr), "aux": name, "w": w,
                                 "n_low": int((low & mask).sum()), "acc": acc})

    # gate on pairwise dis
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        low = dis_pairs >= thr
        for name, aux in [("depth", pdep), ("th", pth), ("th+dep", 0.7 * pth + 0.3 * pdep)]:
            for w in [0.3, 0.5, 0.7, 1.0]:
                out = v7p.copy()
                out[low] = (1 - w) * v7p[low] + w * aux[low]
                acc = float((out[mask].argmax(1) == yt[mask]).mean())
                attempts.append({"signal": "pair_dis", "thr": thr, "aux": name, "w": w,
                                 "n_low": int((low & mask).sum()), "acc": acc})

    # class-conditional: for classes where depth helps more, always mix depth
    ir_pred = ir.argmax(1)
    dep_pred = depth_h.argmax(1)
    help_by_c = {}
    for c in range(40):
        m = yt == c
        if m.sum() < 5:
            continue
        ir_w = (ir_pred[m] != yt[m])
        dep_ok = (dep_pred[m] == yt[m])
        help_by_c[c] = int((ir_w & dep_ok).sum())
    top_classes = sorted(help_by_c, key=lambda c: -help_by_c[c])[:8]
    print("top depth-help classes", [(c, help_by_c[c]) for c in top_classes], flush=True)

    # if predicted class in help set OR agree low -> blend depth
    for w in [0.2, 0.3, 0.4, 0.5]:
        out = v7p.copy()
        sel = np.isin(v7p.argmax(1), top_classes) | (agree < 0.6)
        out[sel] = (1 - w) * v7p[sel] + w * pdep[sel]
        acc = float((out[mask].argmax(1) == yt[mask]).mean())
        attempts.append({"signal": "class+agree", "classes": top_classes, "w": w,
                         "n_low": int((sel & mask).sum()), "acc": acc})

    attempts = sorted(attempts, key=lambda d: -d["acc"])
    print("\nTOP 12:", flush=True)
    for a in attempts[:12]:
        print(a, flush=True)

    # nested for anything >= v7
    nested = []
    for a in attempts:
        if a["acc"] + 1e-9 < V7_HOLD:
            continue
        # rebuild
        if a["signal"] == "agree":
            low = agree < a["thr"]
        elif a["signal"] == "entropy":
            low = ent >= a["thr"]
        elif a["signal"] == "pair_dis":
            low = dis_pairs >= a["thr"]
        else:
            continue
        aux = {"depth": pdep, "th": pth, "mid": pmid,
               "th+dep": 0.7 * pth + 0.3 * pdep,
               "eq3": (pth + pdep + pmid) / 3}[a["aux"]]
        out = v7p.copy()
        out[low] = (1 - a["w"]) * v7p[low] + a["w"] * aux[low]
        folds = []
        for u in (8, 9, 24):
            te = mask & (yu == u)
            folds.append(float((out[te].argmax(1) == yt[te]).mean()))
        nest = float(np.mean(folds))
        dis = int((out[mask].argmax(1) != v7p[mask].argmax(1)).sum())
        nested.append({**a, "nested": nest, "delta_h": a["acc"] - V7_HOLD,
                       "delta_n": nest - V7_NESTED, "disagree": dis,
                       "clear": a["acc"] >= V7_HOLD + MIN_DELTA and nest >= V7_NESTED + MIN_DELTA and dis >= MIN_DISAGREE})
        if len(nested) >= 15:
            break

    print("\nNested hold-winners:", flush=True)
    for r in nested:
        print(r, flush=True)

    outj = {
        "tag": "probe_uncertainty_v12b",
        "best_hold": attempts[0],
        "beat_v7_count": sum(1 for a in attempts if a["acc"] >= V7_HOLD),
        "nested": nested,
        "any_clear": any(r.get("clear") for r in nested),
        "elapsed_s": time.time() - t0,
    }
    (ROOT / "metrics_probe_uncertainty_v12b.json").write_text(json.dumps(outj, indent=2))
    print("any_clear", outj["any_clear"], "beat_v7", outj["beat_v7_count"], flush=True)


if __name__ == "__main__":
    main()
