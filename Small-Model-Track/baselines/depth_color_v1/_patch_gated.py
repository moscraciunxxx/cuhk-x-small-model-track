from pathlib import Path
p = Path(r"D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1\train_gated_fuse.py")
t = p.read_text(encoding="utf-8")
start = t.find("def dump_midfuse_test_logits")
end = t.find("def build_video")
assert start > 0 and end > start
new = '''def dump_midfuse_test_logits(out_path: Path, device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2_model", V2 / "model.py")
    v2m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2m)

    cache = V2 / "cache"
    skel = np.load(cache / "skel_test.npz", allow_pickle=True)
    imu = np.load(cache / "imu_test.npz", allow_pickle=True)
    print("skel_test keys", list(skel.keys()), "imu keys", list(imu.keys()))
    Xs = skel["X"]
    Xi = imu["X"]
    flag = None
    for k in ("has_imu", "imu_mask", "mask", "flag"):
        if k in imu.files:
            flag = imu[k]
            break
    print("Xs", Xs.shape, "Xi", Xi.shape, "flag", None if flag is None else getattr(flag, "shape", type(flag)))

    ckpt = V2 / "checkpoints_midfuse_s123" / "best_all_train.pt"
    if not ckpt.exists():
        ckpt = V2 / "checkpoints" / "best_all_train.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = v2m.build_model("midfuse")
    state = blob.get("model_state") or blob.get("model") or blob.get("state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load missing", len(missing), "unexpected", len(unexpected), "ckpt", ckpt)
    model.to(device).eval()

    n = len(Xs)
    logits = np.zeros((n, 40), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for i in range(0, n, bs):
            xs = torch.from_numpy(np.asarray(Xs[i : i + bs], dtype=np.float32)).to(device)
            xi = torch.from_numpy(np.asarray(Xi[i : i + bs], dtype=np.float32)).to(device)
            if flag is not None:
                f = torch.from_numpy(np.asarray(flag[i : i + bs], dtype=np.float32)).to(device)
                out = model(xs, xi, f)
            else:
                f = torch.ones(len(xs), device=device)
                try:
                    out = model(xs, xi, f)
                except TypeError:
                    out = model(xs, xi)
            logits[i : i + bs] = out.cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, logits)
    print("saved midfuse test logits", out_path, logits.shape, "pred_mode", int(np.bincount(logits.argmax(1)).argmax()))
    return logits


'''
p.write_text(t[:start] + new + t[end:], encoding="utf-8")
print("patched ok")
