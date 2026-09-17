from pathlib import Path
p = Path(r"D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1\train_gated_fuse.py")
t = p.read_text(encoding="utf-8")
# Fix sys.path order at top: V2 before ROOT
t2 = t.replace(
"sys.path.insert(0, str(V2))\nsys.path.insert(0, str(ROOT))\n",
"sys.path.insert(0, str(ROOT))\nsys.path.insert(0, str(V2))  # V2 first for model/dataset dims\n",
)
# But then our dataset import breaks. Better: in dump, temporarily fix path.
start = t2.find("def dump_midfuse_test_logits")
end = t2.find("def build_video")
new = '''def dump_midfuse_test_logits(out_path: Path, device):
    import importlib
    import importlib.util

    # Ensure skeleton_imu_v2 dataset/model resolve, not local video dataset
    sys.path = [str(V2)] + [p for p in sys.path if Path(p).resolve() != ROOT.resolve()]
    for mod in ("dataset", "model"):
        if mod in sys.modules:
            del sys.modules[mod]
    import model as v2m  # noqa

    cache = V2 / "cache"
    skel = np.load(cache / "skel_test.npz", allow_pickle=True)
    imu = np.load(cache / "imu_test.npz", allow_pickle=True)
    Xs = skel["X"]
    Xi = imu["X"]
    flag = imu["has_imu"]
    print("Xs", Xs.shape, "Xi", Xi.shape, "has_imu", flag.shape)

    ckpt = V2 / "checkpoints_midfuse_s123" / "best_all_train.pt"
    if not ckpt.exists():
        ckpt = V2 / "checkpoints" / "best_all_train.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = v2m.build_model(blob.get("model_name", "midfuse"), num_classes=int(blob.get("num_classes", 40)))
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()

    n = len(Xs)
    logits = np.zeros((n, 40), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for i in range(0, n, bs):
            xs = torch.from_numpy(np.asarray(Xs[i : i + bs], dtype=np.float32)).to(device)
            xi = torch.from_numpy(np.asarray(Xi[i : i + bs], dtype=np.float32)).to(device)
            f = torch.from_numpy(np.asarray(flag[i : i + bs], dtype=np.float32)).to(device)
            out = model(xs, xi, f)
            logits[i : i + bs] = out.cpu().numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, logits)
    print("saved midfuse test logits", out_path, logits.shape)

    # restore path for local video modules
    sys.path = [str(ROOT), str(V2)] + [p for p in sys.path if p not in (str(ROOT), str(V2))]
    for mod in ("dataset", "model"):
        if mod in sys.modules:
            del sys.modules[mod]
    return logits


'''
p.write_text(t2[:start] + new + t2[end:], encoding="utf-8")
print("patched")
