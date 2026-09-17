from pathlib import Path
p = Path(r"D:\CUHK-X\Small-Model-Track\baselines\depth_color_v1\train_v2.py")
t = p.read_text(encoding="utf-8")
repls = [
("best_f1, best_state, history = -1.0, None, []", "best_acc, best_state, history = -1.0, None, []"),
("if metrics[\"macro_f1\"] > best_f1:\n            best_f1 = metrics[\"macro_f1\"]", "if metrics[\"acc\"] > best_acc:\n            best_acc = metrics[\"acc\"]"),
("\"val_f1\": best_f1", "\"val_acc\": best_acc, \"val_f1\": metrics[\"macro_f1\"]"),
("best_ep = max((h[\"epoch\"] for h in history if abs(h[\"val_f1\"] - best_f1) < 1e-9), default=epoch)\n        if args.patience > 0 and epoch - best_ep >= args.patience:\n            print(f\"[{tag}] early stop at ep{epoch}, best_f1={best_f1:.4f}\", flush=True)",
 "best_ep = max((h[\"epoch\"] for h in history if abs(h[\"val_acc\"] - best_acc) < 1e-9), default=epoch)\n        if args.patience > 0 and epoch - best_ep >= args.patience:\n            print(f\"[{tag}] early stop at ep{epoch}, best_acc={best_acc:.4f}\", flush=True)"),
("print(f\"[{tag}] BEST val_f1={best_f1:.4f} size_mb={size_fn(model):.2f} took={time.time()-t0:.1f}s\", flush=True)\n    return {\"best_f1\": best_f1,",
 "print(f\"[{tag}] BEST val_acc={best_acc:.4f} size_mb={size_fn(model):.2f} took={time.time()-t0:.1f}s\", flush=True)\n    return {\"best_f1\": metrics[\"macro_f1\"], \"best_acc\": best_acc,"),
("results[\"holdout\"] = {\"macro_f1\": out[\"best_f1\"], \"ckpt\": out[\"ckpt\"]}",
 "results[\"holdout\"] = {\"macro_f1\": out[\"best_f1\"], \"acc\": out.get(\"best_acc\"), \"ckpt\": out[\"ckpt\"]}"),
("return build_model_r2p1d(NUM_CLASSES, in_ch=in_ch, base=48), count_parameters, model_size_mb",
 "return build_model_r2p1d(NUM_CLASSES, in_ch=in_ch, base=64), count_parameters, model_size_mb"),
]
for a,b in repls:
    if a not in t:
        print("MISSING", a[:60])
    else:
        t = t.replace(a,b)
        print("ok", a[:40])
p.write_text(t, encoding="utf-8")
Path(r"D:\CUHK-X\Small-Model-Track\baselines\thermal_v1\train_v2.py").write_text(t, encoding="utf-8")
print("done")
