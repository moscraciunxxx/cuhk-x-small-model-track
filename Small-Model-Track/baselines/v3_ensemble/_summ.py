import json
from pathlib import Path
m=json.loads(Path("metrics_v3_long.json").read_text(encoding="utf-8"))
print("mean", m["mean_val_acc"], "std", m["std_val_acc"])
print("folds", [(f["tag"], round(f["best_val_acc"],4)) for f in m["folds"]])
print("holdout", (m.get("holdout") or {}).get("best_val_acc"))
print("all", (m.get("all_train") or {}).get("best_val_acc"))
