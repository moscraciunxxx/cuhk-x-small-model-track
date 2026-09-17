import torch
from pathlib import Path
for alias in ["kd", "kd_mf_v13", "kd_mf_a02", "kd_eq_a02"]:
    print(alias)
    for fi in range(5):
        ck = torch.load(f"baselines/v18/checkpoints_{alias}/best_fold{fi}.pt", map_location="cpu", weights_only=False)
        print(f"  fold{fi} val_acc={ck.get('val_acc')} epoch={ck.get('epoch')}")
