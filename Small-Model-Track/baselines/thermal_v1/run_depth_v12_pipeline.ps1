# Depth v12: IR-box crops + multi-seed longer train. GPU exclusive vs LMT.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"

Write-Host "=== GPU before train ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

& $py -u train_depth_v11.py `
  --cache-dir ".\cache\depth_color_yolo_v4_irbox" `
  --ckpt-dir ".\checkpoints\depth_yolo_r2p1d18_v12" `
  --epochs 40 --patience 12 --batch-size 6 --lr 1e-4 `
  --seeds 42 123 7 2024 11 `
  --mixup 0.2 --unfreeze-ep 4 `
  2>&1 | Tee-Object -FilePath ".\logs\train_depth_v12.log"

Write-Host "=== train done; running fuse probe ==="
& $py -u write_ir_v12.py 2>&1 | Tee-Object -FilePath ".\logs\write_ir_v12.log"
