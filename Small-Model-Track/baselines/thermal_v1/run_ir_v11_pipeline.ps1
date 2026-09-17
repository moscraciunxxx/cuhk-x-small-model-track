$ErrorActionPreference = "Continue"
Set-Location "D:\CUHK-X\Small-Model-Track\baselines\thermal_v1"
$py = ".\.venv\Scripts\python.exe"
$log = "logs"
function Wait-PidFile($pidFile, $name) {
  if (-not (Test-Path $pidFile)) { Write-Host "no $pidFile"; return }
  $id = [int]((Get-Content $pidFile | Select-Object -First 1).ToString().Trim())
  Write-Host "Waiting $name pid=$id"
  while (Get-Process -Id $id -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
  Write-Host "$name done"
}
Wait-PidFile "$log\cache_depth_yolo_v4.pid" "depth_cache"
if (-not (Test-Path "cache\depth_color_yolo_v4\detect_stats_train.json")) { Write-Host "cache incomplete"; exit 1 }
Write-Host "detect train:" (Get-Content "cache\depth_color_yolo_v4\detect_stats_train.json" -Raw)
Write-Host "detect test:" (Get-Content "cache\depth_color_yolo_v4\detect_stats_test.json" -Raw -EA SilentlyContinue)
Write-Host "=== train depth v11 ==="
& $py -u train_depth_v11.py --seeds 42 123 7 --epochs 32 --batch-size 6 --mixup 0.2 --unfreeze-ep 4 *>&1 | Tee-Object -FilePath "$log\train_depth_v11.log"
Write-Host "=== write_ir_v11 (depth only first) ==="
& $py -u write_ir_v11.py *>&1 | Tee-Object -FilePath "$log\write_ir_v11.log"
# If gate fails, train IR strong and re-eval
$m = Get-Content "metrics_ir_v11.json" -Raw | ConvertFrom-Json
if (-not $m.clear_win) {
  Write-Host "=== gate failed; train IR strong v11 ==="
  & $py -u train_ir_v11_strong.py --seeds 888 2025 314 --epochs 32 --batch-size 6 --mixup 0.25 --unfreeze-ep 3 *>&1 | Tee-Object -FilePath "$log\train_ir_v11_strong.log"
  & $py -u write_ir_v11.py *>&1 | Tee-Object -FilePath "$log\write_ir_v11_after_strong.log"
}
Write-Host "PIPELINE DONE"
Get-Content "metrics_ir_v11.json" -Raw
