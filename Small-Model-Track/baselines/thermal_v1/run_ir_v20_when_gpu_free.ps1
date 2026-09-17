# Wait for idle 3060 then train S3D seed42 + probe fuse. Do NOT kill other jobs.
# Usage: powershell -File run_ir_v20_when_gpu_free.ps1
$ErrorActionPreference = "Stop"
$Root = "D:\CUHK-X\Small-Model-Track\baselines\thermal_v1"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "ir_v20_s3d_$Stamp.log"

function Gpu-Busy {
  $apps = nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null
  if (-not $apps) { return $false }
  $lines = @($apps | Where-Object { $_.Trim() -ne "" -and $_.Trim() -ne "N/A" })
  return $lines.Count -gt 0
}

"[$([DateTime]::Now)] waiting for GPU idle (LMT yield)..." | Tee-Object -FilePath $Log -Append
while (Gpu-Busy) {
  Start-Sleep -Seconds 30
}
"[$([DateTime]::Now)] GPU idle — starting S3D seed42" | Tee-Object -FilePath $Log -Append

Set-Location $Root
& $Py train_ir_s3d_v20.py --seeds 42 --epochs 36 --batch-size 6 --patience 12 --skip-test 2>&1 | Tee-Object -FilePath $Log -Append
$trainExit = $LASTEXITCODE
if ($trainExit -ne 0) {
  "[$([DateTime]::Now)] S3D train failed exit=$trainExit — trying CNN+GRU fallback" | Tee-Object -FilePath $Log -Append
  & $Py train_ir_cnn_gru_v20.py --seeds 42 --epochs 40 --batch-size 16 --patience 12 2>&1 | Tee-Object -FilePath $Log -Append
}

# Quick abort check before multi-seed: read metrics if present
$s3dMetrics = Join-Path $Root "checkpoints\ir_yolo_s3d_v20\metrics_ir_s3d_v20.json"
$solo = $null
if (Test-Path $s3dMetrics) {
  $j = Get-Content $s3dMetrics -Raw | ConvertFrom-Json
  $solo = [double]$j.member_scores.seed42
  "[$([DateTime]::Now)] S3D seed42 solo=$solo" | Tee-Object -FilePath $Log -Append
}

& $Py probe_ir_v20.py 2>&1 | Tee-Object -FilePath $Log -Append

# Multi-seed only if promising (>=0.65) — parent/probe decides complementarity
if ($solo -ne $null -and $solo -ge 0.65) {
  "[$([DateTime]::Now)] promising solo — training seeds 7 55" | Tee-Object -FilePath $Log -Append
  if (-not (Gpu-Busy)) {
    & $Py train_ir_s3d_v20.py --seeds 7 55 --epochs 36 --batch-size 6 --patience 12 --skip-test 2>&1 | Tee-Object -FilePath $Log -Append
    & $Py probe_ir_v20.py 2>&1 | Tee-Object -FilePath $Log -Append
  } else {
    "[$([DateTime]::Now)] GPU reclaimed — skip multi-seed" | Tee-Object -FilePath $Log -Append
  }
}

"[$([DateTime]::Now)] done. GPU handoff." | Tee-Object -FilePath $Log -Append
nvidia-smi | Tee-Object -FilePath $Log -Append
