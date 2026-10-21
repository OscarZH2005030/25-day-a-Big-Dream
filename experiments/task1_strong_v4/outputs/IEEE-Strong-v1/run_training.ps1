param([int]$Epochs=16, [int]$Steps=60, [string]$OutputPath='')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python=Join-Path $taskRoot 'work\gpu-env\Scripts\python.exe'
$cache=Join-Path $taskRoot 'work\ieee-strong-cache-time8020'
$reference=Join-Path $PSScriptRoot 'references\transformer-all-metrics.csv'
if (-not (Test-Path -LiteralPath (Join-Path $cache 'metadata.json'))) { throw 'Prepare the time80/20 feature cache first; see the experiment protocol.' }
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'IEEE-Strong-v1.*train(?:_refinement|_hybrid)?\.py' })
if ($active.Count -gt 0) { throw 'A Strong-v1 training process is already running. Read its progress.json instead of starting a duplicate.' }
if ([string]::IsNullOrWhiteSpace($OutputPath)) { $OutputPath=Join-Path $PSScriptRoot ('training-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
if (Test-Path -LiteralPath $OutputPath) {
    if (@(Get-ChildItem -LiteralPath $OutputPath -Force).Count -gt 0) { throw 'Use a new empty output folder to preserve previous model results.' }
}
$env:PYTHONPYCACHEPREFIX=Join-Path $taskRoot 'work\python-bytecode'
& $python -X utf8 (Join-Path $PSScriptRoot 'train.py') --cache $cache --out $OutputPath --reference $reference --epochs $Epochs --steps $Steps --accumulate 2
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Inspect training.log in the output folder.' }
& $python -X utf8 (Join-Path $PSScriptRoot 'calibrate_and_compare.py') --cache $cache --out $OutputPath --reference $reference
if ($LASTEXITCODE -ne 0) { throw 'Calibration failed. Preserve training outputs and inspect the error.' }
& $python -X utf8 (Join-Path $PSScriptRoot 'write_result.py') --out $OutputPath
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
Write-Output "Completed training and comparison: $OutputPath"
