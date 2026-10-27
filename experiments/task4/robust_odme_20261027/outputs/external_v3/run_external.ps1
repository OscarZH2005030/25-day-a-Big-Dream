# 中文说明：使用独立互斥锁，按下载、提取、建模顺序重现外部数据实验。
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$TaskDataPython = if ($env:TFB_PYTHON) { $env:TFB_PYTHON } else { 'python' }
$TaskModelPython = if ($env:TFB_PYTHON) { $env:TFB_PYTHON } else { 'python' }
$TaskMutex = New-Object System.Threading.Mutex($false, 'Local\TrafficFlowBench_Task4_External_v3')
$TaskAcquired = $TaskMutex.WaitOne(0)
if (-not $TaskAcquired) { throw '外部数据实验正在运行。' }
Push-Location $TaskRoot
try {
    & $TaskDataPython (Join-Path $PSScriptRoot 'acquire_real.py')
    if ($LASTEXITCODE -ne 0) { throw '下载失败。' }
    & $TaskDataPython (Join-Path $PSScriptRoot 'prepare_real.py')
    if ($LASTEXITCODE -ne 0) { throw '真实数据提取失败。' }
    & $TaskModelPython (Join-Path $PSScriptRoot 'run_external.py') 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'latest_run.log')
    if ($LASTEXITCODE -ne 0) { throw '实验失败，请查看日志。' }
    & $TaskModelPython (Join-Path $PSScriptRoot 'run_hybrid.py')
    if ($LASTEXITCODE -ne 0) { throw '混合候选实验失败。' }
    & $TaskModelPython (Join-Path $PSScriptRoot 'validate_external.py')
    if ($LASTEXITCODE -ne 0) { throw '验收失败。' }
} finally {
    Pop-Location
    if ($TaskAcquired) { $TaskMutex.ReleaseMutex() }
    $TaskMutex.Dispose()
}
