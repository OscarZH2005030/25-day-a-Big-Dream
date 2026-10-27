# 中文说明：锁定公开数据版本，复现数据审计及三项路径评分代理实验。
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$TaskPython = if ($env:TFB_PYTHON) { $env:TFB_PYTHON } else { 'python' }
$TaskMutex = New-Object System.Threading.Mutex($false, 'Local\TrafficFlowBench_Task4_FiveMin_v4')
$TaskAcquired = $TaskMutex.WaitOne(0)
if (-not $TaskAcquired) { throw '五分钟数据实验正在运行。' }
Push-Location $TaskRoot
try {
    foreach ($TaskScript in @('acquire_5min.py','audit_5min.py','run_path_experiments.py')) {
        & $TaskPython (Join-Path $PSScriptRoot $TaskScript) 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'latest_run.log') -Append
        if ($LASTEXITCODE -ne 0) { throw "运行失败：$TaskScript" }
    }
} finally {
    Pop-Location
    if ($TaskAcquired) { $TaskMutex.ReleaseMutex() }
    $TaskMutex.Dispose()
}
