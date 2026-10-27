# 中文说明：互斥运行S_dev候选比较和固定候选敏感性实验。
$ErrorActionPreference = 'Stop'
$TaskPython = if ($env:TFB_PYTHON) { $env:TFB_PYTHON } else { 'python' }
$TaskMutex = New-Object System.Threading.Mutex($false, 'Local\TrafficFlowBench_Task4_Dev_v5')
$TaskAcquired = $TaskMutex.WaitOne(0)
if (-not $TaskAcquired) { throw 'S_dev实验正在运行。' }
try {
    foreach ($TaskScript in @('run_dev.py','run_stress.py')) {
        & $TaskPython (Join-Path $PSScriptRoot $TaskScript) 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'latest_run.log') -Append
        if ($LASTEXITCODE -ne 0) { throw "运行失败：$TaskScript" }
    }
} finally {
    if ($TaskAcquired) { $TaskMutex.ReleaseMutex() }
    $TaskMutex.Dispose()
}
