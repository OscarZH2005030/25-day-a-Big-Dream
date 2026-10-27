# 中文说明：默认只重建已选候选；加-Reproduce重做人工实验和选模。
param([switch]$Reproduce)
$ErrorActionPreference = 'Stop'
$TaskPython = if ($env:TFB_PYTHON) { $env:TFB_PYTHON } else { 'python' }
$TaskMutex = New-Object System.Threading.Mutex($false, 'Local\TrafficFlowBench_Task4_ScaleShape_v8')
$TaskAcquired = $TaskMutex.WaitOne(0)
if (-not $TaskAcquired) { throw '尺度与形状实验正在运行。' }
try {
    $TaskScripts = @('infer_balanced.py')
    if ($Reproduce) { $TaskScripts = @('run_scale_shape.py','select_balanced.py','infer_balanced.py') }
    foreach ($TaskScript in $TaskScripts) {
        & $TaskPython (Join-Path $PSScriptRoot $TaskScript) 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'latest_run.log') -Append
        if ($LASTEXITCODE -ne 0) { throw "运行失败：$TaskScript" }
    }
} finally {
    if ($TaskAcquired) { $TaskMutex.ReleaseMutex() }
    $TaskMutex.Dispose()
}
