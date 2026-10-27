# 中文说明：互斥锁防止重复启动；保留日志与逐方向检查点。
$ErrorActionPreference = 'Stop'
$TaskRoot = $PSScriptRoot
$TaskPython = if ($env:TFB_PYTHON) { $env:TFB_PYTHON } else { 'python' }
$TaskMutex = New-Object System.Threading.Mutex($false, 'Local\TrafficFlowBench_Task4_ODME_20261026')
$TaskAcquired = $TaskMutex.WaitOne(0)
if (-not $TaskAcquired) { throw 'Task4 已在运行，请勿重复启动。' }
try {
    $env:OMP_NUM_THREADS = '4'
    $env:OPENBLAS_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    & $TaskPython (Join-Path $TaskRoot 'run_task4.py') 2>&1 | Tee-Object -FilePath (Join-Path $TaskRoot 'latest_run.log')
    if ($LASTEXITCODE -ne 0) { throw '实验失败，请检查日志和已保存检查点。' }
    & $TaskPython (Join-Path $TaskRoot 'validate_task4.py')
    if ($LASTEXITCODE -ne 0) { throw '结果验收失败。' }
} finally {
    if ($TaskAcquired) { $TaskMutex.ReleaseMutex() }
    $TaskMutex.Dispose()
}
