param([string]$PythonPath='')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtime=Join-Path $PSScriptRoot '.venv-task2\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    if ($PythonPath) { & $PythonPath -m venv (Join-Path $PSScriptRoot '.venv-task2') }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.14 -m venv (Join-Path $PSScriptRoot '.venv-task2') }
    else { throw 'Install Python 3.14, or pass -PythonPath to its python.exe.' }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create isolated Task 2 environment' }
}
& $runtime -c "import sys; assert sys.version_info[:2] == (3,14), 'Python 3.14 required'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Wrong Python version' }
& $runtime -m pip install -r requirements-lock.txt -r requirements-torch-deps.txt
if ($LASTEXITCODE -ne 0) { throw 'Task 2 dependencies could not be installed' }
& $runtime -m pip install 'torch==2.10.0+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Pinned CUDA PyTorch installation failed' }
& $runtime -B check_environment.py --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Task 2 environment/GPU/source check failed' }
$previousCuda=$env:CUDA_VISIBLE_DEVICES
$previousThreads=$env:OMP_NUM_THREADS
try {
    $env:CUDA_VISIBLE_DEVICES='-1'
    $env:OMP_NUM_THREADS='2'
    $testTemp=Join-Path $PSScriptRoot ('runs\cpu_test_' + [guid]::NewGuid().ToString('N'))
    & $runtime -B -m pytest tests -q -p no:cacheprovider --basetemp $testTemp
    if ($LASTEXITCODE -ne 0) { throw 'Task 2 CPU tests failed' }
} finally {
    if ($null -eq $previousCuda) { Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    else { $env:CUDA_VISIBLE_DEVICES=$previousCuda }
    if ($null -eq $previousThreads) { Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue }
    else { $env:OMP_NUM_THREADS=$previousThreads }
}
Write-Host 'Task 2 environment checked. Run audit and bounded smoke next; no formal training was started.'
