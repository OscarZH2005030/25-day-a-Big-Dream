param([ValidateSet('cu128','cpu')][string]$TorchBuild = 'cu128')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$basePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$grinPython = Join-Path $PSScriptRoot '.venv-grin\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $grinPython)) {
    if (-not (Test-Path -LiteralPath $basePython)) { throw 'Run setup_task1.ps1 first to create the baseline environment.' }
    & $basePython -m venv (Join-Path $PSScriptRoot '.venv-grin')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $grinPython -m pip install -r (Join-Path $PSScriptRoot 'requirements_grin.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$installedVersion = & $grinPython -c "import importlib.metadata as m; print(m.version('torch') if any(d.metadata['Name'].lower() == 'torch' for d in m.distributions()) else 'none')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $installedVersion.EndsWith('+' + $TorchBuild)) {
    & $grinPython -m pip install --upgrade --force-reinstall 'torch==2.10.0' --index-url ('https://download.pytorch.org/whl/' + $TorchBuild)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $grinPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($TorchBuild -eq 'cu128') {
    & $grinPython -c "import torch; assert torch.cuda.is_available(), 'CUDA build installed but GPU unavailable. Inspect the driver, or select CPU explicitly.'; x=torch.randn(64,64,device='cuda'); print('CUDA compute check:',(x@x.T).mean().item())"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host 'GRIN environment ready. Existing baseline environment is preserved.'
