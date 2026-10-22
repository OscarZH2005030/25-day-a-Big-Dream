param([string]$PythonPath='', [switch]$SkipTests)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtime=Join-Path $PSScriptRoot '.venv-grin\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    if ($PythonPath) { & $PythonPath -m venv (Join-Path $PSScriptRoot '.venv-grin') }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.14 -m venv (Join-Path $PSScriptRoot '.venv-grin') }
    else { & python -m venv (Join-Path $PSScriptRoot '.venv-grin') }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $runtime -c "import sys; assert sys.version_info[:2] == (3,14), 'Install Python 3.14; the reference used 3.14.8'; print(sys.version)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runtime -m pip install -r (Join-Path $PSScriptRoot 'requirements-lock.txt') -r (Join-Path $PSScriptRoot 'requirements-torch-deps.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runtime -m pip install 'torch==2.10.0+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runtime -B (Join-Path $PSScriptRoot 'check_environment.py') --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $SkipTests) {
    $previousCudaDevices=$env:CUDA_VISIBLE_DEVICES
    try {
        $env:CUDA_VISIBLE_DEVICES='-1'
        & $runtime -B -m pytest tests -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        if ($null -eq $previousCudaDevices) { Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
        else { $env:CUDA_VISIBLE_DEVICES=$previousCudaDevices }
    }
}
Write-Host 'Environment ready. Download official data, then run run_team_remaining.ps1 -Mode smoke.'
