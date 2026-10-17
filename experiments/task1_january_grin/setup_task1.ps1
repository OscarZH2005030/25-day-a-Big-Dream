$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$requirementsPath = if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
& $pythonPath -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonPath -m pytest tests -q
exit $LASTEXITCODE
