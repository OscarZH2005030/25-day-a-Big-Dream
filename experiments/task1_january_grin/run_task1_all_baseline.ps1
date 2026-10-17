param(
    [ValidateSet('check','recover','run')]
    [string]$Mode = 'run',
    [int]$Threads = 4,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$cliArgs = @('-u','-B','resume_task1_baseline.py','--mode',$Mode,'--threads',"$Threads")
if ($Output) { $cliArgs += @('--output',$Output) }
& $pythonPath @cliArgs
exit $LASTEXITCODE
