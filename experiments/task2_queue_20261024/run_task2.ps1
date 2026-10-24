param(
    [Parameter(Mandatory=$true)][ValidateSet('audit','prepare','smoke')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$ReleaseRoot,
    [ValidateSet('cuda','cpu')][string]$Device='cuda',
    [ValidateSet('engineering','full-inputs')][string]$Scope='engineering',
    [string]$Output='',
    [string]$CacheDir='',
    [string]$PythonPath='',
    [string[]]$Panels=@(),
    [int]$MaxDays=2,
    [int]$MaxWindows=8,
    [int]$Stride=6,
    [int]$Steps=4,
    [int]$Threads=4
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) { $PythonPath=Join-Path $PSScriptRoot '.venv-task2\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run setup_task2.ps1 first, or pass a matching -PythonPath.' }
$cli=@('-u','-B','-m','task2q',$Mode,'--release-root',$ReleaseRoot,'--device',$Device,'--scope',$Scope,
       '--max-days',$MaxDays,'--max-windows',$MaxWindows,'--stride',$Stride,'--steps',$Steps,'--threads',$Threads)
if ($Output) { $cli+=@('--output',$Output) }
if ($CacheDir) { $cli+=@('--cache-dir',$CacheDir) }
if ($Panels.Count -gt 0) { $cli+='--panels'; $cli+=$Panels }
& $PythonPath @cli
exit $LASTEXITCODE
