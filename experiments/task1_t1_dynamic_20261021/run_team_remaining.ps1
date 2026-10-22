param(
    [ValidateSet('smoke','dev')][string]$Mode='smoke',
    [ValidateSet('all','i210','i10','i405')][string]$Batch='all',
    [ValidateSet('cuda','cpu')][string]$Device='cuda',
    [switch]$PrepareOnly,
    [switch]$Resume,
    [switch]$SummaryOnly,
    [switch]$LogOnly,
    [string]$Output='',
    [string]$CacheDir='',
    [string]$ReleaseRoot='',
    [string]$PythonPath=''
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $PythonPath) { $PythonPath=Join-Path $PSScriptRoot '.venv-grin\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run setup_team.ps1 first, or pass -PythonPath to an existing matching environment.' }
if (@(@($PrepareOnly,$SummaryOnly,$LogOnly) | Where-Object { $_ }).Count -gt 1) { throw 'Choose one action.' }
$action=if ($PrepareOnly) {'prepare'} elseif ($SummaryOnly) {'summary'} elseif ($LogOnly) {'log'} else {'run'}
$cliArgs=@('-u','-B',(Join-Path $PSScriptRoot 'team_remaining.py'),$action,'--batch',$Batch,'--device',$Device)
if ($Mode -eq 'smoke') { $cliArgs+='--smoke' }
if ($Resume) { $cliArgs+='--resume' }
if ($Output) { $cliArgs+=@('--output',$Output) }
if ($CacheDir) { $cliArgs+=@('--cache-dir',$CacheDir) }
if ($ReleaseRoot) { $cliArgs+=@('--release-root',$ReleaseRoot) }
& $PythonPath @cliArgs
exit $LASTEXITCODE
