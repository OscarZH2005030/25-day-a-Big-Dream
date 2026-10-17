param(
    [ValidateSet('smoke','dev')]
    [string]$Mode = 'dev',
    [string]$SourceRun = '',
    [string[]]$Panels = @('all'),
    [string]$CacheDir = '',
    [switch]$NoLog,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_task1.ps1 first.' }
if (-not $SourceRun) {
    if ($Mode -eq 'smoke') { $SourceRun = 'runs\target_comparison_smoke' }
    else { $SourceRun = 'runs\target_dev_20261016_183242' }
}
if (-not $Output) { $Output = 'runs\fusion_' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$cliArgs = @('-m','task1','fusion-compare','--source-run',$SourceRun,'--output',$Output)
if ($Mode -eq 'smoke') { $cliArgs += @('--smoke') }
if ($CacheDir) { $cliArgs += @('--cache-dir',$CacheDir) }
if ($NoLog) { $cliArgs += @('--no-log') }
$cliArgs += @('--panels') + $Panels
& $pythonPath @cliArgs
exit $LASTEXITCODE
