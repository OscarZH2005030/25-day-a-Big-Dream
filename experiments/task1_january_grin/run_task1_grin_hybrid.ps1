param(
    [ValidateSet('i5','d7','smoke')][string]$Mode = 'd7',
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [int]$BatchSize = 2,
    [int]$Threads = 4,
    [string]$GrinRun = '',
    [string]$BaselineRun = '',
    [switch]$NoLog,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv-grin\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_grin.ps1 first.' }
if (-not $Output) { $Output = 'runs\grin_hybrid_' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$action = if ($Mode -eq 'i5') { 'combine' } else { 'validate' }
$panel = if ($Mode -eq 'i5') { 'D12_I5_N' } else { 'D7_I405_N' }
$cliArgs = @('-m','task1.grin.hybrid',$action,'--output',$Output,'--panel',$panel,
    '--device',$Device,'--batch-size',"$BatchSize",'--threads',"$Threads")
if ($Mode -eq 'smoke') { $cliArgs += @('--smoke') }
if ($GrinRun) { $cliArgs += @('--grin-run',$GrinRun) }
if ($BaselineRun) { $cliArgs += @('--baseline-run',$BaselineRun) }
if ($NoLog) { $cliArgs += @('--no-log') }
& $pythonPath @cliArgs
exit $LASTEXITCODE
