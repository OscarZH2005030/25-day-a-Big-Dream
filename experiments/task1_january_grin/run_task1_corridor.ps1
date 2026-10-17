param(
    [ValidateSet('smoke','dev')]
    [string]$Mode = 'dev',
    [string[]]$Panels = @('D7_I405_N'),
    [int]$Threads = 4,
    [string]$CacheDir = 'cache\task1',
    [switch]$NoLog,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_task1.ps1 first.' }
if (-not $Output) { $Output = 'runs\corridor_' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$cliArgs = @('-m','task1','corridor-validate','--output',$Output,'--cache-dir',$CacheDir,'--threads',"$Threads")
if ($Mode -eq 'smoke') {
    $cliArgs += @('--max-fit-days','14','--max-eval-days','3','--sample-per-day','300','--rounds','50')
}
if ($NoLog) { $cliArgs += @('--no-log') }
$cliArgs += @('--panels') + $Panels
& $pythonPath @cliArgs
exit $LASTEXITCODE
