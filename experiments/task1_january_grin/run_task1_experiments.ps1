param(
    [ValidateSet('smoke','dev')]
    [string]$Mode = 'dev',
    [string[]]$Panels = @('D12_I5_N'),
    [string[]]$Variants = @('multihop','congestion'),
    [int]$Threads = 4,
    [string]$ReferenceRun = '',
    [string]$CacheDir = 'cache\task1',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_task1.ps1 first.' }
if (-not $Output) { $Output = 'runs\ablation_' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$cliArgs = @('-m','task1','ablate','--output',$Output,'--cache-dir',$CacheDir,'--threads',"$Threads")
if ($Mode -eq 'smoke') {
    $cliArgs += @('--max-fit-days','14','--max-eval-days','3','--sample-per-day','300','--rounds','50')
}
if ($ReferenceRun) { $cliArgs += @('--reference-run',$ReferenceRun) }
$cliArgs += @('--variants') + $Variants + @('--panels') + $Panels
& $pythonPath @cliArgs
exit $LASTEXITCODE
