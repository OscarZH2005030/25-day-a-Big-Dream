param(
    [ValidateSet('smoke','dev')][string]$Mode = 'dev',
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [int]$BatchSize = 4,
    [int]$Epochs = 20,
    [int]$StepsPerEpoch = 100,
    [int]$Threads = 4,
    [string]$Checkpoint = '',
    [switch]$PrepareOnly,
    [switch]$NoLog,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv-grin\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_grin.ps1 first.' }
if (-not $Output) { $Output = 'runs\grin_' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$action = if ($Checkpoint) { 'evaluate' } elseif ($PrepareOnly) { 'prepare' } else { 'run' }
$cliArgs = @('-m','task1.grin',$action,'--output',$Output,'--device',$Device,'--threads',"$Threads",'--batch-size',"$BatchSize")
if ($Mode -eq 'smoke') {
    $cliArgs += @('--max-fit-days','7','--max-eval-days','3','--sample-per-day','300',
        '--epochs','2','--steps-per-epoch','3','--hidden','16','--window','24','--stride','24')
} else {
    $cliArgs += @('--epochs',"$Epochs",'--steps-per-epoch',"$StepsPerEpoch")
}
if ($Checkpoint) { $cliArgs += @('--checkpoint',$Checkpoint) }
if ($NoLog) { $cliArgs += @('--no-log') }
& $pythonPath @cliArgs
exit $LASTEXITCODE
