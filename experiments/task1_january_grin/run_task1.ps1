param(
    [ValidateSet('audit','smoke','dev','test','final-fit','predict')]
    [string]$Mode = 'smoke',
    [string[]]$Panels = @('D12_I5_N'),
    [int]$Threads = 4,
    [string]$Models = '',
    [string]$Method = 'B2_lightgbm',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run setup_task1.ps1 first.'
}
if (-not $Output) { $Output = 'runs\' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
$baseArgs = @('-m','task1')
switch ($Mode) {
    'audit' { $cliArgs = @('audit','--output',($Output + '.json'),'--panels') + $Panels }
    'smoke' { $cliArgs = @('run','--max-fit-days','14','--max-eval-days','3','--sample-per-day','300','--rounds','50','--threads',"$Threads",'--output',$Output,'--panels') + $Panels }
    'dev' { $cliArgs = @('run','--threads',"$Threads",'--output',$Output,'--panels') + $Panels }
    'test' {
        if (-not $Models) { throw '-Models is required, e.g. runs\dev_xxx\models' }
        $cliArgs = @('evaluate','--models',$Models,'--eval-start','2031-02-01','--eval-end','2031-02-28','--output',$Output,'--panels') + $Panels
    }
    'final-fit' { $cliArgs = @('fit','--fit-end','2031-02-28','--threads',"$Threads",'--output',$Output,'--panels') + $Panels }
    'predict' {
        if (-not $Models) { throw '-Models is required' }
        $cliArgs = @('predict','--models',$Models,'--method',$Method,'--output',($Output + '.csv'),'--panels') + $Panels
    }
}
& $pythonPath @baseArgs @cliArgs
exit $LASTEXITCODE
