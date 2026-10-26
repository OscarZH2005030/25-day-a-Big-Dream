# Windows PowerShell 5.1 / PowerShell 7均可；先验包，再按冻结的十路配置推理。
# 不安装软件、不训练、不修改数据、不提交Kaggle。
param(
    [string]$Date,
    [string]$DataRoot,
    [string]$PythonPath,
    [ValidateSet('cpu','cuda')][string]$Device='cpu',
    [string[]]$Panel=@(),
    [string]$OutputDirectory,
    [switch]$VerifyOnly
)
$ErrorActionPreference='Stop'
$settingsPath=Join-Path $PSScriptRoot 'local-settings.json'
if (Test-Path -LiteralPath $settingsPath) {
    $settings=Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $PythonPath) { $PythonPath=$settings.PythonPath }
    if (-not $DataRoot) { $DataRoot=$settings.DataRoot }
}
if (-not $PythonPath) {
    $pythonCommand=Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $PythonPath=$pythonCommand.Source }
}
if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath)) { throw '请用 -PythonPath 指定已安装所需依赖的python.exe完整路径。' }
& $PythonPath (Join-Path $PSScriptRoot 'verify_package.py')
if ($LASTEXITCODE -ne 0) { throw '模型包校验失败，已停止。请重新解压完整模型包。' }
if ($VerifyOnly) { Write-Host '校验通过；未执行预测。' -ForegroundColor Green; return }
if (-not $Date -or $Date -notmatch '^\d{4}-\d{2}-\d{2}$') { throw '请使用 -Date 2031-03-01 这样的日期参数。' }
if (-not $DataRoot -or -not (Test-Path -LiteralPath $DataRoot)) { throw '请用 -DataRoot 指定包含corridors和task1目录的kaggle_public根目录。' }
if (-not $OutputDirectory) { $OutputDirectory=Join-Path $PSScriptRoot 'predictions' }
$arguments=@('-u',(Join-Path $PSScriptRoot 'predict_selected.py'),'--date',$Date,'--data-root',$DataRoot,'--device',$Device,'--output',$OutputDirectory)
foreach ($road in $Panel) { $arguments+=@('--panel',$road) }
Write-Host "开始按逐道路最佳配置预测 $Date，设备=$Device" -ForegroundColor Cyan
& $PythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw '预测未完成，请查看上方错误。原模型文件未修改。' }
Write-Host "完成，结果目录：$OutputDirectory" -ForegroundColor Green
