# 中文说明：Windows PowerShell5.1统一入口；默认只推理最新候选并验收。
param(
 [Parameter(Mandatory=$true)][string]$ReleaseRoot,
 [string]$Python = 'python',
 [ValidateSet('verify','latest','baseline','external','scale','bounds','robust','joint','joint-infer','joint-dev','joint-dev-infer','bias','invariant','invariant-infer','guarded','guarded-infer','bidirectional','bidirectional-infer','global-screen','all')][string]$Stage = 'latest',
 [string]$DataPython = ''
)
$ErrorActionPreference = 'Stop'
$env:TFB_RELEASE_ROOT = $ReleaseRoot
$env:TFB_PYTHON = $Python
$TaskArguments = @((Join-Path $PSScriptRoot 'reproduce.py'),'--release-root',$ReleaseRoot,'--stage',$Stage)
if ($DataPython) { $TaskArguments += @('--data-python',$DataPython) }
& $Python @TaskArguments
if ($LASTEXITCODE -ne 0) { throw 'Task4 reproduction failed.' }
