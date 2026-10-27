# 下载并校验发布权重，仅提取models，不整包展开深层源码路径。
param([string]$PythonPath)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
if(-not $PythonPath){$candidate=Join-Path $root '.venv-task1-repro\Scripts\python.exe';if(Test-Path $candidate){$PythonPath=$candidate}else{$PythonPath=(Get-Command python.exe -ErrorAction Stop).Source}}
& $PythonPath -X utf8 (Join-Path $PSScriptRoot 'download_weights.py')
if($LASTEXITCODE -ne 0){throw '权重下载或校验失败。'}
