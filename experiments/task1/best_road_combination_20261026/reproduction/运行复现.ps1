# 相同入口区分从零训练、权重复核和提交生成；从零结果不借用历史分数。
param(
 [ValidateSet('Audit','Smoke','Train','Evaluate','Submit')][string]$Stage='Audit',
 [Parameter(Mandatory=$true)][string]$DataRoot,
 [Parameter(Mandatory=$true)][string]$PythonPath,
 [string]$WorkDirectory,
 [string]$Bundle,
 [ValidateSet('cpu','cuda')][string]$Device='cuda'
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
if(-not $WorkDirectory){$WorkDirectory=Join-Path $root ('runs\'+$Stage)}
$map=@{Audit='audit';Smoke='train-all';Train='train-all';Evaluate='evaluate';Submit='submit'}
$argv=@('-X','utf8','-u',(Join-Path $PSScriptRoot 'pipeline.py'),$map[$Stage],'--data-root',$DataRoot,'--work-dir',$WorkDirectory,'--device',$Device)
if($Stage -eq 'Smoke'){$argv+='--smoke'}
if($Bundle){$argv+=@('--bundle',$Bundle)}
& $PythonPath @argv
if($LASTEXITCODE -ne 0){throw ('阶段失败：'+$Stage+'。请检查日志，已完成缓存和模型不会被当作新结果重写。')}
