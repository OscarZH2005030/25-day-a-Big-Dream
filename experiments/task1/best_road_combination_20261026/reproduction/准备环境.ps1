# 仅创建项目独立环境，不修改现有Python或系统CUDA。
param([Parameter(Mandatory=$true)][string]$PythonPath)
$ErrorActionPreference='Stop'
& $PythonPath -c "import sys; assert sys.version_info[:2]==(3,12), 'Need Python 3.12'"
if($LASTEXITCODE -ne 0){throw '请指定Python3.12解释器。'}
$root=Split-Path $PSScriptRoot -Parent
$venv=Join-Path $root '.venv-task1-repro'
if(-not(Test-Path (Join-Path $venv 'Scripts\python.exe'))){& $PythonPath -m venv $venv;if($LASTEXITCODE -ne 0){throw '创建独立环境失败。'}}
$python=Join-Path $venv 'Scripts\python.exe'
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-cpu.txt')
if($LASTEXITCODE -ne 0){throw '依赖安装失败，请查看pip日志，不要静默替换版本。'}
& $python -m pip install 'torch==2.9.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
if($LASTEXITCODE -ne 0){throw 'PyTorch安装失败，请检查网络和对应版本。'}
& $python -c "import torch,numpy,pandas,lightgbm; print(torch.__version__,numpy.__version__,pandas.__version__,lightgbm.__version__); print('CUDA available:',torch.cuda.is_available())"
if($LASTEXITCODE -ne 0){throw '导入依赖失败。'}
Write-Host ('环境路径：'+$python) -ForegroundColor Green
