"""Offline import/GPU check. Does not train a model or contact Kaggle."""
import importlib,json,os,platform,sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
ROOT=Path(__file__).resolve().parents[1]
report={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'platform':platform.platform(),'packages':{}}
failed=[]
for name in ['numpy','pandas','pyarrow','scipy','sklearn','lightgbm','torch']:
    try:
        module=importlib.import_module(name);report['packages'][name]=module.__version__
    except Exception as exc:failed.append({'module':name,'error':str(exc)})
if 'torch' in sys.modules:
    import torch
    report['cuda_available']=torch.cuda.is_available();report['torch_cuda']=torch.version.cuda
    if report['cuda_available']:
        try:
            info=torch.cuda.get_device_properties(0)
            report['gpu']=info.name;report['gpu_memory_GiB']=info.total_memory/2**30
            a=torch.ones((32,32),device='cuda');assert float((a@a)[0,0])==32.;torch.cuda.synchronize()
            report['cuda_matmul_passed']=True
        except Exception as exc:failed.append({'module':'CUDA','error':str(exc)})
report['errors']=failed;report['passed']=not failed
folder=ROOT/'local_reports';folder.mkdir(parents=True,exist_ok=True)
(folder/'environment_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
if failed:raise SystemExit(1)
