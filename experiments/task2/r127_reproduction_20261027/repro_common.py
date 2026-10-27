"""Relative-path backend for the unchanged R127 inference archive."""
import os,sys,json,hashlib,importlib
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def setup(package,data_root,work,io_audit=True,expected_hashes=None,extra_allowed=()):
    package=Path(package).resolve();data_root=Path(data_root).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    cfg=read(package/'profile.json');manifest=read(package/'archive-manifest.json')
    for name,record in manifest['files'].items():assert sha(package/name)==record['sha256'],'Package changed: '+name
    assert (data_root/'task2').is_dir() and (data_root/'corridors').is_dir(),'--data-root must be the kaggle_public directory'
    allowed=[package,data_root,work,HERE,*[Path(p).resolve() for p in extra_allowed]];forbidden=[Path(p).resolve() for p in cfg['forbidden_original_roots']]
    def inside(p,root):return p==root or p.is_relative_to(root)
    if io_audit:
        def hook(event,values):
            if event not in ['open','os.listdir','os.scandir'] or not values or isinstance(values[0],int):return
            try:p=Path(os.fsdecode(values[0])).resolve()
            except (TypeError,ValueError,OSError):return
            if any(inside(p,r) for r in allowed):return
            if any(inside(p,r) for r in forbidden):raise RuntimeError('Original experiment/cache access blocked: '+str(p))
        sys.addaudithook(hook)
    import lightgbm as lgb
    native=[];original_init=lgb.Booster.__init__
    def checked_init(self,*args,**kw):
        name=kw.get('model_file',args[2] if len(args)>2 else None)
        if name is not None:
            path=Path(name).resolve();assert inside(path,package),'Model outside portable archive';native.append(str(path.relative_to(package)))
        original_init(self,*args,**kw)
    lgb.Booster.__init__=checked_init
    entry=package/'payload/outputs'/cfg['entry_round'];sys.path.insert(0,str(entry));module=importlib.import_module(cfg['module']);model=getattr(module,cfg['class'])(entry)
    import ramp_features as rf
    rf.DATA=data_root;rf.WORK=work/'ramps';rf.WORK.mkdir(parents=True,exist_ok=True);rf.day.cache_clear();rf.topology.cache_clear()
    import run_round2 as r2
    r2.DATA=data_root;r2.CACHE=work/'days';r2.CACHE.mkdir(parents=True,exist_ok=True);r2.arrays.cache_clear()
    import numpy as np
    from task2q.data import day,network,make_sample
    expected=read(expected_hashes or HERE/'indices/evaluation-array-hashes.json');networks={};days=set()
    def array_hash(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    def verify_arrays(key,arrays):
        golden=expected[key]
        for name,rec in golden.items():
            v=arrays[name];assert str(v.dtype)==rec['dtype'] and list(v.shape)==rec['shape'] and array_hash(v)==rec['sha256'],'Raw reconstruction changed: '+key+'/'+name
    def get_sample(e):
        nk=e['network_key'];dk=e['day_key'];panel=e['panel'];date=e.get('date',e['origin'][:10])
        if nk not in networks:
            net=network(data_root,panel);verify_arrays(nk,net);np.savez_compressed(r2.CACHE/(nk+'.npz'),**net);networks[nk]=net
        if dk not in days:
            values=day(data_root,panel,date,cache_dir=work/'public-day-cache');values={k:v for k,v in values.items() if isinstance(v,np.ndarray)};verify_arrays(dk,values);np.savez_compressed(r2.CACHE/(dk+'.npz'),**values);days.add(dk)
        s=make_sample(r2.arrays(dk),r2.arrays(nk),int(e['origin_slot']),panel)
        assert s['condition']==e['condition'] and s['has_future_queue']==e['has_future_queue'],'Window/label protocol differs from frozen roster'
        return s
    audit=dict(original_model_cache_IO_blocked=io_audit,package_models_only=True,native_models=native,generated_day_keys=days)
    return model,get_sample,r2,audit
