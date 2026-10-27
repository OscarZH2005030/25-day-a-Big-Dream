"""Portable entry point for the preserved, byte-identical frozen Task2 policy."""
import os,sys
if not sys.flags.utf8_mode:
    os.execve(sys.executable,[sys.executable,'-X','utf8',*sys.argv],dict(os.environ,PYTHONUTF8='1'))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
sys.dont_write_bytecode=True
import argparse,json,hashlib,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--split',choices=['validation','private'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--io-audit',action='store_true');ap.add_argument('--audit-output',type=Path);args=ap.parse_args();cfg=json.loads((ROOT/'profile.json').read_text(encoding='utf8'));manifest=json.loads((ROOT/'archive-manifest.json').read_text(encoding='utf8'))
    for name,rec in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==rec['sha256'],'Saved file changed: '+name
    forbidden=[Path(p).resolve() for p in cfg['forbidden_original_roots']];data=args.data_root.resolve();native=[]
    def inside(p,root):return p==root or p.is_relative_to(root)
    if args.io_audit:
        def hook(event,values):
            if event not in ['open','os.listdir','os.scandir'] or not values:return
            try:
                if isinstance(values[0],int):return
                p=Path(os.fsdecode(values[0])).resolve()
            except (TypeError,ValueError,OSError):return
            if inside(p,ROOT) or inside(p,data):return
            if any(inside(p,r) for r in forbidden):raise RuntimeError('Original model/cache access forbidden: '+str(p))
        sys.addaudithook(hook)
    import lightgbm as lgb
    original_init=lgb.Booster.__init__
    def booster_init(self,*pos,**kw):
        name=kw.get('model_file',pos[2] if len(pos)>2 else None)
        if name is not None:
            p=Path(name).resolve();assert inside(p,ROOT),'Native model must be inside saved package: '+str(p);native.append(str(p.relative_to(ROOT)))
        original_init(self,*pos,**kw)
    lgb.Booster.__init__=booster_init
    entry=ROOT/'payload/outputs'/cfg['entry_round'];sys.path.insert(0,str(entry));module=importlib.import_module(cfg['module']);model=getattr(module,cfg['class'])(entry)
    import ramp_features as rf
    rf.DATA=data;rf.WORK=ROOT/'work/ramp-cache';rf.WORK.mkdir(parents=True,exist_ok=True);rf.day.cache_clear()
    import pandas as pd
    from task2q.data import PANELS,released_sample
    from task2q.submission import validate_queue
    frames=[]
    for panel in PANELS:
        d=data/'task2'/panel/args.split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
        for row in index.itertuples():
            s=released_sample(data,panel,args.split,row.window_id);q=model.predict(s);keys=template[template.window_id==row.window_id].copy();lookup={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=q[h,keys.link_id.map(lookup).to_numpy()].astype(int);parts.append(keys)
        frame=pd.concat(parts,ignore_index=True);validate_queue(frame,template);frame['panel']=panel;frames.append(frame)
    result=pd.concat(frames,ignore_index=True);assert len(result)==87000;args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False);audit=dict(rows=len(result),official_S_queue=None,package_models_only=True,original_model_and_cache_access_blocked=args.io_audit,native_model_files=sorted(set(native)),data_root=str(data),no_Task1_predictions_used=True)
    if args.audit_output:args.audit_output.parent.mkdir(parents=True,exist_ok=True);args.audit_output.write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(dict(rows=len(result),native_models=len(set(native)),original_model_access_blocked=args.io_audit,official_S_queue=None)))
if __name__=='__main__':main()
