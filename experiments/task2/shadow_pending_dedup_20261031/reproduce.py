"""Portable frozen-shadow verification, metric replay and single-window inference."""
import argparse,hashlib,json,sys,zipfile,os
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    publication=read(HERE/'publication_manifest.json')
    for name,meta in publication['files'].items():
        target=(HERE/name).resolve();assert target.is_relative_to(HERE) and sha(target)==meta['sha256'],'Publication file differs: '+name
    manifest=read(HERE/'runtime_manifest.json');archive=HERE/manifest['archive']['path'];assert sha(archive)==manifest['archive']['sha256'],'Archive hash mismatch'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;assert set(z.namelist())==set(manifest['files'])
        for name,meta in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256'],name
    return manifest

def unpack(work):
    manifest=verify();work=work.resolve();work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(HERE/manifest['archive']['path']) as z:
        for name,meta in manifest['files'].items():
            target=(work/name).resolve();assert target.is_relative_to(work),'Unsafe archive member'
            if os.name=='nt' and len(str(target))>=248:raise ValueError('Use a shorter --work-dir; Windows path exceeds248 characters')
            if target.exists():assert sha(target)==meta['sha256'],'Existing runtime differs: '+str(target)
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    return work

def metrics():
    from evaluation_metrics import aggregate_metrics
    old=read(HERE/'reports/task2_deduplicated_windows.json');r127=[dict(panel=e['panel'],condition=e['condition'],window_id=e['date']+'T'+str(e['slot']),iou=e['R127']) for e in old];previous=[dict(e,iou=v['retained']) for e,v in zip(r127,old)];pooled=[];seen={}
    records=read(HERE/'reports/task2_pending_specialist.json')['reports']
    for pop in ['december','january','february_new','february_old']:
        for e in records[pop]['observed_binary_consensus']['windows']:
            t=e['window_id'];slot=int(t[11:13])*12+int(t[14:16])//5;key=(e['panel'],t[:10],slot)
            if key in seen:assert abs(seen[key]['iou']-e['iou'])<1e-12 and seen[key]['condition']==e['condition'];continue
            row=dict(e,window_id=t[:10]+'T'+str(slot));pooled.append(row);seen[key]=row
    scores={n:aggregate_metrics(v,strict_complete=True) for n,v in dict(R127=r127,retained=previous,pending_shadow=pooled).items()};assert len(pooled)==len(old)==1479
    expected=read(HERE/'reports/task2_pending_specialist_ledger.json')['pooled_scores']['proxy_IoU'];assert abs(scores['pending_shadow']['proxy_IoU']-expected)<1e-12
    return dict(scores={n:v['proxy_IoU'] for n,v in scores.items()},absolute_gain=scores['pending_shadow']['proxy_IoU']-scores['R127']['proxy_IoU'],unique_scored_windows=len(pooled),official_truth_available=False,repeated_development=True)

def runtime(args):
    work=unpack(args.work_dir);sys.path.insert(0,str(work/'work/task2-cascade'))
    from core import r
    from frozen_shadow_inference import FrozenShadow
    cfg=SimpleNamespace(package=args.r127_package.resolve(),data=args.data.resolve(),source=(HERE.parent/'r127_reproduction_20261027').resolve(),output=work/'outputs');cfg.output.mkdir(exist_ok=True)
    backend=r.setup(cfg);return FrozenShadow(cfg,backend),backend,work

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True);sub.add_parser('verify');sub.add_parser('metrics')
    for command in ['smoke','predict']:
        p=sub.add_parser(command)
        for n in ['data','r127-package','work-dir']:p.add_argument('--'+n,type=Path,required=True)
        if command=='predict':p.add_argument('--panel',required=True);p.add_argument('--origin',required=True);p.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='verify':m=verify();print(json.dumps(dict(archive_sha256=m['archive']['sha256'],runtime_files=len(m['files']),verified=True)));return
    if args.command=='metrics':print(json.dumps(metrics(),indent=2));return
    pipe,backend,work=runtime(args)
    import numpy as np
    from freeze_shadow_fresh import causal_sample
    if args.command=='smoke':
        result=[]
        for e in read(HERE/'reports/smoke_cases.json'):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;v=backend[1].load_day(e['panel'],e['origin'][:10]);s=causal_sample(v,backend[3][e['panel']],e['panel'],e['origin'][:10],slot);pred,_=pipe.predict(s)
            for name,value in pred.items():assert hashlib.sha256(value.astype('uint8').tobytes()).hexdigest()==e['prediction_sha256'][name],(e['panel'],name)
            result.append(dict(panel=e['panel'],origin=e['origin'],four_policies_exact=True))
        report=dict(passed=True,cases=result,score_replay=metrics());(work/'outputs/portable_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report));return
    from datetime import datetime,timezone
    d=datetime.fromisoformat(args.origin.replace('Z','+00:00'));assert d.tzinfo is not None,'Origin requires timezone';d=d.astimezone(timezone.utc);assert d.second==0 and d.minute%5==0;date=d.date().isoformat();slot=d.hour*12+d.minute//5;v=backend[1].load_day(args.panel,date);s=causal_sample(v,backend[3][args.panel],args.panel,date,slot);pred,_=pipe.predict(s);args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**pred);print(json.dumps(dict(output=str(args.output),policies=list(pred),panel=args.panel,origin=s['origin'],official_truth_available=False)))
if __name__=='__main__':main()
