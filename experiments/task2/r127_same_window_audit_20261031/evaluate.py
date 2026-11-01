"""Reproduce frozen R127 and0.82639 hybrid on the original February64 ongoing roster."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;MODEL=HERE.parent/'best_overall_082639_jaccard_20261031';sys.path.insert(0,str(MODEL/'code'))
from hybrid_ongoing import Hybrid,verify_manifest
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--model-root',type=Path,default=MODEL);ap.add_argument('--expected',type=Path,default=HERE/'results.json');a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True);chosen=a.model_root.resolve()
    if chosen==MODEL.resolve():verify_manifest(MODEL);model=Hybrid(MODEL)
    else:
        sys.path.insert(0,str(chosen/'code'));from hybrid_horizon import Hybrid as ResidualHybrid,verify_manifest as verify_residual;verify_residual(chosen);model=ResidualHybrid(chosen)
    repro=HERE.parent/'r127_reproduction_20261027';sys.path.insert(0,str(repro));from repro_common import setup
    parent,sample,_,audit=setup(a.package,a.data_root,a.work/'raw-cache',True,extra_allowed=(HERE,chosen,MODEL,HERE.parent/'best_overall_081796_20261030',a.work))
    class Capture:
        def probability(self,s):self.last=parent.probability(s);return self.last
    capture=Capture();model.attach_parent(capture);entries=read(HERE/'february64.json')['samples'];gold=read(a.expected);groups={name:{} for name in ['R127','current']};counts={name:dict(FP=0,FN=0) for name in groups}
    for i,e in enumerate(entries):
        s=sample(e);p=model.probability(s);preds=dict(R127=np.asarray(capture.last,dtype='float32')>=.25,current=model.decode(s,p));y=np.asarray(s['y'],bool);m=np.asarray(s['m'],bool)
        for name,b in preds.items():
            union=int(((b|y)&m).sum());inter=int((b&y&m).sum());v=inter/union if union else (1. if m.any() else 0.);groups[name].setdefault(e['panel'],[]).append(v);counts[name]['FP']+=int((b&~y&m).sum());counts[name]['FN']+=int((~b&y&m).sum())
        if (i+1)%16==0:print(i+1,len(entries),flush=True)
    reports={}
    for name,g in groups.items():
        perroad={p:float(np.mean(v)) for p,v in g.items()};assert len(perroad)==8 and all(len(v)==8 for v in g.values());expected=gold['R127' if name=='R127' else 'frozen_current_model']
        for p,v in perroad.items():assert abs(v-expected['per_road'][p])<1e-12,(name,p,v,expected['per_road'][p])
        score=float(np.mean(list(perroad.values())));assert abs(score-expected['onset_IoU'])<1e-12;reports[name]=dict(ongoing_IoU=score,per_road=perroad,**counts[name])
    (a.work/'verification.json').write_text(json.dumps(dict(passed=True,windows=64,per_road_scores_match_to_1e12=True,raw_hashes_verified=True,models_and_thresholds_unchanged=True,used_for_parameter_selection=False,reports=reports),indent=2),encoding='utf8');print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
