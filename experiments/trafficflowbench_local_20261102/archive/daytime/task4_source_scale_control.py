"""Fresh fixed shared-background stress: isolate source normalization effects."""
import common as d
import task4_count_source_pair_stress as g
import numpy as np,pandas as pd,hashlib,msvcrt
from datetime import datetime,timezone
O=d.OUT/'task4_source_scale_control';SEED=2026110328

def crossfit(X,priors,adjusted,fold):
    scales=[max(float(np.median(v[v>0])),1e-9) for v in priors[:2]]
    features={}
    for name,values,den in [('raw',priors[:2],scales),('adjusted_own_scale',adjusted,[max(float(np.median(v[v>0])),1e-9) for v in adjusted]),('adjusted_raw_scale',adjusted,scales)]:
        s=np.mean([v/q for v,q in zip(values,den)],axis=0);z=s-X@g.qr.ridge_qr(X,s);features[name]=np.column_stack([X,z])
    b=priors[2];pred={name:np.zeros(len(b)) for name in features};coefs={name:[] for name in features}
    for k in range(5):
        train=fold!=k;test=~train;scale=max(float(np.median(b[train&(b>0)])),1e-9)
        for name,design in features.items():
            coef=g.qr.ridge_qr(design[train],b[train]/scale);coefs[name].append(coef);pred[name][test]=np.maximum(design[test]@coef,0)*scale
    return pred,coefs

def main():
    O.mkdir(parents=True,exist_ok=True)
    d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,cases=120,noise='shared_positive_background',truths=g.family.TRUTHS,drifts=g.family.DRIFTS,replicates=1,strength=.05,hypothesis='Earlier source-count control changed both source flows and positive-median normalization. Hold the original source scale fixed to isolate that confound.',methods=['baseline','raw','adjusted_own_scale','adjusted_raw_scale'],evaluation='New fixed draws in an already studied mechanism; not whole-system blind, hidden labels or proof of official improvement. All cases retained, no selected routing.',source_sha256=d.sha(__file__),generator_sha256=d.sha(g.__file__),original_baseline='Frozen; no real flow export or Kaggle upload.'))
    rows=[];hashes={}
    for panel in d.night.predictor.load_routes():
        paths,A,c0,_,inputs=g.family.base.load(panel,'train');hashes.update(inputs);X=g.family.structure.features(paths)['span_origin_destination'];level=float(np.median(c0[c0>0]))
        for truth_name in g.family.TRUTHS:
            for drift in g.family.DRIFTS:
                case=f'{panel}/{truth_name}/{drift}';seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
                truth,b,c,dest=g.draw(paths,A,level,truth_name,'shared_positive_background',drift,rng);order=rng.permutation(len(paths));fold=np.empty(len(paths),int);fold[order]=np.arange(len(paths))%5
                adjusted=[]
                for j in [0,1]:
                    estimate,status=g.family.base.ridge(A,c[j],b[j]);assert status['success'];adjusted.append(estimate)
                predictions,coefs=crossfit(X,b,adjusted,fold)
                folder=d.WORK/'task4_source_scale_cases'/hashlib.sha256(case.encode()).hexdigest()[:16];folder.mkdir(parents=True,exist_ok=True)
                artifact=folder/'inputs_and_models.npz';np.savez_compressed(artifact,truth=truth,prior=b,counts=c,source_estimates=adjusted,fold=fold,seed=seed,**{name+'_coef':np.array(value) for name,value in coefs.items()})
                flows={}
                for name in ['baseline',*predictions]:
                    prior=b[2] if name=='baseline' else .95*b[2]+.05*predictions[name]
                    f,status=g.family.base.ridge(A,c[2],prior);assert status['success'];flows[name]=f
                    rows.append(dict(case=case,panel=panel,truth=truth_name,drift=drift,seed=seed,method=name,input_sha256=d.sha(artifact),**g.family.base.metrics(f,truth[2],b[2],A,c[2],dest)))
                np.savez_compressed(folder/'predictions.npz',**flows)
        pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False)
        d.state(status='running',stage='source_scale_control',panel=panel,cases=len(rows)//4,next='Fresh paired scale ablation only; retain negative results, no official score claim.')
    frame=pd.DataFrame(rows);reference=frame[frame.method=='baseline'].set_index('case');summary=[]
    for name,group in frame.groupby('method'):
        q=group.set_index('case');delta=q.synthetic_S_ODME-reference.synthetic_S_ODME;od=q.synthetic_S_od-reference.synthetic_S_od
        summary.append(dict(method=name,cases=len(q),mean_delta=float(delta.mean()),worst_delta=float(delta.min()),harms=int((delta < -1e-12).sum()),OD_harms=int((od < -1e-12).sum())))
    pd.DataFrame(summary).to_csv(O/'paired_risk.csv',index=False);d.save(O/'input_sha256.json',hashes);assert frame.case.nunique()==120
    d.state(status='stage_complete',stage='source_scale_control_complete',next='Independently read back and fully reproduce new scale control before publishing. Results do not prove hidden path improvement.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
