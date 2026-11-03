"""Fixed observable one-sided prior-count mismatch gate, artificial development test."""
import common as d
import task4_count_source_pair_stress as g
import numpy as np,pandas as pd,hashlib,msvcrt
from datetime import datetime,timezone
O=d.OUT/'task4_count_bias_gate';SEED=2026110329

def main():
    O.mkdir(parents=True,exist_ok=True)
    d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,cases=160,truths=g.family.TRUTHS,noises=g.family.NOISES,drift='moderate_drift',strength=.05,threshold=.1,hypothesis='Positive-background stress harms may be partly detectable from observable positive loaded-prior count bias; test a fixed rejection gate rather than relying on weak-prior CV.',gate='Reject cross-input calibration if any of three supplied prior count totals exceed their own observed count total by more than10%. No fitting or threshold search. Rejection returns baseline exactly.',methods=['baseline','raw005','bias_gated005'],scope='New seed in adaptively studied artificial mechanisms, moderate drift only, not whole-system blind, complete noise coverage or official OD validation.',source_sha256=d.sha(__file__),generator_sha256=d.sha(g.__file__)))
    rows=[];hashes={}
    for panel in d.night.predictor.load_routes():
        paths,A,real_c,_,inputs=g.family.base.load(panel,'train');hashes.update(inputs);X=g.family.structure.features(paths)['span_origin_destination'];level=float(np.median(real_c[real_c>0]))
        for truth_name in g.family.TRUTHS:
            for noise in g.family.NOISES:
                case=f'{panel}/{truth_name}/{noise}';seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
                truth,b,c,dest=g.draw(paths,A,level,truth_name,noise,'moderate_drift',rng);order=rng.permutation(len(paths));fold=np.empty(len(paths),int);fold[order]=np.arange(len(paths))%5
                bias=np.array([(float((A@prior).sum())-float(count.sum()))/float(count.sum()) for prior,count in zip(b,c)]);gate=bool((bias<=.1).all())
                # Forecast uses only priors and geometry, retaining the earlier ridge-source recipe.
                source=np.mean([v/max(float(np.median(v[v>0])),1e-9) for v in b[:2]],axis=0);z=source-X@g.qr.ridge_qr(X,source);design=np.column_stack([X,z]);forecast=np.zeros(len(paths));coefs=[]
                for k in range(5):
                    train=fold!=k;test=~train;scale=max(float(np.median(b[2][train&(b[2]>0)])),1e-9);coef=g.qr.ridge_qr(design[train],b[2][train]/scale);coefs.append(coef);forecast[test]=np.maximum(design[test]@coef,0)*scale
                f0,s0=g.family.base.ridge(A,c[2],b[2]);direct,sd=g.family.base.ridge(A,c[2],.95*b[2]+.05*forecast);assert s0['success'] and sd['success'];flows=dict(baseline=f0,raw005=direct,bias_gated005=direct if gate else f0)
                folder=d.WORK/'task4_count_bias_cases'/hashlib.sha256(case.encode()).hexdigest()[:16];folder.mkdir(parents=True,exist_ok=True);file=folder/'inputs.npz';np.savez_compressed(file,truth=truth,prior=b,counts=c,fold=fold,seed=seed,bias=bias,gate_enabled=gate,coefs=coefs);np.savez_compressed(folder/'predictions.npz',**flows)
                for name,f in flows.items():rows.append(dict(case=case,panel=panel,truth=truth_name,noise=noise,seed=seed,method=name,gate_enabled=gate,max_loaded_prior_bias=float(bias.max()),input_sha256=d.sha(file),**g.family.base.metrics(f,truth[2],b[2],A,c[2],dest)))
        pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False);d.state(status='running',stage='count_bias_gate_stress',panel=panel,cases=len(rows)//3,next='Frozen observable rejection rule, retain benefits lost and harms remaining; no official candidate.')
    df=pd.DataFrame(rows);ref=df[df.method=='baseline'].set_index('case');summaries=[]
    for (noise,name),group in df.groupby(['noise','method']):
        q=group.set_index('case');delta=q.synthetic_S_ODME-ref.loc[q.index].synthetic_S_ODME;od=q.synthetic_S_od-ref.loc[q.index].synthetic_S_od
        summaries.append(dict(noise=noise,method=name,cases=len(q),enabled=int(q.gate_enabled.sum()),mean_delta=float(delta.mean()),worst_delta=float(delta.min()),harms=int((delta < -1e-12).sum()),OD_harms=int((od < -1e-12).sum())))
    pd.DataFrame(summaries).to_csv(O/'by_noise.csv',index=False);d.save(O/'input_sha256.json',hashes);assert df.case.nunique()==160
    d.state(status='stage_complete',stage='count_bias_gate_complete',next='Independent score/gate readback and standalone reproduction required; examine all missed benefits and remaining harms, no posthoc threshold tuning or official safety claim.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
