"""Do public input groups share observable count/prior mismatch patterns?"""
import common as d
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd
import hashlib
import json
import msvcrt
import traceback
from scipy.linalg import qr,solve_triangular
from datetime import datetime,timezone

O=d.OUT/'task4_count_residual_alignment'
SPLITS=['train','validation','private']
SEED=2026110322


def projector(X):
    if X.shape[1]==0:return np.empty((len(X),0))
    U,s,_=np.linalg.svd(X,full_matrices=False)
    rank=int((s>max(s[0],1)*1e-11).sum())
    return U[:,:rank]


def remove(values,Q):return values-(values@Q)@Q.T


def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def main():
    O.mkdir(parents=True,exist_ok=True)
    model_file=d.OUT/'task4_variance_controls_audit/full_train_models.json';eta=json.loads(model_file.read_text())['models']['extended_span']['parameter']
    protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,hypothesis='Cross-input prior predictability could partly reflect shared observable count/prior mismatches. Audit their cross-group alignment, retaining both raw and nuisance-projected comparisons; do not attribute covariance to hidden path errors without evidence.',inputs=SPLITS,alignment='Intersect link IDs across all three Task4 count files and assert incidence row equality; QR-select one common independent equation set. No PM/mainline/ramp observations used.',residual='(c-Ab)/median_positive(Ab) on common QR equations.',covariances=['uniform path covariance','fixed previously trained span covariance eta='+str(eta)],projection=['none','remove each group loaded-prior vector','remove each group loaded-prior and constant-path-background vectors'],reference='2000 independent standard Gaussian whitened vector pairs per road/kernel/projection, with the SAME nuisance projection spaces. Conditional illustrative independent-Gaussian reference only, not a distribution-free significance test or proof of noise generation.',source_sha256=d.sha(__file__),model_sha256=d.sha(model_file),scope='All public input groups previously exposed; this is an observable-space development diagnostic, not true OD identification, official score improvement or a new hard cross-group constraint. Frozen baseline unchanged, no Kaggle upload.')
    d.save(O/'protocol.json',protocol);rows=[];hashes={};inventories=[]
    for panel in d.night.predictor.load_routes():
        records={};ids={}
        for split in SPLITS:
            paths,A,c,b,inputs=base.load(panel,split);hashes.update(inputs)
            file=d.DATA/'task4'/panel/split/'synthetic_link_counts.csv'
            ids[split]=pd.read_csv(file,dtype={'link_id':str}).link_id.tolist();assert len(set(ids[split]))==len(ids[split])
            records[split]=dict(A=A,c=c,b=b)
        common=sorted(set.intersection(*(set(v) for v in ids.values())))
        assert common
        for split in SPLITS:
            lookup={link:i for i,link in enumerate(ids[split])};idx=[lookup[link] for link in common];records[split]['A']=records[split]['A'][idx];records[split]['c']=records[split]['c'][idx]
        A=records['train']['A'];assert all(np.array_equal(A,records[s]['A']) for s in SPLITS)
        _,R,piv=qr(A.T,mode='economic',pivoting=True);rank=int((np.abs(np.diag(R))>max(abs(R[0,0]),1)*1e-10).sum());idx=piv[:rank];A=A[idx]
        origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();span=dest-origin;weight=np.exp(eta*(span-span.mean())/span.max())
        inventories.append(dict(panel=panel,shared_link_count=len(common),common_rank=rank,path_count=len(paths)))
        for kernel,w in [('uniform',np.ones(len(paths))),('span',weight)]:
            C=(A*w)@A.T;C/=np.trace(C)/rank;C+=np.eye(rank)*1e-6;L=np.linalg.cholesky(C);vectors={};designs={}
            for split in SPLITS:
                b=records[split]['b'];loaded=A@b;scale=float(np.median(loaded[loaded>0]));r=(records[split]['c'][idx]-loaded)/scale
                vectors[split]=solve_triangular(L,r,lower=True)
                designs[split]=solve_triangular(L,np.column_stack([loaded/scale,A.sum(axis=1)*b.mean()/scale]),lower=True)
            for version,nuisance in [('none',0),('loaded_prior',1),('loaded_and_background',2)]:
                Q={s:projector(designs[s][:,:nuisance]) for s in SPLITS};observed={s:remove(vectors[s],Q[s]) for s in SPLITS}
                seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+panel+'/'+kernel+'/'+version).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed)
                null={s:remove(rng.normal(size=(2000,rank)),Q[s]) for s in SPLITS}
                for i,left in enumerate(SPLITS):
                    for right in SPLITS[i+1:]:
                        actual=cosine(observed[left],observed[right]);reference=np.sum(null[left]*null[right],axis=1)/(np.linalg.norm(null[left],axis=1)*np.linalg.norm(null[right],axis=1))
                        low,high=np.quantile(reference,[.025,.975]);rows.append(dict(panel=panel,kernel=kernel,nuisance=version,input_left=left,input_right=right,rank=rank,nuisance_rank_left=Q[left].shape[1],nuisance_rank_right=Q[right].shape[1],observed_cosine=actual,illustrative_independent_Gaussian_025=float(low),illustrative_independent_Gaussian_975=float(high),outside_model_95=bool(actual<low or actual>high),model_reference_right_tail=float((np.sum(reference>=actual)+1)/2001)))
        pd.DataFrame(rows).to_csv(O/'by_panel_pair.csv',index=False)
        d.state(status='running',stage='task4_count_residual_alignment',panel=panel,next='Audit shared observable mismatches only; Gaussian reference is conditional/illustrative, not hidden-OD or causal-noise evidence.')
    df=pd.DataFrame(rows);df.groupby(['kernel','nuisance']).agg(mean_observed_cosine=('observed_cosine','mean'),minimum_observed_cosine=('observed_cosine','min'),maximum_observed_cosine=('observed_cosine','max'),pairs_outside_illustrative_model_95=('outside_model_95','sum'),pairs=('panel','count')).to_csv(O/'summary.csv')
    pd.DataFrame(inventories).to_csv(O/'common_equation_inventory.csv',index=False);d.save(O/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='task4_count_residual_alignment_complete',next='Check all residual projections and covariance sensitivity before interpreting shared mismatch. This cannot establish true OD errors or authorize unverified hard constraints. No Kaggle upload.')


if __name__=='__main__':
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:main()
    except Exception as error:
        d.state(status='failed',stage='task4_count_residual_alignment',error=repr(error),next='Preserve registered link matching and projection choices; repair without selecting panels or kernels.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
