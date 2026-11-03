"""Stress the numerical stability of idealized time-resolved OD inversion."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from scipy.linalg import svd

SEED = 2026110344
SPREADS = [0,1,2,4]
CUTOFFS = [1e-12,1e-2]


def arrival_operator(times,delays,spread):
    delta=times[:,None]-delays
    if spread==0:
        return (delta==0).astype(float)
    kernel=np.exp(-.5*(delta/spread)**2)
    return kernel/kernel.sum(axis=0)


def main():
    out=d.OUT/'task4_hypothetical_time_conditioning'
    d.save(out/'protocol.json',dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,spreads=SPREADS,relative_l2_noise=.01,
        singular_value_cutoffs=CUTOFFS,
        design='Known simultaneous pulse, deterministic zone-index travel delay and known Gaussian arrival spread. Each cut independently deconvolved then consecutive origin counts differenced.',
        noise='One Gaussian direction per cut, reused across spreads and scaled to 1% observation L2. This is a signed numerical perturbation.',
        scope='Illustrative estimator sensitivity under hypothetical timing, not an optimal inverse bound or any actual detector performance. Retain negative reconstructed paths.',
        source_sha256=d.sha(__file__)))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        pathfile=d.DATA/'corridors'/panel/'network/path_set.csv'
        hashes[str(pathfile.relative_to(d.DATA))]=d.sha(pathfile)
        paths=pd.read_csv(pathfile,dtype=str)
        origin,dest=[paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                     for key in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        n=int(dest.max()+1)
        assert len(paths)==n*(n-1)//2 and len(set(zip(origin,dest)))==len(paths) and (origin<dest).all()
        seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}'.encode()).digest()[:8],'little')%(2**63)
        rng=np.random.default_rng(seed)
        truth=rng.integers(1,1000,size=len(paths))
        times=np.arange(-24,n+24)
        perturb=rng.standard_normal((n-1,len(times)))
        perturb/=np.linalg.norm(perturb,axis=1,keepdims=True)
        downstream=np.zeros((n,n))
        for cut in range(n-1):
            live=(origin<=cut)&(cut<dest)
            np.add.at(downstream[cut],origin[live],truth[live])
        np.testing.assert_array_equal(downstream[dest-1,origin]-downstream[dest,origin],truth)
        for spread in SPREADS:
            reconstructed=np.zeros((len(CUTOFFS),n,n))
            fractions=np.zeros((len(CUTOFFS),n-1))
            worst_condition=1.
            for cut in range(n-1):
                operator=arrival_operator(times,cut-np.arange(cut+1),spread)
                clean=operator@downstream[cut,:cut+1]
                observed=clean+.01*np.linalg.norm(clean)*perturb[cut]
                u,s,vt=svd(operator,full_matrices=False)
                worst_condition=max(worst_condition,float(s[0]/max(s[-1],np.finfo(float).tiny)))
                projections=u.T@observed
                for i,cutoff in enumerate(CUTOFFS):
                    keep=s>=cutoff*s[0]
                    reconstructed[i,cut,:cut+1]=vt[keep].T@(projections[keep]/s[keep])
                    fractions[i,cut]=keep.mean()
            od=reconstructed[:,dest-1,origin]-reconstructed[:,dest,origin]
            assert np.isfinite(od).all()
            folder=d.WORK/'task4_hypothetical_time_conditioning_cases'/panel
            folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'spread{spread}.npz'
            np.savez_compressed(file,seed=seed,truth=truth,perturbation=perturb,downstream=downstream,
                                reconstructed_od=od,retained_fractions=fractions)
            digest=d.sha(file)
            for i,cutoff in enumerate(CUTOFFS):
                rows.append(dict(panel=panel,spread=spread,cutoff=cutoff,seed=seed,
                                 worst_cut_condition=worst_condition,mean_retained_fraction=float(fractions[i].mean()),
                                 relative_od_L2=float(np.linalg.norm(od[i]-truth)/np.linalg.norm(truth)),
                                 negative_od_fraction=float(np.mean(od[i]<0)),artifact_sha256=digest))
        pd.DataFrame(rows).to_csv(out/'by_panel.csv',index=False)
        d.state(status='running',stage='hypothetical_time_conditioning',panel=panel)
    frame=pd.DataFrame(rows)
    frame.groupby(['spread','cutoff'])[['relative_od_L2','worst_cut_condition','mean_retained_fraction']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='hypothetical_time_conditioning_complete',
        next='Verify perturbation scale, singular-value filtering and OD error before publication.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
