"""Independent all-case score readback and standalone full reproduction."""
import common as d
import task4_span_shrink_stress as base
import pandas as pd,numpy as np
import hashlib,subprocess,sys,msvcrt

def main():
    original=d.OUT/'task4_source_scale_control';df=pd.read_csv(original/'all_cases.csv');errors=[];digests={}
    for panel,group in df.groupby('panel'):
        paths,A,_,_,_=base.load(panel,'train');dest=pd.factorize(paths.destination_zone,sort=True)[0]
        for case,rows in group.groupby('case'):
            key=hashlib.sha256(case.encode()).hexdigest()[:16];folder=d.WORK/'task4_source_scale_cases'/key
            z=np.load(folder/'inputs_and_models.npz');pred=np.load(folder/'predictions.npz');assert set(rows.input_sha256)=={d.sha(folder/'inputs_and_models.npz')}
            for name in ['inputs_and_models.npz','predictions.npz']:digests[key+'/'+name]=d.sha(folder/name)
            fstar=z['truth'][2];b=z['prior'][2];c=z['counts'][2];a=np.bincount(dest,weights=fstar);a/=a.sum()
            for row in rows.itertuples():
                f=pred[row.method];od=max(0,1-float(np.abs(f-fstar).sum()/fstar.sum()));link=max(0,1-float(np.abs(A@f-c).sum()/c.sum()));dev=float(np.exp(-abs(float(np.abs(f-b).sum()/max(np.abs(fstar-b).sum(),1e-12))-1)));at=np.bincount(dest,weights=f);at/=at.sum();attr=1-.5*float(np.abs(at-a).sum());score=.45*od+.25*link+.15*dev+.15*attr
                errors.extend(abs(x-getattr(row,k)) for x,k in zip([od,link,dev,attr,score],['synthetic_S_od','synthetic_S_link','synthetic_S_dev','synthetic_S_attr','synthetic_S_ODME']))
    assert len(df)==480 and max(errors)<1e-12
    d.save(original/'independent_readback.json',dict(cases=120,actions=480,max_score_error=max(errors),artifact_hashes=digests,source_sha256=d.sha(__file__)))
    out=d.WORK/'publication_smoke/source_scale';d.state(status='running',stage='source_scale_portable_reproduction',output=str(out),next='Complete fixed120-case standalone rerun and verify all saved artifacts before publication.')
    subprocess.run([sys.executable,'-X','utf8',str(d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_source_scale.py'),'--release-root',str(d.DATA),'--output',str(out)],check=True)
    diffs={}
    for name in ['all_cases.csv','paired_risk.csv']:
        a=pd.read_csv(original/name);b=pd.read_csv(out/'stress'/name);assert a.shape==b.shape and a.columns.equals(b.columns)
        numeric=a.select_dtypes(include='number').columns
        for col in a.columns.difference(numeric):assert a[col].equals(b[col]),col
        assert np.array_equal(a[numeric].isna(),b[numeric].isna());error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())));assert error<1e-12;diffs[name]=error
    for rel,digest in digests.items():assert d.sha(out/'work/task4_source_scale_cases'/rel)==digest,rel
    d.save(original/'portable_reproduction.json',dict(cases=120,actions=480,maximum_CSV_differences=diffs,all240_artifact_hashes_identical=True,source_sha256=d.sha(__file__),scope='Artificial paired ablation only, not official improvement.'))
    d.state(status='stage_complete',stage='source_scale_reproduction_complete',next='Publish verified scale-confound findings and limits; baseline frozen, no Kaggle upload.')

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
