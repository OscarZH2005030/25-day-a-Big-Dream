"""Compare relocated released-input ridge inference with saved Task4 baseline."""
import json,os,sys
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
ROOT=Path(__file__).resolve().parents[1]
os.environ['TFB_RELEASE_ROOT']=str(ROOT/'data/kaggle_public')
sys.path.insert(0,str(ROOT/'research/task4_extended_20261101/base'))
from core import np,pd,load_panel,Operator
source=ROOT/'project/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
saved=pd.read_csv(source,dtype={'panel':str,'departure_time':str,'path_id':str})
rows=[]
for panel in sorted(saved.panel.unique()):
    p=load_panel(ROOT/'data/kaggle_public',panel);op=Operator(p.A)
    for index,split in [(1,'validation'),(2,'private')]:
        pred=op.ridge(p.counts[index],p.priors[index])
        token=p.templates[index].departure_time.unique();assert len(token)==1
        ref=saved[(saved.panel==panel)&(saved.departure_time==token[0])].set_index('path_id')
        assert len(ref)==len(pred) and not ref.index.duplicated().any()
        values=ref.reindex(p.paths.path_id).path_flow.to_numpy(float)
        assert np.isfinite(values).all()
        difference=float(abs(pred-values).max());assert difference<1e-6,(panel,split,difference)
        rows.append({'panel':panel,'split':split,'rows':len(pred),'max_abs_difference':difference})
report={'passed':True,'comparisons':rows,'scope':'Released-input baseline equality only; not hidden path labels or official Task4 score.'}
out=ROOT/'local_reports';out.mkdir(exist_ok=True)
(out/'task4_baseline_smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
