"""Replay the exact published Task1 weights; export public historical diagnostic evidence.
No model fitting, threshold selection, Kaggle authentication or submission.
The archived retrained caches provide official target KEYS ONLY, never predictions.
"""
import argparse,importlib.util,json,os,sys,time,warnings
from pathlib import Path
sys.dont_write_bytecode=True
for envkey in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[envkey]='4'
import numpy as np,pandas as pd,torch
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'local_reports/physics_diagnostic')
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
args=parser.parse_args()
result_dir=args.output.resolve()
for sub in ['predictions','errors','transitions']:(result_dir/sub).mkdir(parents=True,exist_ok=True)
root=ROOT/'project/experiments/task1/best_road_combination_20261026'
data=ROOT/'data/kaggle_public'
keyroot=ROOT/'archives/retrained_task1/work/evaluation/train'
out=result_dir/'predictions'
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('portable',ROOT/'work/tfb_submit/complete/code/portable_entry.py')
portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable);portable.install_unicode_loader()
import predict_selected as pm
torch.set_num_threads(4)
routes=pm.load_routes()
readpq=pd.read_parquet
device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
def infer(pn,date,ks):
    pm.keys_for=lambda *unused:ks.copy()
    def guard(path,*a,**kw):
        if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Truth read inside inference')
        return readpq(path,*a,**kw)
    pd.read_parquet=guard
    try:return pm.predict(pn,date,routes[pn],data,out,device)
    finally:pd.read_parquet=readpq

ref=json.loads((root/'reproduction/acceptance/published-full55-evaluation.json').read_text(encoding='utf-8'))
all_rows=[];all_pairs=[];examples=[];daily_checks=[];per_road=[];sample_plan=[];start=time.time()
def summary(rows,pairs):
    r={n:np.concatenate([x[n] for x in rows]) for n in ['ev','eq','rv','rq','ek','kt','tq','tv']}
    ev,eq,rv,rq,ek,kt,tq,tv=[r[n] for n in ['ev','eq','rv','rq','ek','kt','tq','tv']]
    opp=ev*eq<0;large=(abs(rv)>=.01)&(abs(rq)>=.01);sse=float((ek*ek).sum());sep=float((tq*tq+tv*tv).sum())
    z={n:np.concatenate([x[n] for x in pairs]) for n in ['a','b','both','td','extreme']}
    a,b,both,td=[z[n] for n in ['a','b','both','td']];diff=b-a;flip=(a*b<0)&both
    return dict(same_high_count=int(((ev>0)&(eq>0)).sum()),same_low_count=int(((ev<0)&(eq<0)).sum()),any_exact_zero_count=int(((ev==0)|(eq==0)).sum()),opposite_speed_low_flow_high_count=int(((ev<0)&(eq>0)).sum()),opposite_speed_high_flow_low_count=int(((ev>0)&(eq<0)).sum()),density_abs_rel_gt25pct_count=int((abs(ek)/kt>.25).sum()),density_abs_rel_gt50pct_count=int((abs(ek)/kt>.5).sum()),density_abs_rel_gt100pct_count=int((abs(ek)/kt>1).sum()),density_sse_share_rel_gt50pct=float(100*(ek[abs(ek)/kt>.5]**2).sum()/max(sse,1e-30)),density_absolute_error_share_rel_gt50pct=float(100*abs(ek[abs(ek)/kt>.5]).sum()/max(float(abs(ek).sum()),1e-30)),temporal_sse_share_density_outlier_pairs_pct=float(100*(diff[z['extreme']]**2).sum()/max(float((diff*diff).sum()),1e-30)),temporal_absolute_error_share_density_outlier_pairs_pct=float(100*abs(diff[z['extreme']]).sum()/max(float(abs(diff).sum()),1e-30)),rows=len(ev),speed_bias_kmh=float(ev.mean()),flow_bias_vph=float(eq.mean()),opposite_speed_flow_pct=float(100*opp.mean()),both_relative_error_ge1pct=int(large.sum()),opposite_given_both1pct_pct=float(100*opp[large].mean()) if large.any() else None,density_nmae=float(abs(ek).sum()/kt.sum()),density_relative_error_median=float(np.median(abs(ek)/kt)),density_relative_error_p90=float(np.quantile(abs(ek)/kt,.9)),density_relative_error_p99=float(np.quantile(abs(ek)/kt,.99)),density_sse_share_opposite_pct=float(100*(ek[opp]**2).sum()/max(sse,1e-30)),density_energy_ratio=float(sse/max(sep,1e-30)),n_transitions_any_target=len(a),n_transitions_both_target=int(both.sum()),opposite_density_errors_both_targets_pct=float(100*flip.sum()/max(int(both.sum()),1)),density_lag1_correlation_both_targets=float(np.corrcoef(a[both],b[both])[0,1]) if both.sum()>2 else None,temporal_energy_ratio=float((diff*diff).sum()/max(float((a*a+b*b).sum()),1e-30)),temporal_energy_ratio_both_targets=float((diff[both]**2).sum()/max(float((a[both]**2+b[both]**2).sum()),1e-30)),temporal_sse_share_sign_flip_pct=float(100*(diff[flip]**2).sum()/max(float((diff*diff).sum()),1e-30)),stock_difference_error_mae_vehicles=float(abs(diff).mean()),stock_difference_error_over_true_change=float(abs(diff).sum()/max(float(abs(td).sum()),1e-30)))
print(json.dumps({'scope':'same frozen weights, first 2 holdout days per regime, 10 roads; local evidence export only; no training or submission'}),flush=True)
for road in ref['roads']:
    pn=road['panel'];records=[]
    for regime in ['R1','R2','R3']:records.extend([x for x in road['days'] if x['regime']==regime][:2])
    records=sorted(records,key=lambda x:x['date']);rows=[];pairs=[]
    fd=pd.read_csv(data/'corridors'/pn/'network/fd_parameters.csv').set_index('link_id')
    tfiles={p.stem[-10:].replace('_','-'):p for p in (data/'corridors'/pn/'train/mainline_states').rglob('*.parquet')}
    for record in records:
        day=record['date'];sample_plan.append(dict(panel=pn,**record));ks=readpq(keyroot/(pn+'_'+day+'.parquet'),columns=pm.base.KEYS)
        assert len(ks)==record['rows'] and not ks.duplicated(pm.base.KEYS).any()
        pr=infer(pn,day,ks);tr=readpq(tfiles[day]);tr.timestamp=tr.timestamp.astype(str)
        k=['timestamp','station_id','link_id'];m=pr.merge(tr[k+['speed_kmh','flow_vph','is_score_eligible']],on=k,validate='one_to_one',suffixes=('_pred','_true'))
        assert len(m)==len(pr) and m.is_score_eligible.astype(bool).all()
        vp=m.speed_kmh_pred.to_numpy();vt=m.speed_kmh_true.to_numpy();qp=m.flow_vph_pred.to_numpy();qt=m.flow_vph_true.to_numpy();ev=vp-vt;eq=qp-qt
        lanes=m.link_id.map(fd.lanes).to_numpy();ssev=float(ev@ev);sseq=float((eq/lanes)@(eq/lanes))
        ds=abs(ssev-record['speed_sse'])/max(record['speed_sse'],1);dq=abs(sseq-record['flow_sse'])/max(record['flow_sse'],1)
        daily_checks.append(dict(panel=pn,date=day,speed_sse_rel_diff=ds,flow_sse_rel_diff=dq))
        assert max(ds,dq)<1e-4,('published SSE mismatch',daily_checks[-1])
        vc=np.maximum(vp,1);vct=np.maximum(vt,1);kt=qt/vct;tq=eq/vc;tv=qt*(1/vc-1/vct);ek=qp/vc-kt
        assert (kt>0).all() and np.allclose(tq+tv,ek,atol=1e-9)
        error_frame=m.copy()
        for name,value in dict(speed_error=ev,flow_error=eq,density_true=kt,density_pred=qp/vc,density_error=ek,speed_relative_error=ev/vct,flow_relative_error=eq/np.maximum(qt,1),opposite_speed_flow=(ev*eq<0)).items():error_frame[name]=value
        error_frame.to_parquet(result_dir/'errors'/f'{pn}_{day}.parquet',index=False)
        rr=dict(ev=ev,eq=eq,rv=ev/vct,rq=eq/np.maximum(qt,1),ek=ek,kt=kt,tq=tq,tv=tv);rows.append(rr);all_rows.append(rr)
        sig=(abs(ev/vct)>=.02)&(abs(eq/np.maximum(qt,1))>=.02)&(ev*eq<0)
        idxs=np.flatnonzero(sig)
        if len(idxs):
            ix=idxs[np.argmax(abs(ek[idxs])/kt[idxs])]
            examples.append(dict(panel=pn,date=day,timestamp=str(m.timestamp.iloc[ix]),link_id=str(m.link_id.iloc[ix]),true_speed=float(vt[ix]),pred_speed=float(vp[ix]),true_flow=float(qt[ix]),pred_flow=float(qp[ix]),density_relative_error=float(ek[ix]/kt[ix])))
        full=tr.loc[tr.is_score_eligible.astype(bool),k+['speed_kmh','flow_vph']].merge(pr[k+['speed_kmh','flow_vph']],on=k,how='left',validate='one_to_one',suffixes=('_true','_pred'))
        full['target']=full.speed_kmh_pred.notna();full['vp']=full.speed_kmh_pred.fillna(full.speed_kmh_true);full['qp']=full.flow_vph_pred.fillna(full.flow_vph_true)
        g=full.groupby(['link_id','timestamp'],as_index=False).agg(vt=('speed_kmh_true','mean'),qt=('flow_vph_true','mean'),vp=('vp','mean'),qp=('qp','mean'),target=('target','max'))
        g['timestamp']=pd.to_datetime(g.timestamp,utc=True);g=g.sort_values(['link_id','timestamp']);length=g.link_id.map(fd.length_km).to_numpy();assert np.isfinite(length).all()
        g['Nt']=length*g.qt/np.maximum(g.vt,1);g['eN']=length*g.qp/np.maximum(g.vp,1)-g.Nt
        nxt=g.groupby('link_id')[['timestamp','eN','target','Nt']].shift(-1);nt=nxt.target.fillna(False).astype(bool)
        valid=((nxt.timestamp-g.timestamp).dt.total_seconds()==300)&(g.target|nt);both=g.target&nt
        pp=dict(a=g.eN.to_numpy()[valid],b=nxt.eN.to_numpy()[valid],both=both.to_numpy()[valid],td=(nxt.Nt-g.Nt).to_numpy()[valid]);pp['extreme']=((abs(g.eN)/np.maximum(g.Nt,1e-12)>.5)|(abs(nxt.eN)/np.maximum(nxt.Nt,1e-12)>.5)).to_numpy()[valid];pairs.append(pp);all_pairs.append(pp)
        pair_frame=pd.DataFrame({'panel':pn,'date':day,'link_id':g.link_id.to_numpy()[valid],'timestamp':g.timestamp.to_numpy()[valid],'next_timestamp':nxt.timestamp.to_numpy()[valid],'N_true':g.Nt.to_numpy()[valid],'N_true_next':nxt.Nt.to_numpy()[valid],'N_error':pp['a'],'N_error_next':pp['b'],'delta_N_true':pp['td'],'delta_N_error':pp['b']-pp['a'],'both_target':pp['both'],'density_outlier_either':pp['extreme']})
        pair_frame.to_parquet(result_dir/'transitions'/f'{pn}_{day}.parquet',index=False)
    road_result={'panel':pn,'dates':[x['date'] for x in records],**summary(rows,pairs),'elapsed_s':round(time.time()-start,1)};per_road.append(road_result);print(json.dumps(road_result,ensure_ascii=False),flush=True)
report={'GLOBAL':summary(all_rows,all_pairs),'daily_reference_checks':len(daily_checks),'max_daily_SSE_relative_difference':max(max(x['speed_sse_rel_diff'],x['flow_sse_rel_diff']) for x in daily_checks),'examples':sorted(examples,key=lambda x:abs(x['density_relative_error']),reverse=True)[:10],'scope':'Public historical holdout sample; density=flow/max(speed,1); visible states restored from public eligible truth; only consecutive 300-second within-day link transitions. No official Task3 score or private boundaries.','per_road':per_road,'daily_checks':daily_checks}
(result_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(result_dir/'sample_plan.json').write_text(json.dumps(sample_plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['per_road','daily_checks']},ensure_ascii=False),flush=True)
