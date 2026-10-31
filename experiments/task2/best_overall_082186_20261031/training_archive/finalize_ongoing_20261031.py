"""Verify and archive both predeclared ongoing experiments; preserve incumbents."""
import hashlib, json, os, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import train_ongoing_speed as speed
from predict_ongoing_roads import probability
BASE=speed.BASE;read=speed.read;write=speed.write
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def archive(root):
    write(root/'manifest.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'})
    with zipfile.ZipFile(root.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in root.rglob('*'):
            if p.is_file():z.write(p,Path(root.name)/p.relative_to(root))
    with zipfile.ZipFile(root.with_suffix('.zip')) as z:assert z.testzip() is None

def main():
    roots=[BASE/'outputs'/f'ongoing_{name}_20261031' for name in ['analog','roads']]
    for root in roots:
        assert (root/'February_results.json').exists()
        for p,h in read(root/'frozen_before_February.json').items():assert sha(BASE/p)==h
        for p,h in read(root/'frozen_onset_hashes.json').items():assert sha(BASE/p)==h
    root=roots[1];d,m=speed.query('january')
    labels=np.empty(len(d['X']),dtype='U16')
    for w in m['windows']:labels[w['start']:w['end']]=w['panel']
    ids=np.concatenate([np.flatnonzero(labels==p)[:32] for p in np.unique(labels)]);runtime={}
    for name in read(root/'January_results.json')['candidates']:
        p,threshold=probability(root,d['X'][ids],labels[ids],d['parent'][ids],name);expected=np.load(root/f'january_{name}_probability.npy')[ids];error=float(np.max(np.abs(p-expected)));assert error<=1e-7;runtime[name]=error
    source=read(speed.old.prior.REF.PACKAGE/'archive-manifest.json')
    for p,r in source['files'].items():assert sha(speed.old.prior.REF.PACKAGE/p)==r['sha256']
    pub=BASE/'work/repo/experiments/task2/best_overall_081796_20261030';manifest=read(pub/'manifest.json')
    for p,h in manifest.items():assert sha(pub/p)==h
    best=BASE/'outputs/best_overall_speed_20261030'
    for p,h in read(best/'model_selection.json')['model_archive_sha256'].items():assert sha(best/p)==h
    verification=dict(passed=True,independent_runtime_rows=len(ids),runtime_all8_roads=True,max_probability_error=runtime,onset_unchanged=True,R127_files_verified=len(source['files']),published081796_files_verified=len(manifest),prior0818656782_archive_hashes_verified=True,January_only_thresholds=True,February_parameter_selection=False)
    write(root/'acceptance.json',verification);shutil.copy2(BASE/'outputs/predict_ongoing_roads.py',root/'code/predict_ongoing_roads.py');shutil.copy2(__file__,root/'code'/Path(__file__).name)
    shutil.copy2(speed.OUT/'feature_names.json',root/'feature_names.json')
    analog=roots[0];code=analog/'code';shutil.copytree(speed.OUT/'code',code,dirs_exist_ok=True)
    for name in ['train_ongoing_analog.py','test_ongoing_analog.py']:shutil.copy2(BASE/'outputs'/name,code/name)
    shutil.copy2(BASE/'work/ongoing_analog_tests.txt',analog/'unit_tests.txt')
    write(analog/'acceptance.json',dict(passed=True,unit_tests=4,causal_feature_mutation_panels=8,distinct_prior_days_test=True,missing_future_excluded_test=True,unknown_current_fallback_test=True,onset_unchanged=True,January_only_thresholds=True,February_parameter_selection=False))
    allrows=[];leaders={};diagnostics=[];inc=read(speed.OUT/'overall_leaders.json')['best_measured_by_population']
    for pop in inc:
        options=[dict(inc[pop],experiment=speed.OUT.name,threshold=.3)]
        d,meta=speed.query(pop)
        support={(w['panel'],w['origin']):int((d['y'][w['start']:w['end']].astype(bool)&d['m'][w['start']:w['end']]).sum()) for w in meta['windows']}
        for rroot in roots:
            results=read(rroot/'February_results.json')[pop]
            scores={'R127':results['R127'],'incumbent':results['incumbent'],**{n:r['score'] for n,r in results['candidates'].items()}}
            for model,s in scores.items():
                errors={(w['panel'],w['origin']):w for w in s['window_errors']}
                for w in s['window_rows']:
                    key=(w['panel'],w['origin']);size=support[key];e=errors[key];bucket='1-6' if size<=6 else '7-30' if size<=30 else '31-120' if size<=120 else '121+'
                    diagnostics.append(dict(population=pop,experiment=rroot.name,model=model,panel=w['panel'],origin=w['origin'],known_positive_cells=size,support_bucket=bucket,window_IoU=w['iou'],FP=e['FP'],FN=e['FN']))
            for name,r in results['candidates'].items():
                s=r['score'];record=dict(population=pop,experiment=rroot.name,model=name,ongoing_IoU=s['onset_IoU'],overall_IoU=r['overall_IoU'],FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'],threshold=r['threshold'],January_feasible=read(rroot/'January_results.json')['candidates'][name]['feasible'],delta_vs_incumbent=r['IoU_vs_incumbent']['delta'],day_cluster_95CI=r['IoU_vs_incumbent']['day_cluster_95CI']);allrows.append(record)
                options.append(dict(ongoing_model=name,ongoing_IoU=s['onset_IoU'],fixed_onset_IoU=r['fixed_onset_IoU'],overall_IoU=r['overall_IoU'],experiment=rroot.name,threshold=r['threshold'],January_feasible=record['January_feasible']))
        leaders[pop]=max(options,key=lambda r:r['overall_IoU'])
    summary=dict(frozen_onset='early_full',best_measured_by_population=leaders,candidates=allrows,verification=verification,caveat='Observed February leaders, not independent test validation; no February parameter tuning. Official S_queue unavailable.')
    summaryfile=BASE/'outputs/ongoing_20261031_summary.json';write(summaryfile,summary)
    frame=pd.DataFrame(diagnostics);frame.to_csv(BASE/'outputs/ongoing_20261031_window_diagnostics.csv',index=False,encoding='utf-8-sig')
    frame.groupby(['population','experiment','model','support_bucket']).agg(windows=('window_IoU','size'),mean_window_IoU=('window_IoU','mean'),FP=('FP','sum'),FN=('FN','sum')).to_csv(BASE/'outputs/ongoing_20261031_support_summary.csv',encoding='utf-8-sig')
    for rroot in roots:
        comparison=pd.read_csv(rroot/'comparison.csv');comparison.loc[comparison['model']=='R127','model']='R127_ongoing + frozen_early_full_onset';comparison.to_csv(rroot/'comparison.csv',index=False,encoding='utf-8-sig')
        note='注意：下表overall均固定最好early_full onset；R127行只对照R127的ongoing头，不代表完整原版R127。完整原版R127在新口径overall为0.7959762085。\n'
        report=rroot/'report.txt';report.write_text(note+report.read_text(encoding='utf8'),encoding='utf8')
        archive(rroot)
    leader=leaders['new_phase_population']
    if leader['overall_IoU']>inc['new_phase_population']['overall_IoU']:
        saved=BASE/'outputs/best_overall_ongoing_20261031';saved.mkdir(exist_ok=True);source=BASE/'outputs'/leader['experiment'];config=read(source/'January_results.json')['candidates'][leader['ongoing_model']]
        for original,name in [(source.with_suffix('.zip'),'ongoing_models.zip'),(best/'frozen_onset.zip','frozen_onset.zip'),(best/'R127_frozen.zip','R127_frozen.zip')]:
            dest=saved/name
            if not dest.exists():os.link(original,dest)
        write(saved/'model_selection.json',dict(primary=leader,ongoing_config=config['config'],threshold_selected_on='January only',onset_threshold=[.5,.5,.5,.55,.55,.55],onset='Frozen early_full',IoU_vs_incumbent=read(source/'February_results.json')['new_phase_population']['candidates'][leader['ongoing_model']]['IoU_vs_incumbent'],status='Retained observed point-estimate leader, stability not established',model_archive_sha256={p.name:sha(p) for p in saved.glob('*.zip')}))
        (saved/'README.txt').write_text('最好onset固定；只用一月选门槛，无二月调参。此前0.817962444和0.818656782均原样保留。\n当前描述性最好：'+json.dumps(leader,ensure_ascii=False)+'\n详见../ongoing_20261031_summary.json；二月已被重复用于开发，非独立盲测。',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
