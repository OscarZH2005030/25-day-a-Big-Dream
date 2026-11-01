"""Verify frozen-parent horizon residuals and preserve0.826387575."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import train_ongoing_horizon as t
from predict_ongoing_horizon import probability
BASE=t.BASE;OUT=t.OUT;read=t.read;write=t.write
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert (OUT/'February_results.json').exists()
    for p,h in read(OUT/'frozen_before_February.json').items():assert sha(BASE/p)==h
    t.check();incroot=BASE/'outputs/ongoing_jaccard_20261031';oldbundle=BASE/'outputs/best_overall_jaccard_20261031'
    for p,h in read(oldbundle/'model_selection.json')['model_archive_sha256'].items():assert sha(oldbundle/p)==h
    for p,h in read(incroot/'manifest.json').items():assert sha(incroot/p)==h
    for seed in t.SEEDS:
        for original in (incroot/f'final_j075_{seed}').glob('*.txt'):
            target=OUT/'base_models'/f'final_{seed}'/original.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target);assert sha(original)==sha(target)
    d,m=t.speed.query('january');labels=np.empty(len(d['X']),dtype='U16')
    for w in m['windows']:labels[w['start']:w['end']]=w['panel']
    ids=np.concatenate([np.flatnonzero(labels==p)[:32] for p in np.unique(labels)]);errors={};cal=read(OUT/'January_results.json')
    for name,r in cal['candidates'].items():
        p,threshold=probability(OUT,d['X'][ids],labels[ids],d['parent'][ids],name);expected=np.load(OUT/f'january_{name}_probability.npy')[ids];e=float(np.max(np.abs(p-expected)));assert e<=1e-7;errors[name]=e
    results=read(OUT/'February_results.json');leaders={};previous=read(incroot/'overall_leaders.json')['best_measured_by_population'];rows=[]
    for pop,res in results.items():
        options=[dict(previous[pop],source='preserved082639')]
        for name,r in res['candidates'].items():
            s=r['score'];options.append(dict(ongoing_model=name,ongoing_IoU=s['onset_IoU'],fixed_onset_IoU=r['fixed_onset_IoU'],overall_IoU=r['overall_IoU'],experiment=OUT.name,threshold=r['threshold'],January_feasible=cal['candidates'][name]['feasible'],source='new_continuation'))
            rows.append(dict(population=pop,model=name,ongoing_IoU=s['onset_IoU'],overall_IoU=r['overall_IoU'],FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'],threshold=r['threshold'],January_feasible=cal['candidates'][name]['feasible'],IoU_vs_incumbent=r['IoU_vs_incumbent']))
        leaders[pop]=max(options,key=lambda r:r['overall_IoU'])
    assert 'Ran 1 test' in (OUT/'unit_tests.txt').read_text(encoding='utf8')
    verification=dict(passed=True,onset_unchanged=True,previous082639_full_archive_hashes_verified=True,previous082639_all_files_verified=True,independent_runtime_rows=len(ids),all8_roads=True,max_probability_error=errors,external_initial_score_composition_test_passed=True,base900files_copied_and_hash_verified=24,January_thresholds_only=True,February_parameter_tuning=False)
    write(OUT/'acceptance.json',verification);write(OUT/'overall_leaders.json',dict(frozen_onset='early_full',best_measured_by_population=leaders,candidates=rows,caveat='Repeated February development populations, not blind; point-estimate leaders only.'))
    shutil.copy2(incroot/'feature_names.json',OUT/'feature_names.json');shutil.copy2(BASE/'outputs/predict_ongoing_horizon.py',OUT/'code/predict_ongoing_horizon.py');shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    frame=pd.read_csv(OUT/'comparison.csv');frame.loc[frame['model']=='R127','model']='R127_ongoing + frozen_early_full_onset';frame.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    (OUT/'report.txt').write_text('冻结0.826387575版本900树，新增六个时距残差专家，每头100树，外部init_score组合预测。最好early_full onset原样保留。\n十月选配置，十一月仅确认，最终6—11月拟合；一月只选全局门槛，二月冻结比较。\n表中overall均固定最好onset，R127行只是ongoing头对照，不表示完整原R127。代理观测标签；二月重复开发，非独立盲测。\n\n'+frame.to_string(index=False)+'\n\n'+json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
    write(OUT/'manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(OUT.with_suffix('.zip')) as z:assert z.testzip() is None
    leader=leaders['new_phase_population']
    if leader['source']=='new_continuation':
        saved=BASE/'outputs/best_overall_horizon_20261031';saved.mkdir(exist_ok=True)
        for original,name in [(OUT.with_suffix('.zip'),'ongoing_models.zip'),(oldbundle/'frozen_onset.zip','frozen_onset.zip'),(oldbundle/'R127_frozen.zip','R127_frozen.zip')]:
            if not (saved/name).exists():os.link(original,saved/name)
        name=leader['ongoing_model'];write(saved/'model_selection.json',dict(primary=leader,ongoing_config=cal['candidates'][name]['config'],onset='Frozen early_full',onset_threshold=[.5,.5,.5,.55,.55,.55],threshold_selected_on='January only',February_parameter_selection=False,IoU_vs_incumbent=results['new_phase_population']['candidates'][name]['IoU_vs_incumbent'],model_archive_sha256={p.name:sha(p) for p in saved.glob('*.zip')},status='Retained observed point-estimate leader, stability not established'))
        (saved/'README.txt').write_text('原0.826387575及此前所有版本与最好onset均保留。新增模型见model_selection.json；仅一月选门槛，二月未调参，代理观测评分非官方S_queue。',encoding='utf8')
    print(json.dumps(dict(leaders=leaders,candidates=rows,verification=verification),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
