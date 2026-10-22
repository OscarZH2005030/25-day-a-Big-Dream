from pathlib import Path
import numpy as np
import pandas as pd
from task1.cache import snapshot
from task1.data import save_json, sha256
from task1.transformer_priority.common import read, frozen_plan
from transformer_fusion.core import summarize
from transformer_fusion.source import verify
from .common import BATCHES, CONFIG, AUGMENT, SUMMARY_METHODS, recipes
from .engine import engine_for

HISTORICAL=0.9369103120317692
ANCHORS=['COMPLETE.json','artifact_manifest.json','run_config.json','holdout_metrics_by_panel_regime.csv',
         'holdout_daily_errors.csv','holdout_preservation_checks.json','model_cache_usage.json']


def old_sources(project):
    return [dict(kind='d7',path=str(project/'runs/transformer_training_dev_20261019_183939_881805'),panels=['D7_I405_N','D7_I405_S']),
            dict(kind='i5',path=str(project/'runs/d12_i5_dynamic_dev_20261020_124730_964790'),panels=['D12_I5_N','D12_I5_S'])]


def validate_source_config(cfg,spec):
    if cfg['smoke'] or cfg['panels']!=spec['panels'] or cfg['training']!=CONFIG or cfg['augmentation']!=AUGMENT:
        raise ValueError('Formal source scope/budget/augmentation differs')
    for field in ['history_dates','training_dates','internal_validation_dates','holdout_dates']:
        if cfg['split'][field]!=frozen_plan()[field]:
            raise ValueError('Formal source dates differ')
    if cfg['fixed_fusion']!=recipes('rebuild_t0'):
        raise ValueError('Formal source fixed fusion differs')
    expected={'d7':'t1_training_ablations_v1','i5':'d12_i5_t1_dynamic_replication_v1','batch':'remaining_t1_dynamic_batch_v1'}[spec['kind']]
    if cfg['experiment']!=expected:
        raise ValueError('Wrong source experiment')


def anchor_sources(specs):
    files=[]
    for spec in specs:
        out=Path(spec['path'])
        validate_source_config(read(out/'run_config.json'),spec)
        complete=read(out/'COMPLETE.json')
        if not complete['integrity_passed'] or sha256(out/'artifact_manifest.json')!=complete['artifact_manifest_sha256']:
            raise ValueError('Source completion manifest differs')
        saved={str(Path(x['path']).resolve()):x for x in read(out/'artifact_manifest.json')}
        for name in ANCHORS:
            if name not in ['COMPLETE.json','artifact_manifest.json']:
                item=saved.get(str((out/name).resolve()))
                if item is None or sha256(out/name)!=item['sha256']:
                    raise ValueError('Source anchor differs: '+str(out/name))
        files.extend(out/name for name in ANCHORS)
    return snapshot(files)


def verify_source(spec):
    out=Path(spec['path'])
    if spec['kind']=='d7':
        from transformer_training.runner import verify_complete
    elif spec['kind']=='i5':
        from transformer_cross_corridor.runner import verify_complete
    else:
        verify_complete=engine_for(spec['batch']).verify_complete
    cfg,_=verify_complete(out)
    validate_source_config(cfg,spec)
    return cfg


def validate_tables(detail,daily,preservation,spec,reference):
    detail=detail.loc[detail.method.isin(SUMMARY_METHODS)].copy()
    daily=daily.loc[daily.method.isin(SUMMARY_METHODS)].copy()
    if set(detail.panel)!=set(spec['panels']) or set(detail.method)!=set(SUMMARY_METHODS):
        raise ValueError('Missing/extra source directions or methods')
    if detail.duplicated(['panel','method','regime']).any() or daily.duplicated(['panel','method','date']).any():
        raise ValueError('Duplicate source cells/dates')
    for panel in spec['panels']:
        for method in SUMMARY_METHODS:
            g=detail.loc[detail.panel.eq(panel)&detail.method.eq(method)]
            d=daily.loc[daily.panel.eq(panel)&daily.method.eq(method)]
            p=preservation[panel+'/'+method]
            if len(g)!=3 or set(g.regime)!={'R1','R2','R3'} or g.n.sum()!=reference.loc[panel,'n']:
                raise ValueError('Formal target/regime count differs')
            if set(g.family)!={panel.rsplit('_',1)[0]} or set(d.family)!={panel.rsplit('_',1)[0]}:
                raise ValueError('Corridor family differs')
            if p['rows']!=reference.loc[panel,'n'] or p['isolated_rows']!=reference.loc[panel,'fallback_n'] or not p['isolated_exact']:
                raise ValueError('Original isolated fallback differs')
            if (len(d)!=55 or set(d.date)!=set(frozen_plan()['holdout_dates']) or set(d.mask_regime)!={'R1','R2','R3'}
                or d.n.sum()!=reference.loc[panel,'n']):
                raise ValueError('Incomplete formal calendar dates')
            if not np.isfinite(g[['S_state','rmse_speed_kmh','rmse_flow_per_lane']]).all().all() or not np.isfinite(d[['speed_mse','flow_lane_mse']]).all().all():
                raise ValueError('Nonfinite metrics')
            if (d[['speed_mse','flow_lane_mse']]<0).any().any() or (d.n<=0).any() or (g.n<=0).any():
                raise ValueError('Invalid error/count values')
            d=d.copy()
            d['v_sse']=d.speed_mse*d.n
            d['q_sse']=d.flow_lane_mse*d.n
            rebuilt=d.groupby('mask_regime')[['n','v_sse','q_sse']].sum()
            for r in g.itertuples(index=False):
                a=rebuilt.loc[r.regime]
                v,q=np.sqrt([a.v_sse/a.n,a.q_sse/a.n])
                s=.54*max(0,1-v/25)+.46*max(0,1-q/600)
                if a.n!=r.n or max(abs(v-r.rmse_speed_kmh),abs(q-r.rmse_flow_per_lane),abs(s-r.S_state))>1e-9:
                    raise ValueError('Daily errors do not rebuild regime score')
    return detail,daily


def aggregate_ten(detail,reference):
    if set(detail.panel)!=set(reference.index) or len(detail)!=10*3*len(SUMMARY_METHODS):
        raise ValueError('Ten-direction scope incomplete')
    if detail.duplicated(['method','panel','regime']).any():
        raise ValueError('Duplicate ten-direction score cells')
    table=summarize(detail)
    if not table.panel_count.eq(10).all() or not table.family_count.eq(5).all() or not table.all_regimes_present.all():
        raise ValueError('Macro aggregation scope differs')
    scores=table.set_index('method').S_state
    for column,method in [('delta_vs_T0','T0'),('delta_vs_T1','T1'),('delta_vs_current_fusion','Current_fusion')]:
        table[column]=table.S_state-scores[method]
    table['delta_vs_historical']=table.S_state-HISTORICAL
    return table


def paired_calendar_bootstrap(daily,reference,candidate,replicates=10000):
    """Pair whole dates across ten directions without singleton joint strata."""
    dates=sorted(daily.date.unique());panels=sorted(daily.panel.unique())
    if len(dates)!=55 or len(panels)!=10 or replicates<2:
        raise ValueError('Full ten-direction calendar required')
    cells={}
    for panel in panels:
        for method in [reference,candidate]:
            d=daily.loc[daily.panel.eq(panel)&daily.method.eq(method)].set_index('date')
            if d.index.has_duplicates or set(d.index)!=set(dates):
                raise ValueError('Incomplete paired calendar cells')
            for regime in ['R1','R2','R3']:
                g=d.loc[d.mask_regime.eq(regime)].reindex(dates)
                n=g.n.fillna(0).to_numpy()
                cells[panel,method,regime]=(n,g.speed_mse.fillna(0).to_numpy()*n,g.flow_lane_mse.fillna(0).to_numpy()*n)
    rng=np.random.default_rng(20261021)
    counts=rng.multinomial(55,np.full(55,1/55),size=replicates)
    rejected=0
    for attempt in range(100):
        valid=np.ones(replicates,dtype=bool)
        for n,_,_ in cells.values(): valid &= (counts@n)>0
        if valid.all(): break
        rejected+=int((~valid).sum())
        counts[~valid]=rng.multinomial(55,np.full(55,1/55),size=int((~valid).sum()))
    else: raise ValueError('Cannot resample with all direction/regime cells present')
    scores={}
    for method in [reference,candidate]:
        families={}
        for panel in panels:
            value=np.zeros(replicates)
            for regime in ['R1','R2','R3']:
                n,v,q=cells[panel,method,regime];total=counts@n
                speed=np.sqrt((counts@v)/total);flow=np.sqrt((counts@q)/total)
                value+=(.54*np.maximum(0,1-speed/25)+.46*np.maximum(0,1-flow/600))/3
            families.setdefault(panel.rsplit('_',1)[0],[]).append(value)
        scores[method]=np.mean([np.mean(v,axis=0) for v in families.values()],axis=0)
    lo,hi=np.quantile(scores[candidate]-scores[reference],[.025,.975])
    return dict(reference=reference,candidate=candidate,replicates=replicates,ci95_low=float(lo),ci95_high=float(hi),
        rejected_draws_missing_regime=rejected,
        sampling='Whole calendar dates paired across all ten directions; redraw samples missing a direction/regime',
        limitation='Descriptive fixed-model local holdout interval; no temporal-dependence, training or selection correction; not official test evidence')


def summarize_queue(out,project,cfg):
    if cfg['smoke'] or [b['name'] for b in cfg['batches']]!=[b['name'] for b in BATCHES]:
        raise ValueError('Only all three formal batches can produce a ten-direction result')
    specs=list(cfg['old_sources'])+[dict(kind='batch',batch=b['name'],path=str(out/'batches'/b['folder']),panels=b['panels']) for b in cfg['batches']]
    verify(cfg['old_source_anchors'])
    reference=pd.read_csv(project/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    details=[];days=[];versions=[];groups={k:[] for k in ['gap','flow_gap','regime_gap','congestion','isolation']}
    seen=set()
    for spec in specs:
        source_cfg=verify_source(spec)
        if seen.intersection(spec['panels']):
            raise ValueError('Repeated source directions')
        seen.update(spec['panels'])
        versions.append(source_cfg['versions'])
        folder=Path(spec['path'])
        d,day=validate_tables(pd.read_csv(folder/'holdout_metrics_by_panel_regime.csv'),pd.read_csv(folder/'holdout_daily_errors.csv'),
                             read(folder/'holdout_preservation_checks.json'),spec,reference)
        details.append(d);days.append(day)
        for kind in groups:
            for panel in spec['panels']:
                for method in SUMMARY_METHODS:
                    group=pd.read_csv(folder/'holdout'/panel/f'{method}_{kind}_errors.csv')
                    if group.n.sum()!=reference.loc[panel,'n'] or not group.method.eq(method).all() or not group.panel.eq(panel).all():
                        raise ValueError('Grouped diagnostic target scope differs')
                    groups[kind].append(group)
    if any(v!=cfg['versions'] for v in versions):
        raise ValueError('Source library versions differ from frozen queue')
    detail=pd.concat(details,ignore_index=True)
    daily=pd.concat(days,ignore_index=True)
    table=aggregate_ten(detail,reference)
    detail.to_csv(out/'ten_direction_metrics_by_panel_regime.csv',index=False)
    table.to_csv(out/'ten_direction_comparison.csv',index=False)
    daily.to_csv(out/'ten_direction_daily_errors.csv',index=False)
    panel=detail.groupby(['method','family','panel'],as_index=False).S_state.mean()
    panel.to_csv(out/'ten_direction_scores_by_panel.csv',index=False)
    panel.groupby(['method','family'],as_index=False).S_state.mean().to_csv(out/'ten_direction_scores_by_family.csv',index=False)
    for kind,frames in groups.items():
        pd.concat(frames,ignore_index=True).to_csv(out/('ten_direction_'+kind+'_errors.csv'),index=False)
    intervals=[]
    for reference_name,candidate in [('T1','A_dynamic'),('Current_fusion','A_dynamic_fixed')]:
        intervals.append(paired_calendar_bootstrap(daily,reference_name,candidate))
    pd.DataFrame(intervals).to_csv(out/'ten_direction_paired_calendar_bootstrap.csv',index=False)
    anchors=anchor_sources(specs)
    verify(cfg['old_source_anchors'])
    save_json(out/'summary_sources.json',dict(specs=specs,anchors=anchors))
    save_json(out/'summary_integrity.json',dict(passed=True,panels=10,families=5,regimes_each=3,
        methods=SUMMARY_METHODS,all_sources_formal=True,all_dates_exact=True,all_target_counts_exact=True,
        isolated_fallback_preserved=True,daily_rebuilds_scores=True,aggregation='regime -> direction -> family',
        historical_target=HISTORICAL,limitation='Single-seed repeatedly inspected local holdout; not an official test score'))
    return table


def summarize_smoke(out,cfg):
    # Never mix the four completed formal directions with short smoke models.
    frames=[pd.read_csv(out/'batches'/b['folder']/'holdout_metrics_by_panel_regime.csv') for b in cfg['batches']]
    detail=pd.concat(frames,ignore_index=True)
    detail=detail.loc[detail.method.isin(SUMMARY_METHODS)]
    table=summarize(detail)
    table.to_csv(out/'smoke_selected_directions_comparison.csv',index=False)
    save_json(out/'summary_integrity.json',dict(passed=True,scope='Selected short-model engineering directions only',
        panels=sum(len(b['panels']) for b in cfg['batches']),old_formal_scores_used=False,formal_ten_direction_score=False))
    return table
