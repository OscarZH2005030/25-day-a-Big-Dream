from pathlib import Path
from types import SimpleNamespace
import copy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from task1.data import save_json
from task1.transformer_priority.common import read,frozen_plan
from transformer_remaining import common
from transformer_remaining.engine import engine_for
from transformer_remaining.summary import validate_tables,aggregate_ten,validate_source_config,summarize_queue,paired_calendar_bootstrap


def test_engines_are_isolated_and_do_not_modify_completed_experiment():
    import transformer_cross_corridor.runner as original
    before=original.PANELS.copy()
    a,b=engine_for('i210'),engine_for('i10')
    assert a.PANELS==['D7_I210_E','D7_I210_W'] and b.PANELS==['D7_I10_E','D7_I10_W']
    assert a.run.__globals__ is not b.run.__globals__
    assert original.PANELS==before and original.EXPERIMENT=='d12_i5_t1_dynamic_replication_v1'
    a.PANELS.append('invalid')
    assert len(b.PANELS)==2 and original.PANELS==before
    selected=common.selected_batches(['i210']);selected[0]['panels'][0]='invalid'
    assert common.batch_for('i210')['panels'][0]=='D7_I210_E'


@pytest.mark.parametrize('names',[[],['i10','i210'],['i210','i210'],['unknown']])
def test_queue_rejects_changed_order_or_duplicate_batches(names):
    with pytest.raises(ValueError): common.selected_batches(names)


def source_config(panels):
    return dict(experiment=common.EXPERIMENT,smoke=False,panels=panels,training=dict(common.CONFIG),
        augmentation=copy.deepcopy(common.AUGMENT),split=common.plan(panels),fixed_fusion=common.recipes('rebuild_t0'))


@pytest.mark.parametrize('change',['smoke','dates','budget','fusion','panels'])
def test_formal_summary_rejects_smoke_or_protocol_drift(change):
    panels=common.batch_for('i210')['panels'];spec=dict(kind='batch',panels=panels)
    cfg=source_config(panels)
    validate_source_config(cfg,spec)
    if change=='smoke':cfg['smoke']=True
    if change=='dates':cfg['split']['holdout_dates']=cfg['split']['holdout_dates'][:-1]
    if change=='budget':cfg['training']['epochs']=61
    if change=='fusion':cfg['fixed_fusion']['flow']=.25
    if change=='panels':cfg['panels']=['D12_I5_N','D12_I5_S']
    with pytest.raises(ValueError):validate_source_config(cfg,spec)


def synthetic_ten():
    ref=pd.read_csv(Path(__file__).resolve().parents[1]/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    dates=frozen_plan()['holdout_dates'];rows=[];days=[];checks={}
    for index,panel in enumerate(ref.index):
        n=int(ref.loc[panel,'n']);ns=[n//55+int(i<n%55) for i in range(55)]
        for m,method in enumerate(common.SUMMARY_METHODS):
            v=1+index*.11-m*.02;q=20+index*2-m*.1
            score=.54*(1-v/25)+.46*(1-q/600)
            checks[panel+'/'+method]=dict(rows=n,isolated_rows=int(ref.loc[panel,'fallback_n']),isolated_exact=True)
            for i,date in enumerate(dates):
                days.append(dict(panel=panel,family=panel.rsplit('_',1)[0],method=method,date=date,
                    mask_regime=f'R{i%3+1}',n=ns[i],speed_mse=v*v,flow_lane_mse=q*q))
            for regime in range(3):
                rows.append(dict(panel=panel,family=panel.rsplit('_',1)[0],method=method,regime=f'R{regime+1}',
                    n=sum(ns[regime::3]),rmse_speed_kmh=v,rmse_flow_per_lane=q,S_state=score))
    return pd.DataFrame(rows),pd.DataFrame(days),checks,ref


def test_ten_summary_uses_family_macro_and_independently_rebuilds_daily_scores():
    detail,daily,checks,ref=synthetic_ten()
    for i in range(0,10,2):
        panels=list(ref.index[i:i+2]);spec=dict(panels=panels)
        validate_tables(detail.loc[detail.panel.isin(panels)],daily.loc[daily.panel.isin(panels)],checks,spec,ref)
    actual=aggregate_ten(detail,ref).set_index('method')
    for method in common.SUMMARY_METHODS:
        g=detail.loc[detail.method.eq(method)]
        expected=g.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean()
        assert actual.loc[method,'S_state']==pytest.approx(expected,abs=1e-14)
        assert abs(expected-np.average(g.S_state,weights=g.n))>1e-6
    assert actual.panel_count.eq(10).all() and actual.family_count.eq(5).all()


@pytest.mark.parametrize('change',['missing_direction','duplicate_regime','changed_mse','changed_fallback','missing_date'])
def test_incomplete_or_inconsistent_sources_are_rejected(change):
    detail,daily,checks,ref=synthetic_ten();panels=list(ref.index[:2]);spec=dict(panels=panels)
    detail=detail.loc[detail.panel.isin(panels)].copy();daily=daily.loc[daily.panel.isin(panels)].copy()
    if change=='missing_direction':detail=detail.loc[detail.panel.eq(panels[0])]
    if change=='duplicate_regime':detail=pd.concat([detail,detail.iloc[:1]],ignore_index=True)
    if change=='changed_mse':daily.loc[daily.index[0],'speed_mse']+=1.
    if change=='changed_fallback':checks[panels[0]+'/T1']['isolated_rows']+=1
    if change=='missing_date':daily=daily.iloc[1:]
    with pytest.raises(ValueError):validate_tables(detail,daily,checks,spec,ref)


def test_partial_and_smoke_queues_cannot_generate_formal_ten_results(tmp_path):
    for cfg in [dict(smoke=True,batches=common.selected_batches(common.BATCH_NAMES)),dict(smoke=False,batches=common.selected_batches(['i210']))]:
        with pytest.raises(ValueError):summarize_queue(tmp_path,tmp_path,cfg)
    detail,_,_,ref=synthetic_ten()
    with pytest.raises(ValueError):aggregate_ten(detail.loc[detail.panel.ne(ref.index[0])],ref)


def test_ten_date_bootstrap_does_not_freeze_unique_joint_regime_signatures():
    _,daily,_,_=synthetic_ten()
    daily=daily.loc[daily.method.isin(['T1','A_dynamic'])].copy()
    dates=sorted(daily.date.unique());panels=sorted(daily.panel.unique());rng=np.random.default_rng(53)
    signatures={p:rng.permutation(np.arange(55)%3) for p in panels}
    assert len(set(zip(*(signatures[p] for p in panels))))==55
    for p in panels:
        for i,date in enumerate(dates):
            idx=daily.panel.eq(p)&daily.date.eq(date)
            daily.loc[idx,'mask_regime']=f'R{signatures[p][i]+1}'
            daily.loc[idx&daily.method.eq('A_dynamic'),'speed_mse']*=.8+.2*np.sin(i)
    row=paired_calendar_bootstrap(daily,'T1','A_dynamic',replicates=128)
    assert row['ci95_high']-row['ci95_low']>1e-5
    assert row==paired_calendar_bootstrap(daily,'T1','A_dynamic',replicates=128)
    with pytest.raises(ValueError):paired_calendar_bootstrap(daily.loc[daily.panel.ne(panels[0])],'T1','A_dynamic',128)


def queue_fixture(monkeypatch,tmp_path):
    from transformer_remaining import queue
    args=SimpleNamespace(action='run',project=tmp_path/'project',release_root=tmp_path/'release',cache_dir=tmp_path/'cache',
        output=tmp_path/'output',device='cuda',smoke=True,batches=common.BATCH_NAMES.copy(),resume=False)
    cfg=dict(experiment=common.QUEUE_EXPERIMENT,action='run',project=str(args.project),release_root=str(args.release_root),
        cache_dir=str(args.cache_dir),device='cuda',smoke=True,batches=common.selected_batches(args.batches),
        training=dict(common.CONFIG,epochs=2,steps_per_epoch=4,patience=2),augmentation=copy.deepcopy(common.AUGMENT),
        split=common.plan(),fixed_fusion=common.recipes('rebuild_t0'),implementation=[],old_sources=[],old_source_anchors=[],versions={})
    monkeypatch.setattr(queue,'build_config',lambda _:copy.deepcopy(cfg))
    monkeypatch.setattr(queue,'inspect_child',lambda *a,**kw:None)
    monkeypatch.setattr(queue,'summarize_smoke',lambda *a:None)
    return queue,args,cfg


def test_failure_stops_queue_and_resume_skips_completed_batch(monkeypatch,tmp_path):
    queue,args,_=queue_fixture(monkeypatch,tmp_path);calls=[];failed=False
    def launch(args,batch,folder,resume):
        nonlocal failed
        calls.append((batch['name'],resume));folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/'run_config.json',{})
        if batch['name']=='i10' and not failed:
            failed=True;raise RuntimeError('injected failure')
        save_json(folder/'COMPLETE.json',{})
    monkeypatch.setattr(queue,'spawn_child',launch)
    with pytest.raises(RuntimeError,match='injected'):queue.run_queue(args)
    assert calls==[('i210',False),('i10',False)]
    status=read(args.output/'status.json');assert status['stage']=='failed' and status['completed_batches']==['i210']
    args.resume=True
    assert queue.run_queue(args)==0
    assert calls==[('i210',False),('i10',False),('i10',True),('i405',False)]
    assert (args.output/'SMOKE_COMPLETE.json').exists() and not (args.output/'COMPLETE.json').exists()


def test_interrupt_stops_current_queue_and_preserves_resume_output(monkeypatch,tmp_path):
    queue,args,_=queue_fixture(monkeypatch,tmp_path);calls=[]
    def launch(args,batch,folder,resume):
        calls.append(batch['name']);raise KeyboardInterrupt()
    monkeypatch.setattr(queue,'spawn_child',launch)
    assert queue.run_queue(args)==130 and calls==['i210']
    status=read(args.output/'status.json')
    assert status['stage']=='interrupted' and status['resume_output']==str(args.output)


def test_changed_queue_cannot_reuse_existing_output(monkeypatch,tmp_path):
    queue,args,cfg=queue_fixture(monkeypatch,tmp_path)
    args.output.mkdir();save_json(args.output/'queue_config.json',dict(cfg,device='cpu'));args.resume=True
    with pytest.raises(ValueError,match='config changed'):queue.run_queue(args)


def test_completed_source_directory_is_rejected_before_any_write(monkeypatch,tmp_path):
    queue,args,cfg=queue_fixture(monkeypatch,tmp_path)
    protected=args.output;protected.mkdir();(protected/'important.txt').write_text('preserve')
    cfg['old_sources']=[dict(path=str(protected))]
    with pytest.raises(ValueError,match='overlaps'):queue.run_queue(args)
    assert not (protected/'status.json').exists() and not (protected/'queue_config.json').exists()
    assert (protected/'important.txt').read_text()=='preserve'


def test_child_result_directory_cannot_be_mistaken_for_queue_root(monkeypatch,tmp_path):
    queue,args,_=queue_fixture(monkeypatch,tmp_path)
    args.output.mkdir();save_json(args.output/'run_config.json',dict(protected=True))
    with pytest.raises(ValueError,match='top-level queue'):queue.run_queue(args)
    assert read(args.output/'run_config.json')==dict(protected=True)
    assert not (args.output/'queue_config.json').exists()


def test_ctrl_c_waits_for_child_and_terminates_only_if_needed(monkeypatch,tmp_path):
    from transformer_remaining import queue
    args=SimpleNamespace(action='run',project=tmp_path,release_root=tmp_path/'release',cache_dir=tmp_path/'cache',device='cuda',smoke=True)
    class Process:
        calls=0;terminated=False
        def wait(self,timeout=None):
            self.calls+=1
            if self.calls==1:raise KeyboardInterrupt()
            if self.calls==2:raise queue.subprocess.TimeoutExpired('child',10)
            return 0
        def terminate(self):self.terminated=True
    child=Process();monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**kw:child)
    with pytest.raises(KeyboardInterrupt):queue.spawn_child(args,common.batch_for('i210'),tmp_path/'output',False)
    assert child.terminated and child.calls==3


def test_formal_log_writer_skips_smoke_and_deduplicates(tmp_path):
    from transformer_remaining.logging import append_batch_log,append_summary_log
    project=tmp_path/'project';out=project/'run';out.mkdir(parents=True)
    assert append_batch_log(out,project,dict(smoke=True),None)['status']=='skipped_smoke'
    assert not (project/'实验日志.txt').exists()
    table=pd.DataFrame([dict(method='T1',S_state=.9,delta_vs_T1=0.,delta_vs_current_fusion=0.,rmse_speed_kmh=2.,
        rmse_flow_per_lane=30.,delta_vs_T0=0.,delta_vs_historical=0.)])
    cfg=dict(smoke=False,panels=common.batch_for('i210')['panels'])
    assert append_batch_log(out,project,cfg,table)['status']=='appended'
    before=(project/'实验日志.txt').read_bytes()
    assert append_batch_log(out,project,cfg,table)['status']=='already_logged' and (project/'实验日志.txt').read_bytes()==before
    assert append_summary_log(out,project,table)['status']=='appended'
    before=(project/'实验日志.txt').read_bytes()
    assert append_summary_log(out,project,table)['status']=='already_logged' and (project/'实验日志.txt').read_bytes()==before
