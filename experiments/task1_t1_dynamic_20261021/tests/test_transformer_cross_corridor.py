from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import torch
from test_transformer_training import fixture_data
from task1.data import KEY, TARGETS, save_json
from task1.transformer_priority.common import read
from transformer_training.sampling import Sampler
from transformer_cross_corridor.common import CONFIG, PANELS, AUGMENT, EXPERIMENT, plan, methods, recipes, validate_config
from transformer_cross_corridor.runner import OriginalSampler, inputs, validate_formal_scope


@pytest.mark.parametrize('device_name',['cpu','cuda'])
@pytest.mark.parametrize('method',['T1','T0'])
def test_adapted_training_exactly_matches_original(monkeypatch,tmp_path,device_name,method):
    import transformer_training.train as adapted
    import task1.transformer_priority.train as original
    if device_name=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    data,_,_,mapping = fixture_data(monkeypatch)
    internal = lambda model,*args:float(model.head[-1].weight.detach().sum())
    monkeypatch.setattr(adapted,'internal_score',internal)
    monkeypatch.setattr(original,'internal_score',internal)
    cfg = dict(CONFIG,width=8,heads=2,layers=1,window=4,patch_nodes=4,epochs=2,steps_per_epoch=2)
    torch.set_num_threads(1)
    device = torch.device(device_name)
    sampler = Sampler(data,'T1',mapping) if method=='T1' else OriginalSampler(data,mapping)
    old_name = 'T1_balanced' if method=='T1' else 'T0_original'
    a,ra = adapted.fit(data,sampler,cfg,method,device,tmp_path/'adapted',{},lambda **kw:None)
    b,rb = original.fit(data,cfg,old_name,device,tmp_path/'original',{},lambda **kw:None)
    for key,value in a.state_dict().items():
        assert torch.equal(value,b.state_dict()[key]),key
    ca,_ = original.checkpoint_read(Path(ra['folder']),'last.json',ra['key'])
    cb,_ = original.checkpoint_read(Path(rb['folder']),'last.json',rb['key'])
    assert ca['numpy_rng']==cb['numpy_rng']
    assert ca['coverage']['sampling_sha256']==cb['coverage']['sampling_sha256']
    assert torch.equal(ca['torch_rng'],cb['torch_rng'])


@pytest.mark.parametrize('mode',['rebuild_t0','tree_only'])
@pytest.mark.parametrize('isolated',[False,True])
def test_new_scores_keep_exact_fallback_and_fixed_formula(monkeypatch,tmp_path,mode,isolated):
    from transformer_cross_corridor import runner
    from transformer_training.runner import write_scores
    n=6
    ctx=pd.DataFrame({k:['x']*n for k in KEY})
    ctx['timestamp']=[f'2031-01-05T00:{5*i:02d}:00Z' for i in range(n)]
    ctx['mask_regime']=['R1','R1','R2','R2','R3','R3']
    ctx['panel']='D12_I5_N'
    ctx['date']='2031-01-05'
    ctx['isolated']=np.full(n,isolated)
    ctx['lanes']=2.
    ctx['gap']='<=15min'
    ctx['flow_gap']='<=15min'
    ctx['free_speed_kmh']=100.
    class Data:
        roles=dict(holdout=['2031-01-05'])
        def truth(self,*args): return np.ones((n,2))
    monkeypatch.setattr(runner,'context',lambda *args:ctx)
    def frame(v):
        f=ctx[KEY].copy()
        f[TARGETS]=np.full((n,2),v)
        return f
    raw=dict(T1=frame(20.),A_dynamic=frame(30.))
    if mode=='rebuild_t0': raw['T0']=frame(10.)
    bundles=inputs(Data(),raw,frame(40.),mode,'holdout')
    write_scores({'D12_I5_N':bundles},tmp_path,'holdout')
    pred=pd.read_parquet(tmp_path/'holdout/D12_I5_N/A_dynamic_fixed_predictions.parquet')[TARGETS].to_numpy()
    expected=40. if isolated else 22. if mode=='rebuild_t0' else 31.
    np.testing.assert_array_equal(pred,np.full((n,2),expected))


def configuration(mode='rebuild_t0',smoke=False):
    return dict(experiment=EXPERIMENT,panels=PANELS,variants=['T1','A_dynamic'],augmentation=AUGMENT,
        split=plan(),methods=methods(mode),fixed_fusion=recipes(mode),fusion_mode=mode,smoke=smoke,
        training=dict(CONFIG,epochs=2,steps_per_epoch=4,patience=2) if smoke else dict(CONFIG))


def test_internal_replay_preserves_physical_predictions(monkeypatch):
    from transformer_cross_corridor import runner
    ctx=pd.DataFrame({k:['x']*3 for k in KEY})
    ctx['mask_regime']=['R1','R2','R3']
    ctx['timestamp']=['a','b','c']
    ctx['lanes']=2.
    frame=ctx[KEY].copy()
    frame['speed_kmh']=25.
    frame['flow_vph']=1200.
    unchanged=frame.copy(deep=True)
    class Data:
        roles=dict(internal=['date'])
        panel='D12_I5_N'
        family='D12_I5'
        def truth(self,*args): return np.tile([25.,600.],(3,1))
    monkeypatch.setattr(runner,'context',lambda *args:ctx)
    rows=runner.replay_internal(Data(),{'T1':frame},{'T1':{'model':{'selection':{'best_internal_S_state':1.}}}})
    assert rows[0]['replayed_S']==1.
    pd.testing.assert_frame_equal(frame,unchanged,check_exact=True)


@pytest.mark.parametrize('field,value',[('panels',['D7_I405_N','D7_I405_S']),('training',dict(CONFIG,steps_per_epoch=1025)),
    ('fixed_fusion',dict(beta=.5,speed=.25,flow=.1)),('augmentation',dict(AUGMENT,hidden_fraction=.2))])
def test_protocol_changes_are_rejected(field,value):
    cfg=configuration()
    validate_config(cfg)
    cfg[field]=value
    with pytest.raises(ValueError): validate_config(cfg)


def test_formal_logging_requires_full_dates_counts_and_budget(monkeypatch,tmp_path):
    from transformer_cross_corridor import runner
    cfg=configuration()
    cfg['project']=str(tmp_path)
    table=pd.DataFrame([dict(method=m,S_state=.9,delta_vs_T1=0.,delta_vs_current_fusion=0.,
         rmse_speed_kmh=2.,rmse_flow_per_lane=30.) for m in cfg['methods']])
    monkeypatch.setattr(runner,'verify_complete',lambda _: (cfg,table))
    cfg['smoke']=True
    assert runner.log_results(tmp_path,tmp_path)['status']=='skipped_smoke'
    assert not (tmp_path/'实验日志.txt').exists()
    cfg['smoke']=False
    # Exercise the real formal gate separately; a completed summary is insufficient.
    save_json(tmp_path/'integrity_check.json',dict(passed=True,all_models_frozen_before_holdout=True,
        isolated_exact=True,fixed_formula_exact=True,internal_checkpoint_replayed=True))
    reference=pd.read_csv(Path(__file__).resolve().parents[1]/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    folder=tmp_path/'task1/transformer_priority'
    folder.mkdir(parents=True)
    reference.to_csv(folder/'reference_routing_counts.csv')
    preflight={'panels':{}}
    records={}
    checks={}
    rows=[]
    for p in PANELS:
        preflight['panels'][p]=dict(dates={r:cfg['split'][k] for r,k in [('training','training_dates'),('internal','internal_validation_dates'),('holdout','holdout_dates')]})
        records[p]={m:dict(model=dict(selection=dict(config=CONFIG,optimizer_updates=61440,epochs_run=60))) for m in ['T1','A_dynamic','T0']}
        for method in cfg['methods']:
            checks[p+'/'+method]=dict(isolated_rows=int(reference.loc[p,'fallback_n']),isolated_exact=True)
            total=int(reference.loc[p,'n'])
            for i in range(3):
                rows.append(dict(panel=p,method=method,regime=f'R{i+1}',n=total//3 if i<2 else total-2*(total//3)))
    save_json(tmp_path/'preflight.json',preflight)
    save_json(tmp_path/'model_cache_usage.json',records)
    save_json(tmp_path/'holdout_preservation_checks.json',checks)
    save_json(tmp_path/'frozen_models.json',dict(selection_dates=cfg['split']['internal_validation_dates']))
    pd.DataFrame(rows).to_csv(tmp_path/'holdout_metrics_by_panel_regime.csv',index=False)
    validate_formal_scope(tmp_path,cfg)
    assert runner.log_results(tmp_path,tmp_path)['status']=='appended'
    original=(tmp_path/'实验日志.txt').read_bytes()
    assert runner.log_results(tmp_path,tmp_path)['status']=='already_logged'
    assert (tmp_path/'实验日志.txt').read_bytes()==original
    preflight['panels'][PANELS[0]]['dates']['holdout']=cfg['split']['holdout_dates'][:-1]
    save_json(tmp_path/'preflight.json',preflight)
    with pytest.raises(ValueError,match='dates differ'): validate_formal_scope(tmp_path,cfg)
