from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import torch
from task1.transformer_priority.common import upstream
from task1.transformer_priority.data import CachedData,reliability_from,sampling_spec
from transformer_training.augment import features_from_observations,additional_mask,hide_observations,link_groups
from transformer_training.sampling import Sampler,choose_supervision,Observations
from transformer_training.common import AUGMENT,CONFIG


def fixture_data(monkeypatch):
    rng=np.random.default_rng(17);n=6
    exp=object.__new__(upstream().baseline.Experiment)
    exp.n=n;exp.stations=[f'S{i}' for i in range(n)];exp.links=['L0','L0','L1','L2','L3','L4']
    exp.si={s:i for i,s in enumerate(exp.stations)}
    exp.static=pd.DataFrame(dict(lanes=[2.]*n,length_km=[1.]*n,free_speed_kmh=[100.]*n,capacity_vph=[4000.]*n))
    exp.lanes=exp.static.lanes.to_numpy()
    exp.neighbors={'up':[[] if i==0 else [i-1] for i in range(n)],'down':[[] if i==n-1 else [i+1] for i in range(n)]}
    exp.history=rng.uniform(20,100,size=(2,n,7,288));exp.history[1]*=3
    dates=['2030-07-01','2030-07-02','2030-07-03']
    exp.masked_files={d:Path(d) for d in dates}
    obs=np.empty((n,288,5));obs[:,:,0]=rng.uniform(20,100,(n,288));obs[:,:,1]=rng.uniform(100,2000,(n,288))
    obs[:,:,2]=rng.uniform(0,1,(n,288));obs[:,:,3]=100;obs[:,:,4]=0
    target=np.zeros((n,288),bool);target[:,::4]=True
    full=obs.copy();obs[target,:3]=np.nan
    rows=[]
    for i in range(n):
        for tick in range(288):
            rows.append(dict(station_id=exp.stations[i],timestamp=pd.Timestamp('2030-07-01',tz='UTC')+pd.Timedelta(minutes=tick*5),
                speed_kmh=obs[i,tick,0],flow_vph=obs[i,tick,1],occupancy=obs[i,tick,2],pct_observed=obs[i,tick,3],is_imputed=obs[i,tick,4]))
    frame=pd.DataFrame(rows)
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**kw: frame.copy())
    features,bases,_,_=exp.feature_grid(dates[0])
    names=list(features)
    raw=np.swapaxes(np.stack(list(features.values()),axis=-1),0,1).astype(np.float32)
    base=np.swapaxes(np.stack([bases[0]/25,bases[1]/600],axis=-1),0,1).astype(np.float32)
    visible=np.swapaxes(np.isfinite(obs[:,:,:2]),0,1)
    rel=reliability_from(raw,names,visible)
    labels=np.swapaxes(full[:,:,:2],0,1).copy();labels[:,:,1]/=exp.lanes[None,:];labels=(labels/np.array([25.,600.])).astype(np.float32)
    data=object.__new__(CachedData)
    data.exp=exp;data.n=n;data.panel='P';data.family='P';data.names=names;data.mean=np.zeros(51);data.scale=np.ones(51)
    data.roles=dict(training=dates,internal=['2030-07-04'],holdout=['2030-07-05'])
    data.graph=dict(order=np.arange(n),graph=np.ones((n,n),bool),direction=np.zeros((n,n),int),isolated=np.zeros(n,bool))
    arrays=dict(features=raw,base=base,visible=visible,reliability=rel,target=target.T)
    data.array=lambda date,name:arrays[name]
    data.label_grid=lambda date:labels
    data.labels=lambda date,role:np.zeros((1,2))
    data.label_records={d:dict(key=d) for d in dates+data.roles['internal']}
    data.records={d:dict(key=d) for d in dates+data.roles['internal']}
    data.provenance=dict(normalization_key='fixed')
    mapping=dict(zip(dates,['R1','R2','R3']))
    return data,obs,arrays,mapping


def test_original_51_features_and_subset_are_exact(monkeypatch):
    data,obs,arrays,_=fixture_data(monkeypatch)
    for nodes in [np.arange(data.n),np.array([1,3,5])]:
        rebuilt=features_from_observations(data.exp,'2030-07-01',obs,nodes)
        for name,value in zip(['features','base','visible','reliability'],rebuilt[:4]):
            np.testing.assert_array_equal(value,arrays[name][:,nodes])
        assert rebuilt[4]==data.names


@pytest.mark.parametrize('pattern',['point','temporal','spatial'])
def test_mask_patterns_group_stations_and_remove_all_channels(monkeypatch,pattern):
    data,obs,_,_=fixture_data(monkeypatch)
    units,adj=link_groups(data.exp)
    mask,eligible,stats=additional_mask(obs,units,adj,np.random.default_rng(4),AUGMENT,pattern)
    assert stats['hidden_units']==int(stats['eligible_units']*.1)>0
    assert np.array_equal(mask[0],mask[1])
    modified=hide_observations(obs,mask)
    assert np.isnan(modified[mask]).all()
    np.testing.assert_array_equal(obs[~mask],modified[~mask])
    poisoned=obs.copy();poisoned[mask]=1e12
    a=features_from_observations(data.exp,'2030-07-01',modified,np.arange(data.n))
    b=features_from_observations(data.exp,'2030-07-01',hide_observations(poisoned,mask),np.arange(data.n))
    for x,y in zip(a[:4],b[:4]): np.testing.assert_array_equal(x,y)


def test_no_fake_spatial_edges_when_every_link_is_isolated(monkeypatch):
    data,obs,_,_=fixture_data(monkeypatch);units,adj=link_groups(data.exp)
    mask,_,stats=additional_mask(obs,units,[set() for _ in adj],np.random.default_rng(7),AUGMENT,'spatial')
    assert mask.any() and stats['point_fill_units']==stats['hidden_units']


@pytest.mark.parametrize('count',[0,1,4,5,9,10,100])
def test_supervision_budget_and_original_label_retention(count):
    original=np.zeros((20,10),bool);original.flat[:count]=True
    extra=~original;labels=np.zeros((20,10,2),np.float32);values=np.ones_like(labels)
    target,label=choose_supervision(original,extra,labels,values,np.random.default_rng(2),.2)
    assert target.sum()==count
    assert (target&original).sum()>=.8*count
    assert (target&~original).sum()==int(count*.2)
    assert (label[target&~original]==1).all()


def test_dynamic_sampler_preserves_T1_rng_and_padding(monkeypatch):
    data,obs,_,mapping=fixture_data(monkeypatch)
    sampler=Sampler(data,'A_dynamic',mapping);sampler.observations.get=lambda d:obs
    cfg=dict(CONFIG,patch_nodes=4)
    rng=np.random.default_rng(2026);expected_rng=np.random.default_rng(2026)
    batch,graph,spec,infos=sampler.batch(rng,np.random.default_rng(18),cfg)
    nodes,samples=sampling_spec(expected_rng,data.n,data.graph['order'],data.roles['training'],24,4,2,True)
    assert spec==dict(nodes=nodes.tolist(),samples=samples)
    assert rng.bit_generator.state==expected_rng.bit_generator.state
    assert batch['target'].sum()==sum(info['original_targets'] for info in infos)
    part,info=sampler.dynamic_block('2030-07-01',-10,nodes,24,np.random.default_rng(3))
    assert not part['target'][:10].any() and not part['original_target'][:10].any()
    assert not part['time_valid'][:10].any()


def test_regime_sampler_balanced_and_training_dates_only(monkeypatch):
    data,_,_,mapping=fixture_data(monkeypatch)
    sampler=Sampler(data,'B_regime',mapping);rng=np.random.default_rng(20)
    counts={r:0 for r in ['R1','R2','R3']}
    for _ in range(600):
        batch,_,spec,infos=sampler.batch(rng,np.random.default_rng(0),dict(CONFIG,patch_nodes=4))
        regimes={i['regime'] for i in infos};assert len(regimes)==1
        counts[next(iter(regimes))]+=1
        assert all(d in data.roles['training'] for d,_ in spec['samples'])
        assert batch['target'].equal(batch['original_target'])
        assert sum(i['extra_targets'] for i in infos)==0
    assert all(150<c<250 for c in counts.values())


def test_augmentation_refuses_validation_dates(monkeypatch):
    data,_,_,_=fixture_data(monkeypatch)
    with pytest.raises(ValueError,match='training dates only'):
        Observations(data).ensure(data.roles['holdout'][0])


def test_original_objective_value_and_gradient_unchanged():
    from task1.transformer_priority.model import objective
    pred=torch.tensor([[[[1.,2.],[3.,4.]]]],requires_grad=True)
    batch=dict(target=torch.tensor([[[True,False]]]),label=torch.zeros_like(pred))
    value=objective(pred,batch);value.backward()
    assert torch.isclose(value,torch.tensor(.54*1+.46*2))
    assert torch.equal(pred.grad[0,0,1],torch.zeros(2))


@pytest.mark.parametrize('device_name',['cpu','cuda'])
def test_epoch_resume_restores_parameters_sampling_and_augmentation_rng(monkeypatch,tmp_path,device_name):
    import transformer_training.train as training
    if device_name=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    device=torch.device(device_name)
    data,obs,_,mapping=fixture_data(monkeypatch)
    monkeypatch.setattr(training,'internal_score',lambda model,*a:float(model.head[-1].weight.detach().sum()))
    cfg=dict(CONFIG,width=8,heads=2,layers=1,window=4,patch_nodes=4,epochs=2,steps_per_epoch=2,patience=8)
    def make_sampler():
        sampler=Sampler(data,'A_dynamic',mapping);sampler.observations.get=lambda d:obs;return sampler
    torch.set_num_threads(1)
    continuous,record=training.fit(data,make_sampler(),cfg,'A_dynamic',device,tmp_path/'continuous',{},lambda **kw:None)
    with pytest.raises(RuntimeError,match='Injected epoch interruption'):
        training.fit(data,make_sampler(),cfg,'A_dynamic',device,tmp_path/'resumed',{},lambda **kw:None,stop_after_epoch=1)
    resumed,resumed_record=training.fit(data,make_sampler(),cfg,'A_dynamic',device,tmp_path/'resumed',{},lambda **kw:None)
    for key,value in continuous.state_dict().items(): assert torch.equal(value,resumed.state_dict()[key]),key
    a,_=training.checkpoint_read(Path(record['folder']),'last.json',record['key'])
    b,_=training.checkpoint_read(Path(resumed_record['folder']),'last.json',resumed_record['key'])
    assert a['numpy_rng']==b['numpy_rng'] and a['augmentation_rng']==b['augmentation_rng']
    assert a['coverage']['sampling_sha256']==b['coverage']['sampling_sha256']
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    if device_name=='cuda':
        assert all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))


def test_logger_smoke_gate_formal_budget_and_idempotence(monkeypatch,tmp_path):
    from task1.data import save_json
    from transformer_training import runner
    from transformer_training.common import VARIANTS,METHODS
    project=tmp_path/'project';out=project/'runs/test';out.mkdir(parents=True)
    dates=[f'd{i}' for i in range(22)]
    cfg=dict(smoke=True,training=CONFIG,panels=runner.PILOT,variants=VARIANTS,augmentation=AUGMENT,
             split=dict(internal_validation_dates=dates))
    table=pd.DataFrame([dict(method=m,S_state=.9,delta_vs_T1=0.,delta_vs_current_fusion=0.,
                            rmse_speed_kmh=2.,rmse_flow_per_lane=30.) for m in METHODS])
    monkeypatch.setattr(runner,'verify_complete',lambda _: (cfg,table))
    assert runner.log_results(out,project)['status']=='skipped_smoke'
    assert not (project/'实验日志.txt').exists()
    cfg['smoke']=False
    save_json(out/'frozen_models.json',dict(selection_dates=dates))
    save_json(out/'integrity_check.json',dict(passed=True,all_four_models_frozen_before_holdout=True,
        control_scores_replayed=True,isolated_exact=True,fixed_formula_exact=True))
    refs=pd.read_csv(Path(__file__).resolve().parents[1]/'task1/transformer_priority/reference_routing_counts.csv')
    folder=project/'task1/transformer_priority';folder.mkdir(parents=True);refs.to_csv(folder/'reference_routing_counts.csv',index=False)
    rows=[];checks={};models={}
    for panel in runner.PILOT:
        models[panel]={v:dict(model=dict(selection=dict(config=CONFIG,optimizer_updates=61440))) for v in VARIANTS}
        ref=refs.set_index('panel').loc[panel]
        for method in METHODS:
            checks[panel+'/'+method]=dict(isolated_rows=int(ref.fallback_n),isolated_exact=True)
            for i in range(3):
                rows.append(dict(panel=panel,method=method,regime=f'R{i+1}',n=int(ref.n)//3 if i<2 else int(ref.n)-2*(int(ref.n)//3)))
    pd.DataFrame(rows).to_csv(out/'holdout_metrics_by_panel_regime.csv',index=False)
    save_json(out/'holdout_preservation_checks.json',checks);save_json(out/'model_cache_usage.json',models)
    assert runner.log_results(out,project)['status']=='appended'
    content=(project/'实验日志.txt').read_bytes()
    assert runner.log_results(out,project)['status']=='already_logged'
    assert (project/'实验日志.txt').read_bytes()==content


def test_relative_model_manifest_detects_corruption(tmp_path):
    from task1.data import save_json
    from transformer_training.train import relative_snapshot,verify_model
    (tmp_path/'weights.pt').write_bytes(b'valid')
    save_json(tmp_path/'COMPLETE.json',dict(identity='test',artifacts=relative_snapshot(tmp_path,['weights.pt'])))
    verify_model(tmp_path)
    (tmp_path/'weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'): verify_model(tmp_path)
