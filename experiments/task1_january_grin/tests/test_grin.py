import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from task1.cache import FeatureCache
from task1.data import KEY, TARGETS, save_json
from task1.grin.data import (GraphWindows, day_grid, graph_adjacency, grid_cache,
                            node_table, select_samples, window_starts)
from task1.grin.log import append_grin_log
from task1.grin.model import GRIN, imputation_loss
from task1.grin.runner import blend_windows, validate_protocol


def test_grin_shapes_hidden_payload_invariance_future_context_and_gradient():
    torch.set_num_threads(1); torch.manual_seed(1)
    graph=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    model=GRIN(graph,hidden=4,dropout=0)
    x=torch.randn(2,6,3,2);mask=torch.ones_like(x,dtype=torch.bool);mask[:,2:4,1]=False
    u=torch.randn(2,6,3,4)
    first,aux=model(x,mask,u)
    assert first.shape==x.shape and len(aux)==4
    altered=x.clone();altered[~mask]=1e10
    second,_=model(altered,mask,u)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    future=x.clone();future[:,-1]+=100
    changed,_=model(future,mask,u)
    assert not torch.allclose(first[:,2],changed[:,2])
    loss=imputation_loss(first,aux,torch.zeros_like(first),~mask,torch.tensor([1.,1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    zero=imputation_loss(first,aux,torch.zeros_like(first),torch.zeros_like(mask),torch.ones(2))
    assert zero.item()==0


def test_overlap_weighting_covers_edges_and_preserves_constant_predictions():
    starts=window_starts(48,24)
    values=np.ones((len(starts),48,3,2))*7
    np.testing.assert_array_equal(blend_windows(values,starts),np.ones((288,3,2))*7)
    with pytest.raises(ValueError,match='uncovered'):
        blend_windows(values[:1],starts[:1])
    with pytest.raises(ValueError):
        window_starts(24,48)


def test_directed_graph_bridges_unsensed_mainline_but_not_connectors():
    nodes=pd.DataFrame(dict(station_id=['S1','S2'],link_id=['A','C'],lanes=[2.,3.]))
    net=pd.DataFrame(dict(length_km=[1.,2.,1.]),index=['A','B','C'])
    topology=pd.DataFrame(dict(mainline_link_id=['A','B','C'],
        incoming_link_ids=['','A','B'],outgoing_link_ids=['B;CONN','C','']))
    adj=graph_adjacency(nodes,net,topology)
    assert adj[1,0]>0 and adj[0,1]==0 and not np.diag(adj).any()


def release_fixture(tmp_path):
    root=tmp_path/'release';panel='P';dates=['2030-06-01','2030-06-02','2031-01-01']
    netpath=root/'corridors'/panel/'network';netpath.mkdir(parents=True)
    net=pd.DataFrame(dict(link_id=['A','B'],lanes=[2.,3.],length_km=[1.,1.],
        free_speed_kmh=[100.,100.],capacity_vph=[4000.,6000.]))
    net.to_csv(netpath/'links.csv',index=False)
    net.to_csv(netpath/'fd_parameters.csv',index=False)
    pd.DataFrame(dict(mainline_link_id=['A','B'],incoming_link_ids=['','A'],outgoing_link_ids=['B',''])).to_csv(netpath/'lwr_mainline_topology.csv',index=False)
    (root/'config').mkdir();(root/'config/competition_contract.json').write_text('{}')
    targets=[];frames=[]
    for day in dates:
        rows=[]
        for slot in range(288):
            for j,(station,link) in enumerate([('S1','A'),('S2','B')]):
                timestamp=f'{day}T{slot//12:02}:{(slot%12)*5:02}:00Z'
                rows.append(dict(panel=panel,station_id=station,link_id=link,timestamp=timestamp,
                    mask_regime='R1',pct_observed=100,speed_kmh=50.+slot/100+j,flow_vph=1000.+slot+j))
        full=pd.DataFrame(rows)
        hidden=np.arange(len(full))%7==0
        targets.append(full.loc[hidden,KEY])
        masked=full.copy();masked.loc[hidden,TARGETS]=np.nan;frames.append(masked)
        for kind,frame in [('mainline_states_masked',masked),('mainline_states',full)]:
            # January unmasked data deliberately absent: evaluation cache cannot use it.
            if day.startswith('2031') and kind=='mainline_states':
                continue
            path=root/'corridors'/panel/'train'/kind;path.mkdir(parents=True,exist_ok=True)
            frame.to_parquet(path/f'synthetic_mainline_{day.replace("-","_")}.parquet',index=False)
    targetfolder=root/'task1'/panel/'train';targetfolder.mkdir(parents=True)
    pd.concat(targets).to_csv(targetfolder/'sample_submission_state.csv',index=False)
    cfg=dict(panel=panel,fit_start=dates[0],fit_end=dates[1],eval_start=dates[2],eval_end=dates[2],
        max_fit_days=None,max_eval_days=None,sample_per_day=10,seed=20261016)
    return root,cfg,frames,targets


def test_grin_grid_cache_no_eval_truth_fit_only_scaler_and_reproducible_samples(tmp_path):
    root,cfg,frames,targets=release_fixture(tmp_path)
    cache=FeatureCache(tmp_path/'cache')
    folder,train,hit=grid_cache(cache,root,cfg,'training')
    assert not hit
    assert grid_cache(cache,root,cfg,'training')[2]
    evalfolder,evaluation,_=grid_cache(cache,root,cfg,'evaluation',parent=train)
    assert evaluation['metadata']['scaler']==train['metadata']['scaler']
    with np.load(evalfolder/'2031-01-01.npz',allow_pickle=False) as arrays:
        assert set(arrays.files)=={'x','mask'}
    visible=pd.concat(frames[:2]);expected_speed=visible.speed_kmh.mean()
    assert abs(train['metadata']['scaler']['mean'][0]-expected_speed)<1e-5
    rng=np.random.default_rng(cfg['seed'])
    expected=pd.concat([select_samples(t,rng,10) for t in targets[:2]],ignore_index=True)
    pd.testing.assert_frame_equal(pd.read_parquet(folder/'training_sample_keys.parquet'),expected)
    data=GraphWindows(folder,train,24,24)
    x,mask,u,y,supervise=data[0]
    assert x.shape==(24,2,2) and u.shape==(24,2,4)
    assert not torch.any(mask & supervise)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    with pytest.raises(ValueError,match='Evaluation'):
        GraphWindows(evalfolder,evaluation)[0]


def test_grin_fixed_protocol_rejects_february_and_wrong_panel():
    cfg=dict(panel='D12_I5_N',fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',
        eval_end='2031-01-31',seed=20261016,window=48,stride=24,drop_observed=.05,aux_weight=.1,
        max_fit_days=None,max_eval_days=None,sample_per_day=1000)
    validate_protocol(cfg)
    validate_protocol({**cfg,'panel':'D7_I405_N'})
    for changes in [dict(eval_end='2031-02-28'),dict(panel='D7_I10_E'),dict(sample_per_day=500)]:
        with pytest.raises(ValueError):validate_protocol({**cfg,**changes})


def completed_log_run(tmp_path,smoke=False):
    run=tmp_path/'grin_example';run.mkdir()
    save_json(run/'COMPLETE.json',dict(experiment='grin_dev'))
    cfg=dict(panel='D12_I5_N',fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',eval_end='2031-01-31',
        max_fit_days=7 if smoke else None,max_eval_days=3 if smoke else None,no_log=False,sample_per_day=1000,
        seed=20261016,window=48,stride=24,hidden=32,batch_size=4,epochs=20,steps_per_epoch=100)
    save_json(run/'run_config.json',cfg)
    save_json(run/'selection.json',dict(training_rows=214000,best_epoch=3))
    pd.DataFrame([dict(method=m,S_state=s,rmse_speed_kmh=2.,rmse_flow_per_lane=36.,
        delta_S_state_vs_frozen_baseline=s-.92,all_regimes_present=True) for m,s in [('GRIN',.93),('Fusion_gap60mean',.92)]]).to_csv(run/'comparison_vs_baseline.csv',index=False)
    return run


def test_grin_log_append_dedup_and_smoke_exclusion(tmp_path):
    run=completed_log_run(tmp_path);log=tmp_path/'log.txt';log.write_text('已有\n',encoding='utf-8-sig')
    assert append_grin_log(run,log)['status']=='appended'
    text=log.read_text(encoding='utf-8-sig');assert text.startswith('已有\n') and 'GRIN：S_state=0.930000' in text
    assert append_grin_log(run,log)['status']=='already_logged'
    assert log.read_text(encoding='utf-8-sig')==text
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'));cfg['max_fit_days']=7
    save_json(run/'run_config.json',cfg)
    assert append_grin_log(run,tmp_path/'smoke_log.txt')['status']=='skipped_smoke'
    assert not (tmp_path/'smoke_log.txt').exists()
