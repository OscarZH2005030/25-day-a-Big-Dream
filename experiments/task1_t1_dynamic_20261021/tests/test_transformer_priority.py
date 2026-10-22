from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import torch

from task1.transformer_priority.common import KEY,TARGETS,FROZEN,METHODS,SCALE,upstream,frozen_plan,metric_rows,route,aligned
from task1.transformer_priority.data import sampling_spec,graph_tensors,reliability_from,CachedData
from task1.transformer_priority.model import model_for,objective
from task1.transformer_priority.train import Coverage,checkpoint_read
from task1.transformer_priority.runner import bootstrap,log_results
from task1.wave_neural.cache import NeuralCache
from task1.cache import CacheIntegrityError
from task1.data import save_json,sha256


def cfg():
    return dict(FROZEN,width=8,heads=4,layers=2,window=6,dropout=0.,epochs=2,steps_per_epoch=2,eval_batch=2)


def graph(n=7):
    adjacency=np.eye(n,dtype=bool)
    for i in range(n-1):
        adjacency[i,i+1]=adjacency[i+1,i]=True
    return dict(graph=adjacency,direction=np.sign(np.arange(n)[None,:]-np.arange(n)[:,None]).astype(int),
                order=np.arange(n),isolated=adjacency.sum(1)==1)


def batch(c=None,n=7):
    c=c or cfg();torch.manual_seed(71);t=c['window']
    return dict(features=torch.randn(2,t,n,102),base=torch.rand(2,t,n,2)+.2,reliability=torch.rand(2,t,n,8),
                time_valid=torch.ones(2,t,dtype=torch.bool),target=torch.ones(2,t,n,dtype=torch.bool),label=torch.rand(2,t,n,2))


def keys(n=4):
    return pd.DataFrame(dict(panel=['P']*n,timestamp=[f'2031-01-01T00:{5*i:02d}:00Z' for i in range(n)],
                             station_id=[str(i) for i in range(n)],link_id=[str(i) for i in range(n)],mask_regime=['R1']*n))


def test_reference_hashes_split_and_features():
    p=frozen_plan()
    assert [len(p[k]) for k in ['history_dates','training_dates','internal_validation_dates','holdout_dates']]==[28,168,22,55]
    assert upstream().worker.StateTransformer is not None


def test_original_model_forward_and_gradient_parity():
    c=cfg();new,_=model_for(c,METHODS[0])
    torch.manual_seed(c['seed'])
    old=upstream().worker.StateTransformer(102,c['width'],c['heads'],c['layers'],c['window'],c['dropout'])
    b=batch(c);g=graph_tensors(graph(),np.arange(7))
    with torch.no_grad():
        old.head[-1].weight.fill_(.03);new.head[-1].weight.fill_(.03)
    a=old(b,g);z=new(b,g)
    torch.testing.assert_close(a,z,rtol=1e-6,atol=2e-6)
    a.sum().backward();z.sum().backward()
    for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
        assert name==other
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('method',METHODS)
def test_shared_initialization_zero_readout_and_finite_gradient(method):
    c=cfg();model,digest=model_for(c,method);reference,expected=model_for(c,METHODS[0]);b=batch(c)
    assert digest==expected
    for name,value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    prediction=model(b,graph_tensors(graph(),np.arange(7)))
    desired=b['base'].clone();desired[...,0].clamp_min_(.04);desired[...,1].clamp_min_(0)
    torch.testing.assert_close(prediction,desired,rtol=0,atol=0)
    loss=objective(prediction,b);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('method',METHODS)
def test_padded_tokens_and_labels_cannot_affect_real_predictions(method):
    c=cfg();model,_=model_for(c,method);model.eval();b=batch(c);g=graph_tensors(graph(),np.arange(7))
    with torch.no_grad():
        model.head[-1].weight.fill_(.1)
    b['time_valid'][:,:2]=False
    a=model(b,g)
    changed={k:v.clone() for k,v in b.items()}
    changed['features'][:,:2]=10000;changed['reliability'][:,:2]=-10000
    changed['label'].fill_(999);changed['target'].fill_(False)
    z=model(changed,g,diagnostics=True)
    torch.testing.assert_close(a[:,2:],z[:,2:],rtol=1e-6,atol=2e-6)
    assert all(block.diagnostic['invalid_attention_mass']==0 for block in model.blocks)


def test_delay_and_reliability_learn_with_nonzero_head():
    c=cfg();model,_=model_for(c,METHODS[3]);b=batch(c)
    with torch.no_grad():
        model.head[-1].weight.normal_(std=.1)
    objective(model(b,graph_tensors(graph(),np.arange(7))),b).backward()
    assert any(block.lag_bias.grad.abs().sum()>0 for block in model.blocks)
    assert any(block.reliability[-1].weight.grad.abs().sum()>0 for block in model.blocks)


def test_balanced_candidate_coverage_exact_and_original_biased():
    n,w,patch=13,6,5
    temporal=np.zeros(288,dtype=int)
    for start in range(-(w-1),288):
        ticks=np.arange(start,start+w);temporal[ticks[(ticks>=0)&(ticks<288)]]+=1
    np.testing.assert_array_equal(temporal,np.full(288,w))
    spatial=np.zeros(n,dtype=int)
    for start in range(n):
        spatial[(start+np.arange(patch))%n]+=1
    np.testing.assert_array_equal(spatial,np.full(n,patch))
    old=np.convolve(np.ones(288-w+1),np.ones(w))
    assert old.max()/old[0]==w


def test_cyclic_node_block_does_not_invent_a_physical_edge():
    g=graph(7);nodes=np.array([5,6,0,1]);local=graph_tensors(g,nodes)
    for i,source in enumerate(nodes):
        for j in local['neighbors'][i][local['neighbor_valid'][i]].numpy():
            assert g['graph'][source,nodes[j]]
    assert 2 not in local['neighbors'][1][local['neighbor_valid'][1]].tolist()


def test_sampler_pairing_and_resumable_rng():
    a,b=np.random.default_rng(3),np.random.default_rng(3)
    for _ in range(20):
        x=sampling_spec(a,17,np.arange(17),['a','b'],6,8,2,True)
        y=sampling_spec(b,17,np.arange(17),['a','b'],6,8,2,True)
        np.testing.assert_array_equal(x[0],y[0]);assert x[1]==y[1]
    state=a.bit_generator.state;c=np.random.default_rng();c.bit_generator.state=state
    assert sampling_spec(a,17,np.arange(17),['a','b'],6,8,2,True)[1]==sampling_spec(c,17,np.arange(17),['a','b'],6,8,2,True)[1]


def test_reliability_uses_only_released_visibility_and_age():
    names=['speed_kmh_prev_gap_min','speed_kmh_next_gap_min','flow_vph_prev_gap_min','flow_vph_next_gap_min']
    raw=np.array([[[np.nan,15,0,0],[5,10,20,np.nan]]],dtype=float)
    visible=np.array([[[False,True],[False,False]]])
    result=reliability_from(raw,names,visible)
    np.testing.assert_allclose(result[0,0],[0,24,.25,0,1,0,0,1])
    np.testing.assert_allclose(result[0,1],[0,5/60,10/60,1,0,20/60,24,0])


def test_holdout_label_guard_precedes_cache_or_file_access():
    data=CachedData.__new__(CachedData)
    data.roles=dict(training=['a'],internal=['b'],holdout=['c']);data.allow_holdout=False
    with pytest.raises(ValueError,match='Label access'):
        data.labels('c','holdout')
    with pytest.raises(ValueError,match='Label access'):
        data.labels('c','training')


def test_speed_flow_metric_units_and_exact_isolation_route():
    k=keys();a=k.assign(speed_kmh=[10.,20,30,40],flow_vph=[100.,200,300,400])
    b=k.assign(speed_kmh=[11.,22,33,44],flow_vph=[110.,220,330,440])
    mask=np.array([True,False,True,False]);out=route(k,a,b,mask)
    np.testing.assert_array_equal(out.loc[mask,TARGETS],b.loc[mask,TARGETS])
    np.testing.assert_array_equal(out.loc[~mask,TARGETS],a.loc[~mask,TARGETS])
    truth=np.array([[20.,100.],[30.,150.],[40.,200.]])
    pred=truth+np.array([2.,12.]);rows=metric_rows(truth,pred,np.array(['R1','R2','R3']),'m','p','f')
    assert all(np.isclose(row['S_state'],.54*(1-2/25)+.46*(1-12/600)) for row in rows)
    with pytest.raises(ValueError,match='coverage'):
        aligned(k,a.iloc[:-1])


def test_cache_damage_rejected_and_builder_not_called_on_hit(tmp_path):
    cache=NeuralCache(tmp_path);identity=dict(kind='test',panel='P');calls=[]
    def build(out):
        calls.append(1);(out/'data.txt').write_text('original');return dict(passed=True)
    folder,manifest,hit=cache.get_or_build(identity,build)
    assert not hit
    assert cache.get_or_build(identity,build)[2] and len(calls)==1
    (folder/'data.txt').write_text('changed')
    with pytest.raises(CacheIntegrityError):
        cache.get_or_build(identity,build)


def test_checkpoint_tampering_rejected_before_deserialization(tmp_path):
    path=tmp_path/'epoch.pt';path.write_bytes(b'not pickle')
    save_json(tmp_path/'best.json',dict(name='epoch.pt',sha256='0'*64,identity='k'))
    with pytest.raises(ValueError,match='Checkpoint'):
        checkpoint_read(tmp_path,'best.json','k')


def test_coverage_roundtrip_counts_only_real_template_targets():
    targets=np.zeros((288,4),dtype=bool);targets[0,0]=True;targets[2,3]=True
    data=SimpleNamespace(n=4,graph=dict(order=np.arange(4)),roles=dict(training=['d']),array=lambda date,name:targets)
    coverage=Coverage(data)
    b=dict(target=torch.tensor([[[False,False],[True,False],[False,False],[False,True]]]))
    spec=dict(nodes=[0,3],samples=[('d',-1)])
    coverage.update(b,spec)
    before=coverage.summary();after=Coverage(data,coverage.state()).summary()
    assert before==after and before['unique_targets_seen']==2 and before['target_visits']==2


def test_bootstrap_identical_models_and_constant_improvement():
    rows=[]
    for day in range(6):
        for regime in ['R1','R2','R3']:
            for method,value in [('a',4.),('b',4.),('c',1.)]:
                rows.append(dict(panel='p',date=str(day),mask_regime=regime,method=method,n=5,speed_mse=value,flow_lane_mse=100.))
    daily=pd.DataFrame(rows)
    assert bootstrap(daily,'a','b',200)['ci95_low']==0
    assert bootstrap(daily,'a','c',200)['ci95_low']>0


def test_smoke_logger_never_writes_formal_log(tmp_path):
    save_json(tmp_path/'COMPLETE.json',dict(smoke=True))
    save_json(tmp_path/'run_config.json',{})
    assert log_results(tmp_path)['status']=='skipped_smoke'
