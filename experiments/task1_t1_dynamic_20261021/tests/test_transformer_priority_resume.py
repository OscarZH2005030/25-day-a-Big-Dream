from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

from task1.transformer_priority import train,runner
from task1.transformer_priority.common import FROZEN,METHODS,PILOT,KEY,TARGETS,save_json,read,snapshot,sha256
from task1.transformer_priority.data import CachedData,graph_tensors


class TinyData:
    def __init__(self):
        self.n=3;self.panel='P';self.family='F';self.roles=dict(training=['d'],internal=['v'])
        adjacency=np.ones((3,3),dtype=bool)
        self.graph=dict(graph=adjacency,direction=np.zeros((3,3),dtype=int),order=np.arange(3),isolated=np.zeros(3,dtype=bool))
        self.records={d:dict(key=d) for d in ['d','v']};self.provenance=dict(normalization_key='fixed')
    def array(self,date,name):
        assert name=='target';return np.ones((288,3),dtype=bool)
    def batch(self,rng,config,balanced):
        b,t,n=config['batch_size'],config['window'],self.n
        result=dict(features=torch.tensor(rng.normal(size=(b,t,n,102)),dtype=torch.float32),
                    base=torch.ones(b,t,n,2),reliability=torch.zeros(b,t,n,8),time_valid=torch.ones(b,t,dtype=torch.bool),
                    label=torch.tensor(rng.uniform(.4,.9,size=(b,t,n,2)),dtype=torch.float32),target=torch.ones(b,t,n,dtype=torch.bool))
        return result,graph_tensors(self.graph,np.arange(n)),dict(nodes=list(range(n)),samples=[('d',0)]*b)


def test_interrupted_epoch_resume_matches_uninterrupted_training(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    config=dict(FROZEN,width=8,heads=4,layers=1,window=4,dropout=.1,epochs=2,steps_per_epoch=2,patience=2,eval_batch=2)
    monkeypatch.setattr(train,'internal_score',lambda *args:.5)
    full=TinyData();resumed=TinyData()
    a,ra=train.fit(full,config,METHODS[1],torch.device('cpu'),tmp_path/'full',dict(test=True),lambda **kw:None)
    def interrupted(**kw):
        if kw.get('stage')=='training' and kw.get('epoch')==2:
            raise RuntimeError('synthetic process interruption')
    with pytest.raises(RuntimeError,match='interruption'):
        train.fit(resumed,config,METHODS[1],torch.device('cpu'),tmp_path/'resume',dict(test=True),interrupted)
    b,rb=train.fit(resumed,config,METHODS[1],torch.device('cpu'),tmp_path/'resume',dict(test=True),lambda **kw:None)
    ca,_=train.checkpoint_read(Path(ra['folder']),'last.json',ra['key'])
    cb,_=train.checkpoint_read(Path(rb['folder']),'last.json',rb['key'])
    for name,value in ca['state_dict'].items():
        torch.testing.assert_close(value,cb['state_dict'][name],rtol=0,atol=0)
    assert ca['numpy_rng']==cb['numpy_rng']
    assert ca['coverage']['sampling_sha256']==cb['coverage']['sampling_sha256']
    assert [r['loss'] for r in ca['history']]==[r['loss'] for r in cb['history']]
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
    resumed.batch=lambda *args:(_ for _ in ()).throw(AssertionError('warm cache re-trained'))
    _,warm=train.fit(resumed,config,METHODS[1],torch.device('cpu'),tmp_path/'resume',dict(test=True),lambda **kw:None)
    assert warm['hit']


def test_real_block_padding_preserves_global_time_and_never_supervises_clipped_slots(tmp_path):
    n=3;raw=np.broadcast_to(np.arange(288)[:,None,None],(288,n,51)).astype(np.float32).copy()
    target=np.zeros((288,n),dtype=bool);target[0,0]=True;target[2,2]=True
    for name,value in dict(features=raw,base=np.ones((288,n,2),dtype=np.float32),
                           reliability=np.ones((288,n,8),dtype=np.float32),visible=np.zeros((288,n,2),dtype=bool),
                           target=target,ii=np.array([0,2]),jj=np.array([0,2])).items():
        np.save(tmp_path/(name+'.npy'),value)
    exp=SimpleNamespace(n=n,args=SimpleNamespace(panel='P'))
    data=CachedData(exp,None,dict(d=dict(folder=str(tmp_path))),dict(training=['d']),dict(),np.zeros(51),np.ones(51),dict(feature_names=list(range(51))))
    data.labels=lambda date,purpose:np.array([[25.,600.],[50.,1200.]])
    b=data.block('d',-2,np.array([0,2]),6,True)
    np.testing.assert_array_equal(b['time_valid'].numpy(),[False,False,True,True,True,True])
    assert not b['target'][:2].any() and b['target'].sum()==2
    assert not b['features'][:2].any() and not b['base'][:2].any()
    np.testing.assert_allclose(b['label'][2,0].numpy(),[1.,1.])
    np.testing.assert_allclose(b['label'][4,1].numpy(),[2.,2.])
    data.array.cache_clear();data.keys.cache_clear()


def formal_fixture(root,monkeypatch):
    monkeypatch.setattr(runner,'PROJECT',root)
    monkeypatch.setattr(runner,'verify_snapshot',lambda items:None)
    expected=pd.read_csv(runner.HERE/'reference_routing_counts.csv').set_index('panel')
    rows=[];preserved={};protected=[]
    for panel in PILOT:
        for method in METHODS:
            n=int(expected.loc[panel,'n']);counts=[n//3,n//3,n-2*(n//3)]
            for regime,count in zip(['R1','R2','R3'],counts):
                rows.append(dict(panel=panel,family='D7_I405',method=method,regime=regime,n=count,
                                 rmse_speed_kmh=2.,rmse_flow_per_lane=30.,S_state=.54*(1-2/25)+.46*(1-30/600)))
            preserved[panel+'/'+method]=dict(passed=True,rows=n,isolated_rows=int(expected.loc[panel,'fallback_n']),
                isolated_values_equal_LightGBM=True,nonisolated_values_equal_network=True)
            p=root/panel/(method+'_predictions.parquet');p.parent.mkdir(exist_ok=True);p.write_bytes(b'fixture placeholder')
            protected.append(p)
    detail=pd.DataFrame(rows);detail.to_csv(root/'metrics_by_panel_regime.csv',index=False)
    table=runner.summarize(detail);table['delta_vs_T0']=0.;table.to_csv(root/'comparison_methods.csv',index=False)
    pd.DataFrame([dict(reference=METHODS[0],candidate=m,delta_S_state=0.) for m in METHODS[1:]]).to_csv(root/'incremental_comparison.csv',index=False)
    save_json(root/'preservation_checks.json',preserved)
    save_json(root/'integrity_check.json',dict(passed=True,all_four_methods_same_targets=True,
        all_isolated_values_equal_shared_LightGBM=True,balanced_methods_paired_sampling=True))
    save_json(root/'run_config.json',dict(training=dict(FROZEN,eval_batch=2)))
    for name in ['environment.json','source_manifest.json','cache_usage.json','new_cache_manifest.json']:
        save_json(root/name,{})
    pd.DataFrame([dict(passed=True)]).to_csv(root/'truth_precision.csv',index=False)
    protected+=[root/n for n in ['run_config.json','environment.json','comparison_methods.csv','metrics_by_panel_regime.csv',
                               'integrity_check.json','preservation_checks.json','cache_usage.json','source_manifest.json','new_cache_manifest.json','truth_precision.csv']]
    save_json(root/'artifact_manifest.json',dict(artifacts=snapshot(protected)))
    save_json(root/'COMPLETE.json',dict(smoke=False,experiment='transformer_priorities',methods=METHODS,panels=PILOT,
        all_regimes_present=True,integrity_passed=True,artifact_manifest_sha256=sha256(root/'artifact_manifest.json')))


def test_formal_log_is_deduplicated_and_missing_scope_is_rejected(tmp_path,monkeypatch):
    formal_fixture(tmp_path,monkeypatch)
    assert runner.log_results(tmp_path)['status']=='appended'
    assert runner.log_results(tmp_path)['status']=='already_logged'
    content=(tmp_path/'实验日志.txt').read_text(encoding='utf-8')
    assert content.count('实验编号：')==1
    detail=pd.read_csv(tmp_path/'metrics_by_panel_regime.csv');detail=detail.iloc[:-1]
    detail.to_csv(tmp_path/'metrics_by_panel_regime.csv',index=False)
    with pytest.raises(ValueError,match='coverage'):
        runner.log_results(tmp_path)


def test_formal_log_rejects_unprotected_result_files(tmp_path,monkeypatch):
    formal_fixture(tmp_path,monkeypatch)
    save_json(tmp_path/'artifact_manifest.json',dict(artifacts=[]))
    complete=read(tmp_path/'COMPLETE.json');complete['artifact_manifest_sha256']=sha256(tmp_path/'artifact_manifest.json');save_json(tmp_path/'COMPLETE.json',complete)
    with pytest.raises(ValueError,match='Missing protected'):
        runner.log_results(tmp_path)


def test_lightgbm_roundtrip_in_chinese_project_path(tmp_path):
    import lightgbm as lgb
    from task1.transformer_priority.lightgbm import write_booster,read_booster
    x=np.arange(40,dtype=float).reshape(20,2);y=np.arange(20,dtype=float)
    model=lgb.train(dict(objective='regression',verbosity=-1,num_threads=1,min_data_in_leaf=2),lgb.Dataset(x,label=y),num_boost_round=3)
    folder=tmp_path/'中文比赛';folder.mkdir();path=folder/'模型.txt'
    write_booster(model,path);loaded=read_booster(path)
    np.testing.assert_array_equal(model.predict(x,num_threads=1),loaded.predict(x,num_threads=1))
