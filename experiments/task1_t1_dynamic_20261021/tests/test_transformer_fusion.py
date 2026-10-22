from pathlib import Path
from types import SimpleNamespace
import sys

# Also permits isolated validation before deployment into the competition project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from transformer_fusion.core import mix, select, choose, details, summarize
from transformer_fusion.runner import paired_bootstrap, log_results
from transformer_fusion.source import verify
from task1.data import KEY, TARGETS, save_json
from task1.transformer_priority.common import aligned
from task1.wave_neural.cache import NeuralCache


def test_exact_endpoints_and_isolated_fallback():
    a = np.array([[11.,101.],[22.,202.],[33.,303.]])
    b, tree = a+7, a+17
    iso = np.array([False,True,False])
    result = mix(a,b,tree,iso,0)
    assert np.array_equal(result[~iso],a[~iso])
    assert np.array_equal(result[iso],tree[iso])
    result = mix(a,b,tree,iso,1,0,.25)
    assert np.array_equal(result[~iso,0],b[~iso,0])
    np.testing.assert_allclose(result[~iso,1],.75*b[~iso,1]+.25*tree[~iso,1])
    assert np.array_equal(mix(a,b,tree,iso,.5,1,1),tree)


@pytest.mark.parametrize('bad', [np.nan,np.inf,-1.])
def test_invalid_prediction_rejected(bad):
    a = np.ones((3,2)); a[0,0] = bad
    with pytest.raises(ValueError):
        mix(a,np.ones((3,2)),np.ones((3,2)),np.zeros(3,bool),0)


def test_bad_masks_shapes_weights_rejected():
    a = np.ones((3,2))
    for iso, beta in [(np.zeros(2,bool),0),(np.zeros(3),0),(np.zeros(3,bool),1.01)]:
        with pytest.raises(ValueError):
            mix(a,a,a,iso,beta)
    with pytest.raises(ValueError):
        mix(a,a[:2],a,np.zeros(3,bool),0)


def keys():
    return pd.DataFrame([['P',f'2030-12-14T00:{i*5:02d}:00Z',str(i),str(i),'R1'] for i in range(3)],columns=KEY)


def test_alignment_shuffled_and_bad_coverage():
    k = keys(); pred = k.assign(speed_kmh=[1.,2.,3.],flow_vph=[10.,20.,30.])
    pd.testing.assert_frame_equal(aligned(k,pred.iloc[::-1]),pred)
    for bad in [pred.iloc[:2],pd.concat([pred,pred.iloc[:1]]),pred.assign(speed_kmh=np.inf),pred.assign(flow_vph=-1.)]:
        with pytest.raises(ValueError):
            aligned(k,bad)


def bundles():
    dates = ['2030-12-14','2030-12-15','2030-12-16','2030-12-17','2030-12-18','2030-12-19']
    result = {}
    for panel in ['D7_I405_N','D7_I405_S']:
        ctx = pd.DataFrame(dict(panel=panel,date=dates,mask_regime=['R1','R2','R3']*2,
             isolated=[False]*6,lanes=[2.]*6,free_speed_kmh=[100.]*6))
        truth = np.tile([50.,100.],(6,1))
        # Equal neural prediction cancels error exactly; flow is published TOTAL flow.
        result[panel] = dict(context=ctx,truth=truth,T0=np.tile([45.,180.],(6,1)),
                            T1=np.tile([55.,220.],(6,1)),LightGBM=np.tile([70.,280.],(6,1)))
    return result, dates


def test_internal_only_selection_and_no_tree_when_neural_exact():
    data, dates = bundles()
    selected, neural, grid, stability = select(data,dates)
    assert selected['neural'] == 'T01_equal'
    assert selected['recipe'] == dict(beta=.5,speed=0.,flow=0.)
    assert len(grid) == 16*2*3
    assert set(stability.period) == {'early','late'}
    with pytest.raises(ValueError, match='outside internal'):
        select(data,dates[:-1])


def test_macro_score_uses_per_lane_flow_and_regime_not_rows():
    data,_ = bundles()
    t = summarize(details(data,dict(beta=.5),'Equal'))
    assert t.S_state.iloc[0] == 1.0
    assert t.rmse_flow_per_lane.iloc[0] == 0.0
    one = data['D7_I405_N']
    for k in ['T0','T1','LightGBM']:
        one[k] = one[k][:1].repeat(6,axis=0)
    one['truth'] = np.tile([50.,100.],(6,1))
    result = details({'D7_I405_N':one},dict(beta=0),'T0')
    assert np.isclose(result.rmse_flow_per_lane.iloc[0],10.)


def test_tie_rule_deterministic():
    table=pd.DataFrame(dict(method=['B','A'],S_state=[.9,.9-5e-13]))
    assert choose(table,{'A':0,'B':1}) == 'A'


def test_corrupt_daily_cache_rejected(tmp_path):
    cache=NeuralCache(tmp_path)
    identity=dict(kind='test',panel='P',value=1)
    calls=[]
    def builder(out):
        calls.append(1); (out/'result.txt').write_text('valid'); return dict(rows=1)
    folder,_,hit=cache.get_or_build(identity,builder)
    assert not hit
    assert cache.get_or_build(identity,builder)[2] and len(calls)==1
    (folder/'result.txt').write_text('broken')
    with pytest.raises(RuntimeError):
        cache.get_or_build(identity,builder)


def test_log_refuses_incomplete_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        log_results(tmp_path,tmp_path)
    assert not (tmp_path/'实验日志.txt').exists()


def test_holdout_guard_runs_before_any_prediction_or_truth_access(tmp_path):
    from transformer_fusion.source import Source
    source=object.__new__(Source)
    source.data={'P':object()}; source.roles={'P':dict(holdout=['2031-01-05'])}
    with pytest.raises(ValueError,match='Freeze internal selection'):
        source.bundle('P','holdout',tmp_path/'missing.json')


def test_paired_regime_bootstrap_handles_one_date_per_regime():
    rows=[]
    for method in ['T0','Selected']:
        for panel in ['D7_I405_N','D7_I405_S']:
            for i,r in enumerate(['R1','R2','R3']):
                rows.append(dict(method=method,panel=panel,date=f'2030-12-{14+i}',mask_regime=r,
                    n=100,speed_mse=4. if method=='T0' else 1.,flow_lane_mse=100.))
    result=paired_bootstrap(pd.DataFrame(rows),['Selected'],100)
    assert result[0]['ci95_low'] > 0
    assert np.isfinite(result[0]['ci95_high'])


def test_source_hash_corruption_refused(tmp_path):
    from task1.cache import snapshot
    path=tmp_path/'source';path.write_text('ok')
    record=snapshot([path]);verify(record)
    path.write_text('tampered')
    with pytest.raises(ValueError,match='Protected file changed'):
        verify(record)


def test_bootstrap_allows_different_regimes_on_same_date():
    rows=[]
    for method in ['T0','Selected']:
        for panel,offset in [('D7_I405_N',0),('D7_I405_S',1)]:
            for i in range(6):
                rows.append(dict(method=method,panel=panel,date=f'2030-12-{14+i}',
                    mask_regime=['R1','R2','R3'][(i+offset)%3],n=100,
                    speed_mse=4. if method=='T0' else 1.,flow_lane_mse=100.))
    result=paired_bootstrap(pd.DataFrame(rows),['Selected'],100)
    assert result[0]['ci95_low'] > 0


def test_macro_score_is_not_dominated_by_row_count_or_family_size():
    rows=[]
    for panel, family, values in [('A_N','A',[.9,.6,.3]),('A_S','A',[.9,.9,.9]),('B_N','B',[.3,.3,.3])]:
        for i,value in enumerate(values):
            rows.append(dict(method='M',panel=panel,family=family,regime=f'R{i+1}',
                n=1 if i else 1000000,S_state=value,rmse_speed_kmh=1.,rmse_flow_per_lane=1.))
    table=summarize(pd.DataFrame(rows))
    # A: direction means .6/.9 -> .75; B: .3; family macro -> .525.
    assert np.isclose(table.S_state.iloc[0],.525)
    assert table.all_regimes_present.iloc[0]


def test_formal_logger_appends_once_and_smoke_never_logs(tmp_path,monkeypatch):
    import transformer_fusion.runner as runner
    project=tmp_path/'project';out=project/'runs'/'test'
    out.mkdir(parents=True)
    dates=[f'day{i}' for i in range(22)]
    cfg=dict(smoke=True,split=dict(internal_validation_dates=dates),source_run='frozen-source')
    table=pd.DataFrame([dict(method='T0',S_state=.9,delta_vs_T0=0.,
                            rmse_speed_kmh=2.,rmse_flow_per_lane=30.)])
    monkeypatch.setattr(runner,'verify_complete',lambda _: ({},cfg,table))
    assert runner.log_results(out,project)['status']=='skipped_smoke'
    assert not (project/'实验日志.txt').exists()
    cfg['smoke']=False
    save_json(out/'selection.json',dict(selection_dates=dates,recipe=dict(beta=1.,speed=.1,flow=.1)))
    save_json(out/'integrity_check.json',dict(passed=True,source_scores_replayed=True,
             all_isolated_exact=True,selection_frozen_before_holdout=True))
    refs=pd.read_csv(Path(__file__).resolve().parents[1]/'task1/transformer_priority/reference_routing_counts.csv')
    reference_path=project/'task1/transformer_priority'
    reference_path.mkdir(parents=True)
    refs.to_csv(reference_path/'reference_routing_counts.csv',index=False)
    rows=[];checks={}
    for panel in runner.PILOT:
        ref=refs.set_index('panel').loc[panel]
        for method in list(runner.CONTROLS)+['Selected']:
            checks[panel+'/'+method]=dict(isolated_exact=True,isolated_rows=int(ref.fallback_n))
            for i in range(3):
                rows.append(dict(panel=panel,method=method,regime=f'R{i+1}',
                                 n=int(ref.n)//3 if i<2 else int(ref.n)-2*(int(ref.n)//3)))
    pd.DataFrame(rows).to_csv(out/'metrics_by_panel_regime.csv',index=False)
    save_json(out/'preservation_checks.json',checks)
    assert runner.log_results(out,project)['status']=='appended'
    content=(project/'实验日志.txt').read_bytes()
    assert runner.log_results(out,project)['status']=='already_logged'
    assert (project/'实验日志.txt').read_bytes()==content
