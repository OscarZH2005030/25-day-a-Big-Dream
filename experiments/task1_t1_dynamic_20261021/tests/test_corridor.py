import json

import numpy as np
import pandas as pd
import pytest

from task1 import corridor_validation as cv
from task1.cache import CacheIntegrityError, FeatureCache
from task1.data import KEY, TARGETS, save_json
from task1.experiment_log import append_corridor_log
from task1.fusion_experiments import DIRECT, RESIDUAL, VALIDATION_RULES, fuse_predictions


def protocol():
    return dict(panels=['D7_I405_N'], fit_start='2030-06-01', fit_end='2030-12-31',
        eval_start='2031-01-01', eval_end='2031-01-31', max_fit_days=None, max_eval_days=None,
        seed=20261016, max_gap_minutes=60, rounds=350, sample_per_day=1000, threads=1, no_log=False)


def test_new_rule_60_boundary_missing_sides_and_alignment():
    keys = pd.DataFrame(dict(panel=['P']*5, timestamp=['t1','t2','t3','t4','t5'],
        station_id=['S']*5, link_id=['L']*5, mask_regime=['R1']*5))
    r, d = keys.copy(), keys.copy()
    r[TARGETS], d[TARGETS] = [[10., 600.]]*5, [[30., 1200.]]*5
    result = fuse_predictions(keys, r.iloc[::-1], d, [15.,60.,60.001,np.nan,np.inf], rules=VALIDATION_RULES)
    assert set(result) == set(cv.METHODS)
    np.testing.assert_array_equal(result['Fusion_gap60mean'][TARGETS],
        [[20.,900.],[20.,900.],[30.,1200.],[30.,1200.],[30.,1200.]])
    np.testing.assert_array_equal(result['Fusion_mean50'][TARGETS], [[20.,900.]]*5)
    with pytest.raises(ValueError, match='rules'):
        fuse_predictions(keys, r, d, [15.]*5, rules=['unknown'])


def test_protocol_keeps_new_family_fixed_budget_and_february_sealed():
    families = {'D7_I405_N':'D7_I405','D12_I5_S':'D12_I5'}
    cv.check_protocol(protocol(), families)
    for change in [dict(panels=['D12_I5_S']), dict(panels=['D7_I405_N']*2),
                   dict(eval_end='2031-02-28'), dict(rounds=700), dict(seed=2), dict(sample_per_day=500)]:
        with pytest.raises(ValueError):
            cv.check_protocol({**protocol(), **change}, families)
    cv.check_protocol({**protocol(), 'max_fit_days':14, 'rounds':50, 'sample_per_day':300}, families)


def test_model_pair_cache_reuses_fits_and_invalidates_budget(monkeypatch, tmp_path):
    calls = []
    cfg = protocol()
    train = dict(key='train', artifacts=[dict(name='training.parquet',sha256='sample')],
        metadata=dict(training_rows=30, fit_dates=['2030-06-01'],feature_columns=['speed_initial','mh_x','cg_x']))
    labels = dict(key='labels',artifacts=[dict(name='direct_targets.parquet',sha256='labels')])
    def fake_fit(folder, training, destination, config, variant, target_mode, label_folder, metadata_name):
        calls.append(target_mode)
        destination.mkdir(parents=True)
        save_json(destination / metadata_name, dict(target_mode=target_mode, feature_variant=variant,
            parameters=cv.model_parameters(config), config=config, training_cache_key='train',
            training_table_sha256='sample', training_rows=30, fit_dates=['2030-06-01'],
            direct_targets_sha256='labels' if target_mode=='direct' else None,
            features=['speed_initial','mh_x']))
    monkeypatch.setattr(cv, 'fit_cached', fake_fit)
    cache = FeatureCache(tmp_path / 'cache')
    args = (cache,'P',tmp_path,train,tmp_path,labels)
    first = cv.paired_model_cache(*args,cfg)
    second = cv.paired_model_cache(*args,cfg)
    assert not first[2] and second[2] and calls == ['residual','direct']
    changed = cv.paired_model_cache(*args,{**cfg,'rounds':700})
    assert not changed[2] and changed[0] != first[0] and len(calls)==4
    (first[0]/'direct/target_model_config.json').write_text('{}',encoding='utf-8')
    with pytest.raises(CacheIntegrityError):
        cv.paired_model_cache(*args,cfg)


def test_prediction_cache_skips_inference_on_hit(monkeypatch,tmp_path):
    keys = pd.DataFrame(dict(panel=['P'],timestamp=['2031-01-01T00:00:00Z'],
        station_id=['S'],link_id=['L'],mask_regime=['R1']))
    pred = keys.copy(); pred[TARGETS]=[[50.,1200.]]
    diagnostics=keys.copy(); diagnostics['gap_minutes']=60.; diagnostics['gap']='30-60min'
    calls=[]
    def fake_predict(*args,**kwargs):
        calls.append(1)
        return keys,{RESIDUAL:pred,DIRECT:pred},diagnostics
    monkeypatch.setattr(cv,'predict_cached',fake_predict)
    monkeypatch.setattr(cv,'cached_diagnostics',lambda *args: diagnostics)
    cache=FeatureCache(tmp_path/'cache')
    evaluation=dict(key='eval',metadata=dict(eval_dates=['2031-01-01']))
    args=(cache,'P',tmp_path,dict(key='models'),tmp_path,evaluation,protocol())
    first=cv.paired_prediction_cache(*args); second=cv.paired_prediction_cache(*args)
    assert not first[2] and second[2] and calls==[1]
    pd.testing.assert_frame_equal(pd.read_parquet(first[0]/f'{DIRECT}_predictions.parquet'),pred)


def completed_run(tmp_path):
    run=tmp_path/'corridor_example'; run.mkdir()
    cfg=protocol()
    save_json(run/'run_config.json',cfg)
    save_json(run/'COMPLETE.json',dict(experiment='corridor_validation',panels=['D7_I405_N'],
        methods=cv.METHODS,pairing_passed=True,integrity_passed=True))
    for name in ['pairing_check.json','integrity_check.json']:
        save_json(run/name,dict(passed=True,records=[dict(panel='D7_I405_N',passed=True)]))
    save_json(run/'cache_usage.json',[dict(panel='D7_I405_N',training_hit=True,evaluation_hit=True,
        direct_labels_hit=True,model_pair_hit=False,prediction_hit=False)])
    table=pd.DataFrame([dict(method=m,S_state=.9+i*.001,rmse_speed_kmh=2.,rmse_flow_per_lane=40.,
        delta_S_state_vs_residual=i*.001,delta_S_state_vs_direct=(i-1)*.001,
        delta_S_state_vs_mean50=(i-2)*.001,all_regimes_present=True) for i,m in enumerate(cv.METHODS)])
    table.to_csv(run/'comparison_corridor.csv',index=False)
    table['panel']='D7_I405_N'; table.to_csv(run/'comparison_by_panel.csv',index=False)
    return run,cfg


def test_corridor_log_preserves_history_deduplicates_and_requires_checks(tmp_path):
    run,_=completed_run(tmp_path)
    log=tmp_path/'log.txt'; log.write_text('历史\n',encoding='utf-8-sig')
    assert append_corridor_log(run,log)['status']=='appended'
    text=log.read_text(encoding='utf-8-sig')
    assert text.startswith('历史\n') and all(m in text for m in cv.METHODS)
    assert 'Fusion_gap60mean：S_state=0.903000' in text
    assert append_corridor_log(run,log)['status']=='already_logged'
    assert log.read_text(encoding='utf-8-sig')==text
    save_json(run/'integrity_check.json',dict(passed=True,records=[]))
    with pytest.raises(ValueError,match='panel checks'):
        append_corridor_log(run,log)


@pytest.mark.parametrize('option,value,status',[
    ('max_fit_days',14,'skipped_smoke'),('max_eval_days',3,'skipped_smoke'),('no_log',True,'skipped_by_option')])
def test_corridor_subset_and_optout_skip_formal_log(tmp_path,option,value,status):
    run,cfg=completed_run(tmp_path);cfg[option]=value;save_json(run/'run_config.json',cfg)
    log=tmp_path/'log.txt'
    assert append_corridor_log(run,log)['status']==status and not log.exists()
