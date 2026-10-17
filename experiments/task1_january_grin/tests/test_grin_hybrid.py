import argparse
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

from task1.cache import FeatureCache
from task1.data import KEY,TARGETS,save_json,sha256
from task1.grin import hybrid as h
from task1.grin.hybrid_log import append_hybrid_log


def predictions():
    keys=pd.DataFrame(dict(panel=['P']*5,timestamp=['t1','t2','t3','t4','t5'],
        station_id=['S']*5,link_id=['L']*5,mask_regime=['R1']*5))
    b,g=keys.copy(),keys.copy();b[TARGETS]=[[10.,1000.]]*5;g[TARGETS]=[[20.,2000.]]*5
    return keys,b,g


def config():
    return dict(panel='D7_I405_N',fit_start='2030-06-01',fit_end='2030-12-31',
        eval_start='2031-01-01',eval_end='2031-01-31',max_fit_days=None,max_eval_days=None,
        sample_per_day=1000,seed=20261016,hidden=32,batch_size=2,window=48,stride=24,
        epochs=20,steps_per_epoch=100,patience=5,learning_rate=.001,dropout=.1,
        drop_observed=.05,aux_weight=.1,release_root='data')


def test_hybrid_aligns_keys_switches_only_finite_over_60_and_keeps_total_flow():
    keys,b,g=predictions()
    out=h.switch_predictions(keys,b.iloc[::-1],g.sample(frac=1,random_state=1),[15.,60.,60.001,np.nan,np.inf])
    np.testing.assert_array_equal(out[TARGETS],[[10.,1000.],[10.,1000.],[20.,2000.],[10.,1000.],[10.,1000.]])
    assert out.selected_model.tolist()==[h.BASELINE,h.BASELINE,h.GRIN_METHOD,h.BASELINE,h.BASELINE]
    assert out.gap.iloc[-1]=='one_side_or_none'
    pd.testing.assert_frame_equal(out[KEY],keys)


def test_hybrid_rejects_missing_duplicate_predictions_or_invalid_gaps():
    keys,b,g=predictions()
    for broken in [g.iloc[:-1],pd.concat([g,g.iloc[:1]])]:
        with pytest.raises(ValueError):h.switch_predictions(keys,b,broken,[15.]*5)
    for gap in [[15.],[-1.]*5,[0.]*5,[-np.inf]*5]:
        with pytest.raises(ValueError):h.switch_predictions(keys,b,g,gap)


def test_hybrid_reference_check_detects_metrics_and_count_changes():
    d=pd.DataFrame([dict(panel='P',regime='R1',method=h.BASELINE,n=10,S_state=.9,
        rmse_speed_kmh=2.,rmse_flow_per_lane=40.)])
    assert h.check_scores(d,d,h.BASELINE)['passed']
    changed=d.copy();changed.loc[0,'n']=11
    assert not h.check_scores(d,changed,h.BASELINE)['passed']
    changed=d.copy();changed.loc[0,'S_state']+=.001
    assert not h.check_scores(d,changed,h.BASELINE)['passed']


def make_source(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'PROJECT',tmp_path)
    code=tmp_path/'task1/grin';code.mkdir(parents=True)
    for filename in ['data.py','model.py']:(code/filename).write_text('test source')
    metrics=tmp_path/'task1/metrics.py';metrics.write_text('test metrics')
    source=tmp_path/'runs/grin_hybrid_d7_example/grin_run';source.mkdir(parents=True)
    cfg=config();cfg['cache_dir']=str(tmp_path/'cache')
    save_json(source/'run_config.json',cfg)
    save_json(source/'COMPLETE.json',dict(experiment='grin_dev',panel=cfg['panel']))
    save_json(source/'sample_check.json',dict(passed=True))
    save_json(source/'environment.json',dict(source_sha256={str(__import__('pathlib').Path('task1/grin')/name):sha256(code/name)
        for name in ['data.py','model.py']},metric_sha256=sha256(metrics)))
    identity=dict(kind='grin_evaluation',panel=cfg['panel'],inputs=[])
    def build(folder):
        (folder/'inputs.bin').write_bytes(b'test masked cache')
        return {}
    _,manifest,_=FeatureCache(tmp_path/'cache').get_or_build(identity,build)
    save_json(source/'cache_usage.json',dict(evaluation_key=manifest['key']))
    _,_,g=predictions();g.to_parquet(source/'GRIN_predictions.parquet',index=False)
    return source,cfg


def test_completed_matching_grin_is_found_without_training_and_wrong_scope_rejected(tmp_path,monkeypatch):
    source,cfg=make_source(tmp_path,monkeypatch)
    assert h.find_grin(cfg)==source
    assert h.require_compatible_grin(source,cfg)==source
    assert h.find_grin({**cfg,'panel':'D12_I5_N'}) is None
    with pytest.raises(ValueError,match='configuration'):
        h.require_compatible_grin(source,{**cfg,'batch_size':4})
    with pytest.raises(ValueError,match='configuration'):
        h.require_compatible_grin(source,{**cfg,'max_fit_days':7})
    (tmp_path/'task1/grin/model.py').write_text('changed model')
    with pytest.raises(ValueError,match='implementation'):
        h.require_compatible_grin(source,cfg)


def test_requested_d7_training_keeps_i5_settings_and_correct_baseline_samples(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'PROJECT',tmp_path)
    source=tmp_path/'runs/i5';source.mkdir(parents=True)
    cfg={**config(),'panel':'D12_I5_N'}
    save_json(source/'run_config.json',cfg);save_json(source/'COMPLETE.json',dict(experiment='grin_dev'))
    args=argparse.Namespace(template_grin_run=source,output=tmp_path/'runs/new',panel='D7_I405_N',
        device='cpu',threads=4,batch_size=2,frozen_reference=tmp_path/'runs/ref',smoke=False)
    bundle=dict(config=cfg,training_table=tmp_path/'cache/d7/training.parquet')
    requested=h.requested_config(args,bundle)
    assert requested['panel']=='D7_I405_N'
    assert requested['baseline_sample_table']==str(bundle['training_table'])
    for key in h.CONFIG_KEYS:
        if key!='panel':assert requested[key]==cfg[key]
    args.smoke=True
    requested=h.requested_config(args,bundle)
    assert requested['max_fit_days']==7 and requested['epochs']==2


def completed_log(tmp_path):
    run=tmp_path/'hybrid';run.mkdir()
    source=tmp_path/'grin_source';source.mkdir()
    save_json(source/'selection.json',dict(best_epoch=18))
    methods=[h.BASELINE,h.GRIN_METHOD,h.HYBRID]
    cfg=dict(panel='D7_I405_N',baseline_run='old',fit_start='2030-06-01',fit_end='2030-12-31',
        eval_start='2031-01-01',eval_end='2031-01-31',smoke=False,no_log=False,training_config=config())
    save_json(run/'run_config.json',cfg)
    save_json(run/'COMPLETE.json',dict(experiment='grin_hybrid',reference_passed=True,panel=cfg['panel'],
        methods=methods,evaluated_rows=30,gated_rows=3))
    save_json(run/'reference_check.json',dict(passed=True))
    save_json(run/'reuse.json',dict(grin_source=str(source),baseline_predictions_reused=True,
        baseline_models_retrained=False,grin_trained_this_run=False,grin_predictions_reused=True))
    pd.DataFrame([dict(method=m,S_state=.9+i*.001,rmse_speed_kmh=2.,rmse_flow_per_lane=40.,
        delta_S_state_vs_frozen_baseline=i*.001,delta_S_state_vs_GRIN=(i-1)*.001,all_regimes_present=True)
        for i,m in enumerate(methods)]).to_csv(run/'comparison_hybrid.csv',index=False)
    pd.DataFrame([dict(method=m,regime=r,n=10,S_state=.9) for m in methods for r in ['R1','R2','R3']]).to_csv(run/'metrics_by_panel_regime.csv',index=False)
    return run,cfg


def test_hybrid_log_preserves_history_deduplicates_and_requires_reference(tmp_path):
    run,_=completed_log(tmp_path);log=tmp_path/'log.txt';log.write_text('历史\n',encoding='utf-8-sig')
    assert append_hybrid_log(run,log)['status']=='appended'
    text=log.read_text(encoding='utf-8-sig')
    assert text.startswith('历史\n') and 'Baseline_GRIN_gap60：S_state=0.902000' in text
    assert append_hybrid_log(run,log)['status']=='already_logged'
    assert log.read_text(encoding='utf-8-sig')==text
    save_json(run/'reference_check.json',dict(passed=False))
    with pytest.raises(ValueError,match='reference'):append_hybrid_log(run,log)


@pytest.mark.parametrize('field,status',[('smoke','skipped_smoke'),('no_log','skipped_by_option')])
def test_hybrid_smoke_and_optout_skip_formal_log(tmp_path,field,status):
    run,cfg=completed_log(tmp_path);cfg[field]=True;save_json(run/'run_config.json',cfg)
    log=tmp_path/'log.txt'
    assert append_hybrid_log(run,log)['status']==status and not log.exists()
