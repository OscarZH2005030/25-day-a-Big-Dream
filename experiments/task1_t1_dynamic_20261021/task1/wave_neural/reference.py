from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..data import KEY,TARGETS,sha256
from ..fusion_experiments import aligned_prediction
from ..grin.hybrid import check_scores
from ..traffic_wave.reference import load_sources,read,verified_manifest

CONTROL='Current_best_wave_hybrid'
METHODS=[CONTROL,'Neural_only','Wave_neural','Wave_neural_FD','Wave_neural_FD_LWR']


def sources(cfg):
    base=load_sources(cfg)
    wave=Path(cfg['wave_run'])
    complete,old=read(wave/'COMPLETE.json'),read(wave/'run_config.json')
    if complete.get('experiment')!='traffic_wave' or complete.get('smoke') or not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed']):
        raise ValueError('Use a completed formal traffic-wave experiment as the current best')
    for name in ['panel','seed','sample_per_day','fit_start','fit_end','eval_start','eval_end','release_root','baseline_run','grin_run','hybrid_run']:
        if old[name]!=cfg[name]:
            raise ValueError('Current-best wave source protocol differs: '+name)
    if complete['training_rows']!=214000 or complete['evaluation_rows']!=550149:
        raise ValueError('Current-best experiment has incomplete target scope')
    if not all(read(wave/name).get('passed') for name in ['reference_check.json','sample_check.json','integrity_check.json']):
        raise ValueError('Current-best source checks failed')
    for item in read(wave/'environment.json')['source_files']:
        if sha256(Path(item['path']))!=item['sha256']:
            raise ValueError('Frozen wave implementation changed')
    base['control']=aligned_prediction(base['expected'],pd.read_parquet(wave/'Wave_enhanced_hybrid_predictions.parquet'))
    base['control_detail']=pd.read_csv(wave/'metrics_by_panel_regime.csv')
    base['control_detail']=base['control_detail'].loc[base['control_detail'].method.eq('Wave_enhanced_hybrid')].assign(method=CONTROL)
    base['control_daily']=pd.read_csv(wave/'Wave_enhanced_hybrid_daily_errors.csv')
    usage=read(wave/'cache_usage.json');cache=Path(old['cache_dir'])
    if not usage['original_control_predictions_reused'] or usage['original_grin_retrained']:
        raise ValueError('Current-best source model provenance differs')
    base['wave_records']={}
    for record in usage['training_days']+usage['evaluation_days']:
        folder=cache/record['key'];manifest=verified_manifest(folder)
        if manifest['identity']['source_grid_key']!=record['source_grid_key'] or manifest['identity']['kernel_key']!=usage['kernel']['key']:
            raise ValueError('Wave source identity differs')
        base['wave_records'][record['date']]=(folder,manifest)
    if len(base['wave_records'])!=245:
        raise ValueError('Expected 214 training and 31 evaluation wave days')
    base['provenance'] += [wave/name for name in ['Wave_enhanced_hybrid_predictions.parquet','metrics_by_panel_regime.csv',
        'Wave_enhanced_hybrid_daily_errors.csv','run_config.json','COMPLETE.json','cache_usage.json','environment.json']]
    return base


def control_check(source,actual,dates,smoke):
    if not smoke:
        reference=source['control_detail']
    else:
        daily=source['control_daily'];daily=daily.loc[daily.date.isin(dates)]
        rows=[]
        for regime,g in daily.groupby('mask_regime'):
            speed=np.sqrt(np.average(g.speed_mse,weights=g.n))
            flow=np.sqrt(np.average(g.flow_lane_mse,weights=g.n))
            rows.append(dict(method=CONTROL,panel=actual.panel.iloc[0],regime=regime,n=int(g.n.sum()),
                rmse_speed_kmh=speed,rmse_flow_per_lane=flow,S_state=.54*max(0,1-speed/25)+.46*max(0,1-flow/600)))
        reference=pd.DataFrame(rows)
    result=check_scores(reference,actual,CONTROL)
    if not result['passed']:
        raise ValueError('Frozen 0.931453 control failed to reproduce')
    return result
