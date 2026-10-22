from __future__ import annotations

from pathlib import Path

from ..data import sha256
from ..traffic_wave.reference import read, verified_manifest
from ..wave_neural.reference import CONTROL, sources as original_sources, control_check

METHODS = [CONTROL, 'Long_context_neural', 'Long_context_wave_features']


def verify_files(items):
    for item in items:
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Frozen source changed: '+item['path'])


def sources(cfg):
    base = original_sources(cfg)
    run = Path(cfg['input_run'])
    complete, old = read(run/'COMPLETE.json'), read(run/'run_config.json')
    if complete.get('experiment')!='wave_neural' or complete.get('smoke') or complete.get('training_rows')!=214000 or complete.get('evaluation_rows')!=550149:
        raise ValueError('Input source must be a completed formal five-way I5 experiment')
    if not all(complete.get(k) for k in ['reference_passed', 'pairing_passed', 'integrity_passed', 'all_regimes_present']):
        raise ValueError('Input source is incomplete')
    for name in ['panel','seed','sample_per_day','fit_start','fit_end','eval_start','eval_end','release_root','baseline_run','grin_run','hybrid_run','wave_run']:
        if old[name] != cfg[name]:
            raise ValueError('Input source protocol differs: '+name)
    for name in ['reference_check.json', 'sample_check.json', 'integrity_check.json']:
        if not read(run/name).get('passed'):
            raise ValueError('Input source check failed: '+name)
    verify_files(read(run/'environment.json')['source_files'])
    usage = read(run/'cache_usage.json')
    cache = Path(old['cache_dir'])
    stats_folder = cache/usage['statistics']['key']
    attr_folder = cache/usage['physical_structure']['key']
    stats, attributes = verified_manifest(stats_folder), verified_manifest(attr_folder)
    si, ai = stats['identity'], attributes['identity']
    if si['kind']!='wave_neural_statistics' or si['training_grid_key']!=base['grid_train']['key']:
        raise ValueError('Training statistics belong to another grid')
    if ai['kind']!='wave_neural_physical_structure' or ai['nodes']!=base['grid_train']['metadata']['nodes']:
        raise ValueError('Static node attributes differ')
    verify_files(si['implementation']); verify_files(ai['implementation']); verify_files(ai['inputs'])
    records = {}
    for split, key in [('training','train'),('evaluation','eval')]:
        rows = usage[split+'_days']
        expected_dates = base['grid_'+key]['metadata']['dates']
        if [r['date'] for r in rows] != expected_dates:
            raise ValueError('Input dates/order differ from original grid')
        records[split] = []
        for record in rows:
            date = record['date']; folder = cache/record['key']
            manifest = verified_manifest(folder); identity = manifest['identity']
            grid_folder = base['grid_'+key+'_folder']
            wave_folder, wave = base['wave_records'][date]
            required = dict(kind='wave_neural_day_input', panel=cfg['panel'], date=date,
                grid_key=base['grid_'+key]['key'], grid_archive_sha256=sha256(grid_folder/(date+'.npz')),
                wave_key=wave['key'], statistics_key=stats['key'], physical_structure_key=attributes['key'], truth_used=False)
            if any(identity.get(k)!=v for k,v in required.items()):
                raise ValueError('Input parent identity differs: '+date)
            verify_files(identity['implementation']); verify_files(identity['ramp_inputs'])
            records[split].append(dict(record, hit=True, folder=str(folder), grid_folder=str(grid_folder), wave_folder=str(wave_folder)))
        print(f'[context inputs] verified existing {split} days: {len(rows)}; no rebuild', flush=True)
    base.update(input_records=records, stats_folder=stats_folder, stats=stats, attributes_folder=attr_folder, attributes_manifest=attributes)
    base['provenance'] += [run/name for name in ['COMPLETE.json','run_config.json','cache_usage.json','environment.json']]
    base['provenance'] += [stats_folder/'manifest.json', attr_folder/'manifest.json']
    return base
