"""CPU-only, nested blocked OOF LightGBM and compact residual target caches."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from common import (CHANNELS, SCALE, baseline_grid, make_folds, read_json, save_json, save_array, sha)

_bundle = None


def model_bundle(paths, load=False):
    """Keep one immutable predictor pair in memory across its date grids."""
    global _bundle
    signature = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    if _bundle is None or _bundle['signature'] != signature:
        _bundle = dict(signature=signature, hashes=[sha(p) for p in paths], models=None)
    if load and _bundle['models'] is None:
        import lightgbm as lgb
        _bundle['models'] = [lgb.Booster(model_file=str(p)) for p in paths]
    return _bundle


def params(args):
    return dict(objective='regression', metric='rmse', learning_rate=.05, num_leaves=31,
        min_data_in_leaf=80, lambda_l2=1., feature_fraction=.9, num_threads=args.threads,
        seed=args.seed, deterministic=True, force_col_wise=True, verbosity=-1, device_type='cpu')


def materialize(cache, dates, scratch):
    sizes = [len(np.load(cache/d/'ii.npy', mmap_mode='r')) for d in dates]
    x = np.lib.format.open_memmap(scratch/'x.npy', mode='w+', dtype='float32', shape=(sum(sizes), 51))
    y = np.lib.format.open_memmap(scratch/'y.npy', mode='w+', dtype='float64', shape=(sum(sizes), 2))
    cursor = 0
    for k, (date, size) in enumerate(zip(dates, sizes)):
        folder = cache/date
        ii, jj = np.load(folder/'ii.npy'), np.load(folder/'jj.npy')
        end = cursor+size
        x[cursor:end] = np.load(folder/'features.npy', mmap_mode='r')[jj, ii]
        truth = np.load(folder/'label.npy', mmap_mode='r')[jj, ii].astype('float64')*SCALE
        initial = np.load(folder/'base.npy', mmap_mode='r')[jj, ii].astype('float64')*SCALE
        if not np.isfinite(truth).all() or not np.isfinite(initial).all(): raise ValueError('Invalid target')
        y[cursor:end] = truth-initial
        cursor = end
        if (k+1)%28 == 0 or k == len(dates)-1:
            print('  materialize', k+1, '/', len(dates), 'days;', cursor, 'targets', flush=True)
    x.flush(); y.flush()
    return x, y, sizes


def fit_fold(args, plan, panel, fold):
    import lightgbm as lgb
    cache = args.reference_run/panel/'cache'
    folder = args.output/panel/'crossfit'/('fold_'+str(fold['fold']))
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'COMPLETE.json'
    paths = [folder/(channel+'.txt') for channel in CHANNELS]
    if marker.exists():
        old = read_json(marker)
        if old['fold'] != fold or old['model_sha256'] != [sha(p) for p in paths]: raise ValueError('OOF model changed')
        return paths
    scratch = folder/'scratch'; scratch.mkdir(exist_ok=True)
    # Ordered refit rows make the final inner validation dates a suffix.
    dates = fold['inner_fit_dates']+fold['inner_valid_dates']
    assert dates == fold['refit_dates'] and set(dates).isdisjoint(fold['predict_dates'])
    x, y, sizes = materialize(cache, dates, scratch)
    cut = sum(sizes[:len(fold['inner_fit_dates'])])
    names = read_json(cache/'ready.json')['names']
    choices = []
    for ci, channel in enumerate(CHANNELS):
        print(panel, 'OOF fold', fold['fold'], channel, 'inner selection; held dates excluded', flush=True)
        train = lgb.Dataset(x[:cut], label=np.asarray(y[:cut, ci]).copy(), feature_name=names)
        valid = lgb.Dataset(x[cut:], label=np.asarray(y[cut:, ci]).copy(), reference=train)
        trace = {}
        selection = lgb.train(params(args), train, num_boost_round=args.max_trees,
            valid_sets=[valid], valid_names=['inner'], callbacks=[lgb.early_stopping(args.tree_patience, first_metric_only=True),
                lgb.record_evaluation(trace), lgb.log_evaluation(100)])
        count = max(1, selection.best_iteration)
        save_json(folder/(channel+'_inner_trace.json'), trace)
        del train, valid, selection; gc.collect()
        print(panel, 'OOF fold', fold['fold'], channel, 'refit all', len(dates), 'non-held days;', count, 'trees', flush=True)
        full = lgb.Dataset(x, label=np.asarray(y[:, ci]).copy(), feature_name=names)
        fitted = lgb.train(params(args), full, num_boost_round=count)
        temp = paths[ci].with_suffix('.tmp')
        fitted.save_model(str(temp)); temp.replace(paths[ci])
        choices.append(dict(channel=channel, trees=count))
        del full, fitted; gc.collect()
    del x, y; gc.collect()
    for name in ['x.npy', 'y.npy']: (scratch/name).unlink()
    save_json(marker, dict(fold=fold, choices=choices, model_sha256=[sha(p) for p in paths],
        selection_uses='inner dates from non-held supervised training dates only',
        prediction_dates_excluded_from_fit_and_selection=True))
    return paths


def ensure_day(args, panel, date, model_paths, producer, supervised):
    """A marker commits the baseline plus optional OOF residual target together."""
    import lightgbm as lgb
    folder = args.output/panel/'baseline'/date
    folder.mkdir(parents=True, exist_ok=True)
    cache = args.reference_run/panel/'cache'/date
    identity = dict(date=date, producer=producer, models_sha256=model_bundle(model_paths)['hashes'],
        keys_sha256=sha(cache/'keys.parquet'), supervised_oof=supervised)
    marker = folder/'COMPLETE.json'
    if marker.exists():
        old = read_json(marker)
        if old['identity'] != identity: raise ValueError('Baseline date producer changed: '+date)
        for name, digest in old['artifacts'].items():
            if sha(folder/name) != digest: raise ValueError('Baseline cache changed: '+date+'/'+name)
        return old
    print(panel, 'baseline grid', date, producer, flush=True)
    raw = np.load(cache/'features.npy', mmap_mode='r')
    interpolation = np.load(cache/'base.npy', mmap_mode='r')
    visible = np.load(cache/'visible.npy', mmap_mode='r').astype(bool)
    observed = np.load(cache/'observed.npy', mmap_mode='r')
    models = model_bundle(model_paths, load=True)['models']
    base = baseline_grid(raw, interpolation, visible, observed, models, args.threads)
    if not np.isfinite(base).all() or (base < 0).any(): raise ValueError('Invalid LightGBM grid')
    np.testing.assert_array_equal(base[visible], np.asarray(observed)[visible])
    save_array(folder/'base.npy', base)
    artifacts = {'base.npy': sha(folder/'base.npy')}
    if supervised:
        ii, jj = np.load(cache/'ii.npy'), np.load(cache/'jj.npy')
        if visible[jj, ii].any(): raise ValueError('Template target is visible')
        # Only target labels are ever read here. No non-target labels enter inputs.
        truth = np.load(cache/'label.npy', mmap_mode='r')[jj, ii]
        residual = (truth-base[jj, ii]).astype('float32')
        if not np.isfinite(residual).all(): raise ValueError('Invalid residual labels')
        save_array(folder/'residual_targets.npy', residual)
        artifacts['residual_targets.npy'] = sha(folder/'residual_targets.npy')
    result = dict(identity=identity, artifacts=artifacts, shape=list(base.shape),
                  units='speed/25 and flow_per_lane/600; observed entries copied from masked observations')
    save_json(marker, result)
    return result


def prepare(args, plan, panel):
    folder = args.output/panel
    if (folder/'OOF_READY.json').exists():
        verify_ready(args, plan, panel)
        print(panel, 'OOF grids already complete', flush=True)
        return
    folds = make_folds(plan['training_dates'], args.folds, args.inner_days)
    save_json(folder/'folds.json', folds)
    for fold in folds:
        paths = fit_fold(args, plan, panel, fold)
        for date in fold['predict_dates']:
            ensure_day(args, panel, date, paths, 'oof_fold_'+str(fold['fold']), supervised=True)
    final = [args.final_lgb/panel/(c+'.txt') for c in CHANNELS]
    for date in plan['internal_validation_dates']:
        ensure_day(args, panel, date, final, 'final_full168', supervised=False)
    markers = {d: sha(folder/'baseline'/d/'COMPLETE.json') for d in plan['training_dates']+plan['internal_validation_dates']}
    save_json(folder/'OOF_READY.json', dict(folds=folds, markers=markers,
        residual_dates=plan['training_dates'], final_model_dates=plan['internal_validation_dates'],
        holdout_labels_accessed=False, normalization='fixed source normalization from 168 supervised days'))
    verify_ready(args, plan, panel)


def verify_ready(args, plan, panel):
    folder = args.output/panel; ready = read_json(folder/'OOF_READY.json')
    if ready['folds'] != make_folds(plan['training_dates'], args.folds, args.inner_days): raise ValueError('Fold manifest mismatch')
    if set(ready['markers']) != set(plan['training_dates']+plan['internal_validation_dates']): raise ValueError('Incomplete OOF dates')
    if ready['residual_dates'] != plan['training_dates']: raise ValueError('Residual dates mismatch')
    model_hashes = {}
    def cached_hash(path):
        if path not in model_hashes: model_hashes[path] = sha(path)
        return model_hashes[path]
    for date, digest in ready['markers'].items():
        marker = folder/'baseline'/date/'COMPLETE.json'
        if sha(marker) != digest: raise ValueError('Baseline manifest changed: '+date)
        info = read_json(marker)
        for name, expected in info['artifacts'].items():
            if sha(marker.parent/name) != expected: raise ValueError('Baseline artifact changed: '+date+'/'+name)
        if date in plan['training_dates']:
            producer = info['identity']['producer']
            fold = next(f for f in ready['folds'] if producer == 'oof_fold_'+str(f['fold']))
            assert date in fold['predict_dates'] and date not in fold['refit_dates']
            models = folder/'crossfit'/('fold_'+str(fold['fold']))
            paths = [models/(c+'.txt') for c in CHANNELS]
        else:
            paths = [args.final_lgb/panel/(c+'.txt') for c in CHANNELS]
        if [cached_hash(p) for p in paths] != info['identity']['models_sha256']: raise ValueError('Producer model changed')
    return ready
