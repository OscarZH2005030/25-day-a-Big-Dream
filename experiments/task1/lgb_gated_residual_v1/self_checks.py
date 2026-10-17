"""CPU synthetic regression checks. No LightGBM fitting or optimizer steps."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from common import (HERE, KEYS, SCALE, RunLock, baseline_grid, make_folds, neural_features,
                    read_json, save_array, save_json)
from crossfit import ensure_day
from dataset import DayStore, graph_tensors
from model import GatedResidual, objective


def main():
    torch.set_num_threads(2); torch.manual_seed(2026)
    assert not torch.cuda.is_initialized()
    results = []
    dates = pd.date_range('2030-06-29', periods=168).strftime('%Y-%m-%d').tolist()
    folds = make_folds(dates)
    assert [len(f['predict_dates']) for f in folds] == [42]*4
    for fold in folds:
        assert len(fold['refit_dates']) == 126 and len(fold['inner_fit_dates']) == 112 and len(fold['inner_valid_dates']) == 14
        assert set(fold['predict_dates']).isdisjoint(fold['inner_fit_dates']+fold['inner_valid_dates'])
    results.append('4x42 outer prediction dates excluded from both inner selection and 126-day refit')
    graph = dict(graph=np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
        direction=np.sign(np.arange(3)[None, :]-np.arange(3)[:, None]).astype('int64'), isolated=np.array([False, False, True]))
    g = graph_tensors(graph, np.array([0, 2]))
    assert g['neighbor_valid'].shape == (2, 1) and g['isolated'].tolist() == [False, True]
    batch = dict(features=torch.randn(2, 24, 2, 106), base=torch.ones(2, 24, 2, 2),
                 visible=torch.zeros(2, 24, 2, 2, dtype=torch.bool), target=torch.ones(2, 24, 2, dtype=torch.bool),
                 residual=torch.randn(2, 24, 2, 2)*.1)
    model = GatedResidual(106, 16, 4, 1, 24, 0)
    output = model(batch, g)
    torch.testing.assert_close(output['prediction'], batch['base'], rtol=0, atol=0)
    loss, _ = objective(output, batch, g, .001); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    results.append('zero initialization exactly equals LightGBM; CPU backward has finite nonzero gradients; no optimizer step')
    with torch.no_grad():
        model.residual_heads[0][-1].bias.fill_(1.)
        model.residual_heads[1][-1].bias.fill_(2.)
        model.gates[0][-1].bias.fill_(-100.)
        model.gates[1][-1].bias.fill_(100.)
        batch['visible'][0, 0, 0, 1] = True
        output = model(batch, g)
        torch.testing.assert_close(output['prediction'][..., 1, :], batch['base'][..., 1, :], rtol=0, atol=0)
        torch.testing.assert_close(output['prediction'][..., 0, 0], batch['base'][..., 0, 0], rtol=0, atol=1e-6)
        assert output['prediction'][1, 1, 0, 1] > batch['base'][1, 1, 0, 1]
        assert output['prediction'][0, 0, 0, 1] == batch['base'][0, 0, 0, 1]
    results.append('independent gates, global isolation despite local patch isolation, visible observations anchored')
    batch['target'][:, 0] = False
    a, _ = objective(output, batch, g, .001)
    batch['residual'][:, 0] += 1000
    b, _ = objective(output, batch, g, .001)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    results.append('non-template residual labels cannot change loss')

    class FakeBooster:
        def __init__(self, model_file): self.value = 2. if 'speed' in str(model_file) else 12.
        def predict(self, features, num_threads): return np.full(len(features), self.value)
    with tempfile.TemporaryDirectory(prefix='synthetic_', dir=HERE) as temporary:
        root = Path(temporary)
        assert root.resolve().is_relative_to(HERE.resolve())
        with RunLock(root/'lock'):
            try:
                with RunLock(root/'lock'): pass
            except RuntimeError: pass
            else: raise AssertionError('Duplicate lock was accepted')
        with RunLock(root/'lock'): pass
        args = SimpleNamespace(reference_run=root/'reference', output=root/'output', threads=2, window=24)
        raw = np.ones((288, 3, 51), dtype='float32'); raw[0, 0, 1] = np.nan
        interpolation = np.ones((288, 3, 2), dtype='float32')
        seen = np.zeros_like(interpolation, dtype=bool); seen[0, 1] = True
        observed = np.full_like(interpolation, .75)
        labels = np.full_like(interpolation, 1.25)
        ii, jj = np.array([0, 1]), np.array([1, 2])
        model_paths = [root/'speed.txt', root/'flow.txt']
        for p in model_paths: p.write_text('FAKE predictor, not fitted')
        models = [FakeBooster(p) for p in model_paths]
        predicted = baseline_grid(raw, interpolation, seen, observed, models, 2)
        x = neural_features(raw, predicted, interpolation, np.zeros(51), np.ones(51))
        assert x.shape[-1] == 106 and x[0, 0, 1] == 0 and x[0, 0, 52] == 0
        original_lgb = sys.modules.get('lightgbm')
        sys.modules['lightgbm'] = SimpleNamespace(Booster=FakeBooster)
        try:
            for index, date in enumerate(['2030-06-29', '2030-06-30', '2030-07-01']):
                f = args.reference_run/'synthetic'/'cache'/date; f.mkdir(parents=True)
                current = labels.copy()
                if index == 1: current[jj, ii] += .5
                if index == 2: current[200, 2] += 99
                for name, value in dict(features=raw, base=interpolation, visible=seen, observed=observed,
                    label=current, ii=ii, jj=jj).items(): save_array(f/(name+'.npy'), value)
                keys = pd.DataFrame(dict(panel=['synthetic']*2, timestamp=[date+'T00:05:00Z', date+'T00:10:00Z'],
                    station_id=['s0', 's1'], link_id=['l0', 'l1'], mask_regime=['R1', 'R2']))
                keys.to_parquet(f/'keys.parquet', index=False)
                ensure_day(args, 'synthetic', date, model_paths, 'oof_fold_0', supervised=True)
            first, second, third = [args.output/'synthetic'/'baseline'/d for d in ['2030-06-29', '2030-06-30', '2030-07-01']]
            np.testing.assert_array_equal(np.load(first/'base.npy'), np.load(second/'base.npy'))
            np.testing.assert_allclose(np.load(second/'residual_targets.npy')-np.load(first/'residual_targets.npy'), .5, atol=1e-6)
            np.testing.assert_array_equal(np.load(first/'residual_targets.npy'), np.load(third/'residual_targets.npy'))
            np.testing.assert_allclose(np.load(first/'residual_targets.npy'), labels[jj, ii]-predicted[jj, ii], atol=0, rtol=0)
            save_json(args.reference_run/'synthetic'/'cache/ready.json', dict(mean=[0.]*51, scale=[1.]*51))
            # Exercise actual training block assembly, without reading any raw labels.
            target = np.zeros((288, 3), dtype=bool); target[jj, ii] = True
            save_array(args.reference_run/'synthetic'/'cache/2030-06-29/target.npy', target)
            ds = DayStore.__new__(DayStore)
            from collections import OrderedDict
            ds.args, ds.source, ds.baselines = args, args.reference_run/'synthetic'/'cache', args.output/'synthetic'/'baseline'
            ds.plan = dict(training_dates=['2030-06-29']); ds.mean, ds.scale, ds.n = np.zeros(51), np.ones(51), 3
            ds._residual = OrderedDict()
            (ds.source/'2030-06-29/label.npy').unlink()
            block = ds.block('2030-06-29', 0, np.array([0, 1]), supervised=True)
            assert block['features'].shape == (24, 2, 106)
            torch.testing.assert_close(block['residual'][1, 0], torch.from_numpy(np.load(first/'residual_targets.npy')[0]))
        finally:
            if original_lgb is None: sys.modules.pop('lightgbm', None)
            else: sys.modules['lightgbm'] = original_lgb
        state = torch.get_rng_state(); expected = torch.rand(8); torch.set_rng_state(state)
        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)
        torch.save(dict(state=state, weights=model.state_dict()), root/'checkpoint.pt')
        restored = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=False)
        torch.set_rng_state(restored['state'])
        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)
    results.extend(['OS lock rejects duplicate writers and releases after context',
        'changing hidden target labels changes only residual targets, never baseline/input; non-target labels ignored',
        'actual disk block loader works after removing raw label file', '106 feature order/mask and scaled residual units checked',
        'CPU checkpoint/RNG round trip checked; CUDA resume not exercised'])
    assert not torch.cuda.is_initialized()
    save_json(HERE/'self_checks.json', dict(passed=True, checks=results, training_started=False,
        tree_fitting_started=False, optimizer_steps=0, device='cpu', cuda_initialized=False))
    for result in results: print('PASS', result, flush=True)


if __name__ == '__main__': main()
