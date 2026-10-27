"""Engineering acceptance only: four updates, tiny trees, no formal results."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from .cache import atomic_json, sha256_file
from .data import FEATURE_NAMES, day, load_sample, make_sample, network, prepare, released_sample
from .metrics import aggregate_metrics, window_iou
from .model import QueueForecaster, masked_queue_loss
from .protocol import PROTOCOL, PROTOCOL_SHA256
from .submission import validate_queue


def tensor_input(sample, device):
    net = sample['network']
    return dict(x=torch.as_tensor(sample['x'], device=device).unsqueeze(0),
                static=torch.as_tensor(net['static'], device=device),
                edges=torch.as_tensor(net['edges'], dtype=torch.long, device=device),
                condition=torch.tensor([sample['condition']], dtype=torch.long, device=device),
                length_km=torch.as_tensor(net['length'], device=device))


def _records(samples, probabilities):
    rows = []
    for sample, prob in zip(samples, probabilities):
        rows.append({'panel': sample['panel'], 'condition': sample['condition'],
                     'window_id': sample['panel'] + '/' + sample['origin'],
                     'iou': window_iou(sample['y'], np.asarray(prob) >= .5, sample['m']),
                     'has_future_queue': sample['has_future_queue']})
    return rows


def _diagnostic(samples, probabilities):
    records = _records(samples, probabilities)
    primary = [r for r in records if r['has_future_queue']]
    return {'all_windows_diagnostic': aggregate_metrics(records),
            'future_positive_primary_diagnostic': aggregate_metrics(primary) if primary else None,
            'engineering_only': True, 'official_S_queue': None,
            'decision_threshold': .5, 'threshold_tuning_performed': False}


def run_smoke(args, panels):
    if not 1 <= args.steps <= 4:
        raise ValueError('This package permits engineering smoke, at most four optimizer updates')
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    random.seed(20261024)
    np.random.seed(20261024)
    torch.manual_seed(20261024)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use setup_task2.ps1 and check the NVIDIA driver')
        torch.cuda.manual_seed_all(20261024)
        torch.cuda.reset_peak_memory_stats()
    manifest = prepare(args.release_root, args.cache_dir, args.output / 'dataset', panels=panels,
                       roles=('fit', 'tune'), max_days=args.max_days, max_windows=args.max_windows,
                       stride=args.stride, engineering=True)
    index_path = args.output / 'dataset' / 'index.json'
    samples = [load_sample(index_path, entry, cache_dir=args.cache_dir) for entry in manifest['samples']]
    fit = [s for s in samples if s['role'] == 'fit']
    tune = [s for s in samples if s['role'] == 'tune']
    if not fit or not tune:
        raise ValueError('Need nonempty fitting and tuning samples for a real data smoke')
    if any(s['role'] in ('calibrate', 'audit') for s in samples):
        raise AssertionError('Reserved label roles must never be opened by smoke')
    for sample in samples:
        assert np.isfinite(sample['x']).all()
        assert sample['x'].shape[-1] == len(FEATURE_NAMES)
    print(f'[smoke] bounded dataset: fit={len(fit)} tune={len(tune)}, engineering only', flush=True)

    # Re-prepare separately: verify content-addressed payloads were reused rather
    # than asserting reuse merely because a folder happened to exist.
    cache_before = {p.name: (p.stat().st_mtime_ns, sha256_file(p)) for p in args.cache_dir.glob('*.npz')}
    prepare(args.release_root, args.cache_dir, args.output / 'cache_replay', panels=panels,
            roles=('fit', 'tune'), max_days=args.max_days, max_windows=args.max_windows,
            stride=args.stride, engineering=True)
    cache_after = {p.name: (p.stat().st_mtime_ns, sha256_file(p)) for p in args.cache_dir.glob('*.npz')}
    assert cache_before == cache_after, 'An unchanged cache payload was unnecessarily rewritten'

    # Future-data perturbation reaches the actual feature constructor. Neither
    # changed future observations nor future eligibility may change model input.
    first = manifest['samples'][0]
    original_day = day(args.release_root, first['panel'], first['date'], args.cache_dir)
    altered = deepcopy(original_day)
    for prefix in ['', 'origin_']:
        for field in ['speed', 'flow', 'occupancy', 'quality']:
            altered[prefix + field][first['origin_slot'] + 1:] = 999.
        altered[prefix + 'eligible'][first['origin_slot'] + 1:] = False
    reference_net = network(args.release_root, first['panel'])
    reference = make_sample(original_day, reference_net, first['origin_slot'], first['panel'], first['role'])
    changed = make_sample(altered, reference_net, first['origin_slot'], first['panel'], first['role'])
    np.testing.assert_array_equal(reference['x'], changed['x'])
    assert reference['condition'] == changed['condition']
    assert not np.array_equal(reference['m'], changed['m']), 'Future perturbation must actually change supervision'

    model_cfg = dict(PROTOCOL['default_model'])
    model = QueueForecaster(len(FEATURE_NAMES), **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    before = model.input_projection.weight.detach().clone()
    losses = []
    # Round robin across panel/condition strata rather than taking four adjacent
    # windows from a single day and claiming balanced model training.
    strata = {}
    for s in fit:
        strata.setdefault((s['panel'], s['condition']), []).append(s)
    chosen = [values[0] for _, values in sorted(strata.items())]
    for step in range(args.steps):
        sample = chosen[step % len(chosen)]
        assert sample['role'] == 'fit'
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(**tensor_input(sample, device))
        loss = masked_queue_loss(result['logits'],
            torch.as_tensor(sample['y'], device=device).unsqueeze(0),
            torch.as_tensor(sample['m'], device=device).unsqueeze(0))['total']
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f'[smoke] optimizer update {step+1}/{args.steps}: loss={losses[-1]:.6f}', flush=True)
    assert not torch.equal(before, model.input_projection.weight)
    model.eval()
    with torch.no_grad():
        p1 = model(**tensor_input(reference, device))['logits']
        p2 = model(**tensor_input(changed, device))['logits']
        # Input tensors match exactly. CUDA sparse scatter may sum neighbors in
        # different orders across invocations/GPUs, so allow float32 roundoff.
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)
        neural = [model(**tensor_input(s, device))['logits'].sigmoid()[0].cpu().numpy() for s in tune]

    # Full 352-link graph, standard 128x4 architecture: memory/shape check only.
    largest_net = network(args.release_root, 'D7_I405_N')
    shape_sample = dict(reference, x=np.zeros((13, len(largest_net['link_ids']), len(FEATURE_NAMES)), np.float32),
                        network=largest_net)
    with torch.no_grad():
        largest = model(**tensor_input(shape_sample, device))['logits']
    assert largest.shape == (1, 6, len(largest_net['link_ids'])) and torch.isfinite(largest).all()

    checkpoint = args.output / 'engineering_checkpoint.pt'
    torch.save({'model': model.state_dict(), 'config': model_cfg, 'input_dim': len(FEATURE_NAMES),
                'protocol_sha256': PROTOCOL_SHA256, 'engineering_only': True,
                'optimizer_updates': args.steps}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded = QueueForecaster(saved['input_dim'], **saved['config']).to(device)
    reloaded.load_state_dict(saved['model'])
    reloaded.eval()
    with torch.no_grad():
        replay = reloaded(**tensor_input(reference, device))['logits']
    torch.testing.assert_close(p1, replay, atol=1e-6, rtol=1e-5)

    from .baselines import official_persistence, persistence, propagation, fit_lightgbm, predict_lightgbm, tree_features
    for predictor in [official_persistence, persistence, propagation, tree_features]:
        np.testing.assert_array_equal(predictor(reference, reference_net), predictor(changed, reference_net))
    nets = {s['panel']: s['network'] for s in samples}
    tree = fit_lightgbm(fit, nets, rounds=12, seed=20261024)
    methods = {
        'official_persistence_Tminus5': [official_persistence(s, s['network']) for s in tune],
        'causal_carry_Tminus5': [persistence(s, s['network'], at_origin=False) for s in tune],
        'causal_carry_T': [persistence(s, s['network'], at_origin=True) for s in tune],
        'propagation': [propagation(s, s['network']) for s in tune],
        'lightgbm_12_rounds': [predict_lightgbm(tree, s, s['network']) for s in tune],
        'graph_transformer_4_updates': neural,
    }
    metrics = {name: _diagnostic(tune, pred) for name, pred in methods.items()}
    atomic_json(args.output / 'engineering_proxy_diagnostics.json', metrics)
    # Check the deployment-facing loader on both published conditions and all
    # three splits, without opening the corresponding unknown future labels.
    official_checks = []
    for panel in panels:
        for split in ['train', 'validation', 'private']:
            base = args.release_root / 'task2' / panel / split
            index = pd.read_csv(base / 'window_index.csv')
            for condition in ['queue_onset', 'queue_ongoing']:
                row = index[index.condition.eq(condition)].iloc[0]
                sample = released_sample(args.release_root, panel, split, row.window_id)
                assert sample['label_kind'] == 'withheld' and not sample['m'].any()
                with torch.no_grad():
                    probability = model(**tensor_input(sample, device))['logits'].sigmoid()[0].cpu().numpy()
                template = pd.read_csv(base / 'sample_submission_queue.csv')
                template = template[template.window_id.eq(row.window_id)].copy()
                positions = {str(link): i for i, link in enumerate(sample['network']['link_ids'])}
                origin = pd.Timestamp(sample['origin'])
                times = pd.to_datetime(template.timestamp, utc=True)
                horizons = ((times - origin).dt.total_seconds() / 300).astype(int).to_numpy()-1
                nodes = template.link_id.astype(str).map(positions).to_numpy()
                if not set(horizons).issubset(set(range(6))) or pd.isna(nodes).any():
                    raise ValueError('Official window template does not align with the six-step output')
                output = template.copy()
                output['queue_pred'] = (probability[horizons, nodes] >= .5).astype(int)
                checked = validate_queue(output, template)
                checked.update(panel=panel, split=split, condition=condition,
                               window_id=str(row.window_id), labels='withheld')
                official_checks.append(checked)
    atomic_json(args.output / 'official_window_schema_checks.json', official_checks)
    atomic_json(args.output / 'runtime.json', {'python': __import__('sys').version,
        'torch': torch.__version__, 'numpy': np.__version__, 'threads': args.threads,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None})
    report = {'passed': True, 'scope': 'engineering_smoke_only', 'formal_training_started': False,
              'official_queue_score': None, 'calibrate_or_audit_labels_opened': False,
              'protocol_sha256': PROTOCOL_SHA256, 'panels': panels,
              'fit_windows': len(fit), 'tune_windows': len(tune),
              'primary_future_positive_windows': manifest['primary_sample_count'],
              'optimizer_updates': args.steps, 'lightgbm_rounds': 12,
              'model_config': model_cfg, 'parameters': sum(p.numel() for p in model.parameters()),
              'largest_graph_links': len(largest_net['link_ids']),
              'cache_payload_reused': True, 'future_perturbation_invariant': True,
              'prediction_comparison_tolerance': {'atol': 1e-6, 'rtol': 1e-5},
              'checkpoint_replay_passed': True, 'losses': losses,
              'official_sample_windows_checked': len(official_checks),
              'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'peak_gpu_mib': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
              'seconds': time.perf_counter()-started,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'warnings': manifest.get('warnings', [])}
    atomic_json(args.output / 'SMOKE_COMPLETE.json', report)
    return report
