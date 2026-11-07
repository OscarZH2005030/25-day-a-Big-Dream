"""Bounded official Time-LLM/GPT2 pilot on public June traffic observations."""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
import hashlib
import argparse
import json
import msvcrt
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get('TFB_WORKSPACE_ROOT', Path(__file__).resolve().parents[2]))
UPSTREAM = Path(os.environ.get('TIME_LLM_UPSTREAM', ROOT / 'work/daytime/time_llm_upstream'))
OUT = ROOT / 'outputs/daytime/time_llm_pilot'
DATA = Path(os.environ.get('TFB_DATA_ROOT', 'C:/Users/Lenovo/Downloads/2026-ieee-big-data-traffic-flow-bench/kaggle_public'))
SEED = 2026110701


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--road', default='D7_I10_E')
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    files = sorted((DATA / f'corridors/{args.road}/train/mainline_states/year_month=2030-06').glob('*.parquet'))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    frame.loc[frame.is_score_eligible != 1, ['speed_kmh', 'flow_vph']] = np.nan
    stations = frame.loc[frame.timestamp.dt.day <= 20].groupby('station_id').speed_kmh.count().nlargest(4).index.tolist()
    times = pd.date_range('2030-06-01', '2030-07-01', freq='5min', inclusive='left', tz='UTC')
    arrays = []
    labels = []
    for station in stations:
        selected = frame.loc[frame.station_id == station].set_index('timestamp').reindex(times)
        for column, scale in [('speed_kmh', 100.0), ('flow_vph', 1000.0)]:
            arrays.append(selected[column].to_numpy(dtype=np.float32) / scale)
            labels.append((station, column, scale))
    values = np.stack(arrays, axis=1)
    train_means = np.nanmean(values[:20 * 288], axis=0)
    input_values = pd.DataFrame(values).ffill(limit=12).fillna(dict(enumerate(train_means))).to_numpy(dtype=np.float32)
    splits = {'train': (96, 20 * 288), 'validation': (20 * 288, 24 * 288), 'test': (24 * 288, 30 * 288)}
    windows = {}
    for split, (start, end) in splits.items():
        keys = []
        for tick in range(start, end - 11, 12):
            for channel in range(values.shape[1]):
                if (np.isfinite(values[tick:tick + 12, channel]).all()
                        and np.isfinite(values[tick - 96:tick, channel]).mean() >= 0.75):
                    keys.append((tick, channel))
        windows[split] = keys
    config = SimpleNamespace(task_name='short_term_forecast', pred_len=12, seq_len=96,
        d_ff=32, llm_dim=768, patch_len=16, stride=8, llm_model='GPT2', llm_layers=args.layers,
        prompt_domain=True, content='Freeway station traffic measurements at five-minute intervals, including speed or vehicle flow.',
        dropout=0.1, d_model=32, n_heads=4, enc_in=1)
    protocol = {'seed': SEED, 'road': args.road, 'source_month': '2030-06',
        'splits': splits, 'windows': {key: len(value) for key, value in windows.items()},
        'channels': labels, 'config': vars(config), 'steps': args.steps, 'batch_size': 8,
        'learning_rate': args.learning_rate,
        'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_hashes': {path.name: digest(path) for path in files},
        'runner_sha256': digest(Path(__file__)), 'scope': 'Exposed public-data forecasting diagnostic, not official imputation or hidden score.',
        'selection': 'Fixed config, best validation scaled MSE at 50-step checkpoints. Test evaluated once per run; exposed after first pilot.',
        'eligibility': 'Targets fully observed and eligible. Inputs at least 75% observed, causal forward-fill up to 12 ticks then training-only channel mean. Stations selected on train availability only.',
        'train_channel_means': train_means.tolist()}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'window_keys.npz', **{key: np.array(value) for key, value in windows.items()})
    print('PROTOCOL', protocol['windows'], flush=True)
    if any(len(keys) < 8 for keys in windows.values()):
        raise RuntimeError('Insufficient fully observed windows')
    sys.path.insert(0, str(UPSTREAM))
    from models.TimeLLM import Model
    model = Model(config).cuda()
    model.llm_model.config.use_cache = False
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate)
    rng = np.random.default_rng(SEED)

    def batch(keys):
        x = np.stack([input_values[tick - 96:tick, channel] for tick, channel in keys])
        y = np.stack([values[tick:tick + 12, channel] for tick, channel in keys])
        return torch.from_numpy(x[:, :, None]).cuda(), torch.from_numpy(y[:, :, None]).cuda()

    def evaluate(keys):
        model.eval()
        predictions = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for index in range(0, len(keys), 8):
                x, _ = batch(keys[index:index + 8])
                predictions.append(model(x, None, None, None).float().cpu().numpy()[:, :, 0])
        return np.concatenate(predictions)

    # A fixed bounded subset keeps this a deadline-feasible pilot.
    validation_keys = windows['validation'][::max(1, len(windows['validation']) // 64)][:64]
    test_keys = windows['test'][::max(1, len(windows['test']) // 128)][:128]
    (OUT / 'evaluation_keys.json').write_text(json.dumps({'validation': validation_keys, 'test': test_keys}), encoding='utf-8')
    best = float('inf')
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        model.llm_model.eval()
        keys = [windows['train'][index] for index in rng.integers(len(windows['train']), size=8)]
        x, y = batch(keys)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(x, None, None, None)
            loss = (prediction.float() - y).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 10 == 0:
            print(json.dumps({'step': step, 'train_scaled_mse': loss.item(), 'seconds': time.monotonic() - started}), flush=True)
        if step % 50 == 0:
            predicted = evaluate(validation_keys)
            truth = np.stack([values[tick:tick + 12, channel] for tick, channel in validation_keys])
            score = float(np.mean((predicted - truth) ** 2))
            print('VALIDATION', step, score, flush=True)
            if score < best:
                best = score
                torch.save({name: value.cpu() for name, value in model.state_dict().items()
                            if not name.startswith('llm_model.') and name != 'word_embeddings'}, OUT / 'adapter.pt')
                (OUT / 'best.json').write_text(json.dumps({'step': step, 'validation_scaled_mse': score}))
    model.load_state_dict(torch.load(OUT / 'adapter.pt', weights_only=True), strict=False)
    prediction = evaluate(test_keys)
    if not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite prediction')
    rows = []
    for index, (tick, channel) in enumerate(test_keys):
        station, variable, scale = labels[channel]
        truth = values[tick:tick + 12, channel] * scale
        persistence = np.repeat(input_values[tick - 1, channel] * scale, 12)
        daily = values[tick - 288:tick - 288 + 12, channel] * scale
        for method, predicted in [('time_llm', prediction[index] * scale), ('persistence', persistence), ('previous_day', daily)]:
            valid = np.isfinite(predicted)
            # Common support across all three methods for fair comparisons.
            common = valid & np.isfinite(daily)
            for horizon in np.flatnonzero(common):
                rows.append({'station': station, 'variable': variable, 'method': method,
                             'squared_error': float((predicted[horizon] - truth[horizon]) ** 2),
                             'absolute_error': float(abs(predicted[horizon] - truth[horizon]))})
    errors = pd.DataFrame(rows)
    metrics = errors.groupby(['variable', 'method']).agg(mse=('squared_error', 'mean'), mae=('absolute_error', 'mean'), n=('squared_error', 'size')).reset_index()
    metrics['rmse'] = np.sqrt(metrics.mse)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    errors.groupby(['station', 'variable', 'method']).agg(mse=('squared_error', 'mean'), mae=('absolute_error', 'mean'), n=('squared_error', 'size')).to_csv(OUT / 'per_station.csv')
    (OUT / 'acceptance.json').write_text(json.dumps({'completed': True, 'seconds': time.monotonic() - started,
        'adapter_sha256': digest(OUT / 'adapter.pt'), 'torch': torch.__version__,
        'model_revision': getattr(model.llm_model.config, '_commit_hash', None),
        'official_score': None, 'test_windows': len(test_keys)}, indent=2))
    (OUT / 'gpu_memory.json').write_text(json.dumps({
        'device': torch.cuda.get_device_name(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved()}, indent=2))
    cache = Path(os.environ['HF_HOME']) / 'hub/models--openai-community--gpt2'
    provenance = {'upstream_model_sha256': digest(UPSTREAM / 'models/TimeLLM.py'),
        'model_files': {str(path.relative_to(cache)): digest(path) for path in cache.rglob('*')
                        if path.is_file() and path.suffix in {'.safetensors', '.json', '.txt'}},
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    (OUT / 'model_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(metrics.to_string(index=False), flush=True)


if __name__ == '__main__':
    with (ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            main()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
