"""Train/evaluate gated residuals, called only by an explicit pipeline stage."""
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import CHANNELS, KEYS, METHOD, SCALE, read_json, save_json, sha, score
from crossfit import ensure_day, verify_ready
from dataset import DayStore, graph_tensors
from model import GatedResidual, objective


def move(values, device): return {k:v.to(device) for k, v in values.items()}


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp'); torch.save(value, tmp); tmp.replace(path)


@torch.inference_mode()
def evaluate(model, store, args, dates, baseline_only=False, final=False):
    model.eval()
    nodes = np.arange(store.n)
    graph = move(graph_tensors(store.graph, nodes), args.device)
    frames, predictions, bases, truths, regimes = [], [], [], [], []
    starts = list(range(0, 289-args.window, max(1, args.window//2)))
    if starts[-1] != 288-args.window: starts.append(288-args.window)
    for date in dates:
        if final:
            paths = [args.final_lgb/store.panel/(c+'.txt') for c in CHANNELS]
            ensure_day(args, store.panel, date, paths, 'final_full168', supervised=False)
        print(store.panel, 'evaluate', date, 'baseline' if baseline_only else METHOD, flush=True)
        base = np.load(store.baselines/date/'base.npy', mmap_mode='r')
        if baseline_only:
            reconstruction = base.astype('float64')
        else:
            total = np.zeros_like(base, dtype='float64'); weights = np.zeros((288, 1, 1))
            for offset in range(0, len(starts), args.eval_batch):
                current = starts[offset:offset+args.eval_batch]
                blocks = [store.block(date, tick, nodes) for tick in current]
                batch = move({k:torch.stack([b[k] for b in blocks]) for k in blocks[0]}, args.device)
                result = model(batch, graph)['prediction'].cpu().numpy()
                for j, tick in enumerate(current):
                    total[tick:tick+args.window] += result[j]
                    weights[tick:tick+args.window] += 1
            reconstruction = total/weights
        # All inputs/predictions are built before reading the scoring labels.
        folder = store.source/date
        keys = pd.read_parquet(folder/'keys.parquet')
        ii, jj = np.load(folder/'ii.npy'), np.load(folder/'jj.npy')
        if len(keys) != len(ii): raise ValueError('Target key count mismatch')
        if np.load(folder/'visible.npy', mmap_mode='r')[jj, ii].any(): raise ValueError('Visible target')
        np.testing.assert_array_equal(keys.link_id.to_numpy(), np.array(store.graph['links'])[ii])
        np.testing.assert_array_equal(keys.station_id.to_numpy(), np.array(store.graph['stations'])[ii])
        p, b = reconstruction[jj, ii]*SCALE, base[jj, ii].astype('float64')*SCALE
        iso = store.graph['isolated'][ii]
        np.testing.assert_allclose(p[iso], b[iso], rtol=0, atol=1e-10)
        predictions.append(p); bases.append(b)
        truths.append(np.load(folder/'label.npy', mmap_mode='r')[jj, ii].astype('float64')*SCALE)
        regimes.append(keys.mask_regime.to_numpy()); frames.append(keys)
    output = pd.concat(frames, ignore_index=True)
    pred, base, truth = np.concatenate(predictions), np.concatenate(bases), np.concatenate(truths)
    regime = np.concatenate(regimes)
    if output.duplicated(KEYS).any() or not np.isfinite(pred).all() or (pred < 0).any(): raise ValueError('Invalid prediction coverage/values')
    rows = score(truth, pred, regime)
    baseline_rows = score(truth, base, regime, 'LightGBM_grid_baseline')
    fd = pd.read_csv(Path(store.plan['release_root'])/'corridors'/store.panel/'network/fd_parameters.csv').set_index('link_id')
    lanes = output.link_id.map(fd.lanes).to_numpy()
    output['speed_kmh'], output['flow_vph'] = pred[:, 0], pred[:, 1]*lanes
    output['flow_per_lane'] = pred[:, 1]
    output['lgb_speed_kmh'], output['lgb_flow_per_lane'] = base[:, 0], base[:, 1]
    output['correction_speed_kmh'], output['correction_flow_per_lane'] = pred[:, 0]-base[:, 0], pred[:, 1]-base[:, 1]
    output['graph_isolated'] = output.link_id.isin(np.array(store.graph['links'])[store.graph['isolated']])
    return rows, output, baseline_rows


def build_model(args):
    return GatedResidual(106, args.width, args.heads, args.layers, args.window, args.dropout)


def train(args, plan, panel, progress):
    folder = args.output/panel
    verify_ready(args, plan, panel)
    if (folder/'TRAINED.json').exists():
        trained = read_json(folder/'TRAINED.json')
        if trained['best_sha256'] != sha(folder/'best.pt'): raise ValueError('Selected checkpoint changed')
        print(panel, 'residual training already complete', flush=True)
        return
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback was silently selected')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    store = DayStore(args, plan, panel)
    model = build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*args.steps_per_epoch, eta_min=args.learning_rate*.1)
    identity = dict(plan_sha256=sha(args.output/'plan.json'), oof_sha256=sha(folder/'OOF_READY.json'),
        input_names=store.input_names, feature_mean=store.mean.tolist(), feature_scale=store.scale.tolist(),
        links=store.graph['links'], stations=store.graph['stations'])
    first, best, stale, trace = 0, -float('inf'), 0, []
    def checkpoint(epoch):
        return dict(identity=identity, state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            epoch=epoch, best=best, stale=stale, trace=trace, numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state().cpu(), cuda_rng=[s.cpu() for s in torch.cuda.get_rng_state_all()] if args.device == 'cuda' else None,
            network=dict(inputs=106, width=args.width, heads=args.heads, layers=args.layers, window=args.window, dropout=args.dropout),
            physics=dict(fd_weight=0., conservation_weight=0.))
    if (folder/'last.pt').exists():
        if not args.resume: raise ValueError('Existing residual checkpoint; pass --resume')
        ck = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        if ck['identity'] != identity: raise ValueError('Checkpoint input/config identity changed')
        model.load_state_dict(ck['state_dict']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        # Optimizer.load_state_dict casts moments to parameter devices and keeps
        # non-capturable AdamW step counters on their supported device.
        first, best, stale, trace = ck['epoch'], ck['best'], ck['stale'], ck['trace']
        rng.bit_generator.state = ck['numpy_rng']; torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device == 'cuda' and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
        print(panel, 'resume after full epoch', first, flush=True)
    else:
        # Include the exact zero-residual candidate in model selection.
        rows, prediction, baseline_rows = evaluate(model, store, args, plan['internal_validation_dates'], baseline_only=True)
        best = rows[-1]['S_state']
        trace = [dict(epoch=0, internal_S_state=best, baseline_candidate=True)]
        save_checkpoint(folder/'best.pt', checkpoint(0)); save_checkpoint(folder/'last.pt', checkpoint(0))
        prediction.to_parquet(folder/'internal_predictions.parquet', index=False)
        pd.DataFrame(rows+baseline_rows).to_csv(folder/'internal_metrics.csv', index=False)
    started = time.time()
    for epoch in range(first, args.epochs):
        if stale >= args.patience: break
        model.train(); records = []
        for step in range(args.steps_per_epoch):
            batch, graph = store.sample(rng)
            batch, graph = move(batch, args.device), move(graph, args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, graph)
            loss, terms = objective(output, batch, graph, args.shrink)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            records.append(dict(loss=float(loss.detach()), **terms))
            if step == 0 or (step+1)%32 == 0 or step+1 == args.steps_per_epoch:
                progress(stage='train', epoch=epoch+1, step=step+1)
                print(panel, f'epoch {epoch+1}/{args.epochs}, step {step+1}/{args.steps_per_epoch}, loss={np.mean([r["loss"] for r in records]):.6f}', flush=True)
        progress(stage='internal_validation', epoch=epoch+1, step=None)
        rows, prediction, baseline_rows = evaluate(model, store, args, plan['internal_validation_dates'])
        value = rows[-1]['S_state']; improved = value > best
        stale = 0 if improved else stale+1; best = max(best, value)
        trace.append(dict(epoch=epoch+1, internal_S_state=value, **pd.DataFrame(records).mean().to_dict()))
        if improved:
            save_checkpoint(folder/'best.pt', checkpoint(epoch+1))
            prediction.to_parquet(folder/'internal_predictions.parquet', index=False)
            pd.DataFrame(rows+baseline_rows).to_csv(folder/'internal_metrics.csv', index=False)
        save_checkpoint(folder/'last.pt', checkpoint(epoch+1))
        pd.DataFrame(trace).to_csv(folder/'training_trace.csv', index=False)
        print(panel, f'internal S_state={value:.9f}, best={best:.9f}, stale={stale}', flush=True)
    selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    # Rebuild artifacts from best.pt even if an interruption occurred between checkpoint and table writes.
    model.load_state_dict(selected['state_dict'])
    rows, prediction, baseline_rows = evaluate(model, store, args, plan['internal_validation_dates'])
    prediction.to_parquet(folder/'internal_predictions.parquet', index=False)
    pd.DataFrame(rows+baseline_rows).to_csv(folder/'internal_metrics.csv', index=False)
    pd.DataFrame(trace).to_csv(folder/'training_trace.csv', index=False)
    save_checkpoint(folder/'model.pt', {k:selected[k] for k in ['identity', 'state_dict', 'network', 'physics', 'epoch', 'best']})
    save_json(folder/'TRAINED.json', dict(best_epoch=selected['epoch'], internal_S_state=rows[-1]['S_state'],
        best_sha256=sha(folder/'best.pt'), elapsed_current_session_seconds=time.time()-started,
        normalization='original supervised 168 days only', physics=False))
    del model, optimizer, scheduler, store; gc.collect()
    if args.device == 'cuda': torch.cuda.empty_cache()


def final_evaluate(args, plan, panel):
    torch.set_num_threads(args.threads)
    folder = args.output/panel
    trained = read_json(folder/'TRAINED.json')
    if trained['best_sha256'] != sha(folder/'best.pt'): raise ValueError('Selected model changed')
    store = DayStore(args, plan, panel)
    ck = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    if ck['identity']['plan_sha256'] != sha(args.output/'plan.json'): raise ValueError('Final evaluation config changed')
    model = build_model(args).to(args.device); model.load_state_dict(ck['state_dict'])
    rows, prediction, baseline_rows = evaluate(model, store, args, plan['holdout_dates'], final=True)
    reference_keys = pd.read_parquet(args.reference_run/panel/'holdout_predictions.parquet', columns=KEYS)
    if not prediction[KEYS].astype(str).equals(reference_keys[KEYS].astype(str)): raise ValueError('Holdout template order/coverage changed')
    prediction.to_parquet(folder/'holdout_predictions.parquet', index=False)
    pd.DataFrame(rows+baseline_rows).to_csv(folder/'metrics.csv', index=False)
    save_json(folder/'COMPLETE.json', dict(panel=panel, S_state=rows[-1]['S_state'],
        baseline_S_state=baseline_rows[-1]['S_state'], targets=len(prediction), best_epoch=ck['epoch'],
        best_sha256=sha(folder/'best.pt'), prediction_sha256=sha(folder/'holdout_predictions.parquet'),
        metrics_sha256=sha(folder/'metrics.csv'), training_physics=False, method=METHOD,
        note='Previously inspected local holdout, not an official score or full Kaggle submission.'))
    del model, store; gc.collect()
    if args.device == 'cuda': torch.cuda.empty_cache()
