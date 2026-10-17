from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .data import HistoricalProfile, eligible, read_csv


def neighbor_map(topology: pd.DataFrame, observed_links: set[str]) -> dict:
    """Find the nearest instrumented mainline links through unobserved links.

    Connector and ramp IDs are excluded by membership in the mainline topology.
    Multiple branches at the nearest depth are retained, with cycle protection.
    """
    id_col = 'mainline_link_id' if 'mainline_link_id' in topology else 'link_id'
    allowed = set(topology[id_col].astype(str))
    result = {}
    for direction, col in [('up', 'incoming_link_ids'), ('down', 'outgoing_link_ids')]:
        graph = {}
        for _, row in topology.iterrows():
            raw = row.get(col)
            graph[str(row[id_col])] = ([] if pd.isna(raw) else
                                       [x for x in str(raw).split(';') if x in allowed])
        for origin in observed_links:
            queue = deque((n, 1) for n in graph.get(origin, []))
            seen, found, depth_found = {origin}, [], None
            while queue:
                node, depth = queue.popleft()
                if node in seen or (depth_found is not None and depth > depth_found):
                    continue
                seen.add(node)
                if node in observed_links:
                    found.append(node)
                    depth_found = depth
                elif depth < 20:
                    queue.extend((n, depth + 1) for n in graph.get(node, []))
            result[(origin, direction)] = sorted(found)
    return result


def temporal_context(values: np.ndarray, max_gap_minutes: int = 60):
    """Return strictly previous/next observations and an interpolation candidate.

    Input is a 288 x station matrix. Current-cell values are never features.
    No interpolation is allowed across the release's long all-corridor gaps.
    """
    f = pd.DataFrame(values)
    n = len(f)
    ticks = np.broadcast_to(np.arange(n)[:, None], values.shape).astype(float)
    obs_ticks = np.where(np.isfinite(values), ticks, np.nan)
    prev = f.shift(1).ffill().to_numpy()
    nxt = f.shift(-1).bfill().to_numpy()
    prev_t = pd.DataFrame(obs_ticks).shift(1).ffill().to_numpy()
    next_t = pd.DataFrame(obs_ticks).shift(-1).bfill().to_numpy()
    before, after = (ticks - prev_t) * 5, (next_t - ticks) * 5
    span = before + after
    both = np.isfinite(prev) & np.isfinite(nxt) & (span <= max_gap_minutes)
    init = np.full(values.shape, np.nan)
    np.divide(prev * after + nxt * before, span, out=init, where=both)
    # A short one-sided extrapolation is a fallback, never an unlimited fill.
    init = np.where(~np.isfinite(init) & (before <= 15), prev, init)
    init = np.where(~np.isfinite(init) & (after <= 15), nxt, init)
    return prev, nxt, before, after, init


def _build_base_features(frame: pd.DataFrame, profile: HistoricalProfile,
                   net: pd.DataFrame, topology: pd.DataFrame,
                   max_gap_minutes: int = 60):
    f = frame.reset_index(drop=True)
    if f.duplicated(['station_id', 'slot']).any():
        raise ValueError('A day must have at most one record per station/slot')
    hist = profile.lookup(f)
    lanes = f.link_id.map(net.lanes).to_numpy(dtype=float)
    if not np.isfinite(lanes).all() or (lanes <= 0).any() or (lanes > 16).any():
        raise ValueError('Missing or invalid mainline lane counts')
    history = hist.copy()
    history[:, 1] /= lanes
    x = pd.DataFrame(index=f.index)
    x['slot'] = f.slot.astype(float)
    x['weekday'] = f.weekday.astype(float)
    x['time_sin'] = np.sin(2 * np.pi * f.slot / 288)
    x['time_cos'] = np.cos(2 * np.pi * f.slot / 288)
    for col in ['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph']:
        x[col] = f.link_id.map(net[col]).to_numpy(dtype=float)
    x['capacity_per_lane'] = x.capacity_vph / lanes
    x['pct_observed'] = pd.to_numeric(f.pct_observed, errors='coerce')
    stations = sorted(f.station_id.unique())
    s_index = {s: i for i, s in enumerate(stations)}
    rr = f.slot.to_numpy(dtype=int)
    cc = f.station_id.map(s_index).to_numpy(dtype=int)
    valid = eligible(f)
    neighbor_links = neighbor_map(topology, set(f.link_id))
    initial = np.empty((len(f), 2))
    for ch, col in enumerate(['speed_kmh', 'flow_vph']):
        clean = f[['station_id', 'slot', 'link_id']].copy()
        raw = f[col].to_numpy(dtype=float)
        if ch == 1:
            raw = raw / lanes
        clean['value'] = np.where(valid, raw, np.nan)
        matrix = clean.pivot(index='slot', columns='station_id', values='value')
        matrix = matrix.reindex(index=range(288), columns=stations).to_numpy()
        prev, nxt, before, after, interp = temporal_context(matrix, max_gap_minutes)
        prefix = 'speed' if ch == 0 else 'flow_lane'
        x[prefix + '_history'] = history[:, ch]
        x[prefix + '_prev'] = prev[rr, cc]
        x[prefix + '_next'] = nxt[rr, cc]
        x[prefix + '_minutes_before'] = before[rr, cc]
        x[prefix + '_minutes_after'] = after[rr, cc]
        interp_rows = interp[rr, cc]
        initial[:, ch] = np.where(np.isfinite(interp_rows), interp_rows, history[:, ch])
        x[prefix + '_initial'] = initial[:, ch]
        x[prefix + '_prev_history_diff'] = prev[rr, cc] - history[:, ch]
        x[prefix + '_next_history_diff'] = nxt[rr, cc] - history[:, ch]
        for offset in [-6, -3, -1, 1, 3, 6]:
            shifted = pd.DataFrame(matrix).shift(-offset).to_numpy()
            x[f'{prefix}_offset_{offset}'] = shifted[rr, cc]
        link_matrix = clean.groupby(['slot', 'link_id']).value.mean().unstack('link_id')
        link_matrix = link_matrix.reindex(index=range(288))
        for direction in ['up', 'down']:
            context = pd.DataFrame(index=range(288), columns=link_matrix.columns, dtype=float)
            for link in context.columns:
                neighbors = neighbor_links.get((link, direction), [])
                if neighbors:
                    context[link] = link_matrix[neighbors].mean(axis=1)
            for offset in [0, -3, 3]:
                shifted = context.shift(-offset)
                link_columns = {link: i for i, link in enumerate(shifted.columns)}
                link_positions = f.link_id.map(link_columns).to_numpy(dtype=int)
                x[f'{prefix}_{direction}_{offset}'] = shifted.to_numpy()[rr, link_positions]
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return x, initial, history, lanes


VARIANTS = ('baseline', 'multihop', 'congestion')


def feature_columns(columns, variant):
    """The two additions are mutually exclusive; baseline stays at 50 columns."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown feature variant: {variant}')
    return [c for c in columns if (
        not c.startswith(('mh_', 'cg_'))
        or (variant == 'multihop' and c.startswith('mh_'))
        or (variant == 'congestion' and c.startswith('cg_')))]


def multihop_neighbors(topology, observed_links, net, max_hops=3):
    """Rank by number of observed mainline links reached, then shortest distance.

    An unobserved link adds physical distance but does not consume a sensor hop.
    Shortest paths use link-center distances; branches are retained and cycles
    cannot revisit the origin. Connector/ramp nodes are outside the graph.
    """
    import heapq
    id_col = 'mainline_link_id' if 'mainline_link_id' in topology else 'link_id'
    allowed = set(topology[id_col].astype(str))
    lengths = net.length_km.to_dict()
    result = {}
    for direction, col in [('up', 'incoming_link_ids'), ('down', 'outgoing_link_ids')]:
        graph = {}
        for _, row in topology.iterrows():
            raw = row.get(col)
            graph[str(row[id_col])] = [] if pd.isna(raw) else [
                v for v in str(raw).split(';') if v in allowed]
        for origin in sorted(observed_links):
            queue = [(0, 0., origin, 0)]
            best = {}
            while queue:
                hops, distance, node, steps = heapq.heappop(queue)
                if node in best:
                    continue
                best[node] = (hops, distance)
                if node != origin and node in observed_links:
                    result.setdefault((origin, direction, hops), []).append((node, distance))
                if hops >= max_hops or steps >= 20:
                    continue
                for neighbor in graph.get(node, []):
                    if neighbor in best:
                        continue
                    step = (float(lengths[node]) + float(lengths[neighbor])) / 2
                    if not np.isfinite(step) or step <= 0:
                        raise ValueError('Invalid mainline length for spatial distance')
                    heapq.heappush(queue, (hops + int(neighbor in observed_links),
                                          distance + step, neighbor, steps + 1))
    return result


def multihop_features(frame, net, topology, lanes):
    f = frame.reset_index(drop=True)
    valid = eligible(f)
    neighbors = multihop_neighbors(topology, set(f.link_id), net)
    rr = f.slot.to_numpy(dtype=int)
    columns = {}
    for ch, field in [('speed', 'speed_kmh'), ('flow_lane', 'flow_vph')]:
        clean = f[['slot', 'link_id']].copy()
        raw = f[field].to_numpy(dtype=float)
        if ch == 'flow_lane':
            raw = raw / lanes
        clean['value'] = np.where(valid, raw, np.nan)
        table = clean.groupby(['slot', 'link_id']).value.mean().unstack('link_id').reindex(range(288))
        link_positions = f.link_id.map({v: i for i, v in enumerate(table.columns)}).to_numpy(dtype=int)
        for direction in ['up', 'down']:
            for hop in [2, 3]:
                context = np.full(table.shape, np.nan)
                coverage = np.zeros(table.shape)
                distances = {}
                for j, link in enumerate(table.columns):
                    pairs = neighbors.get((link, direction, hop), [])
                    if not pairs:
                        distances[link] = np.nan
                        continue
                    ids, dist = zip(*pairs)
                    values = table[list(ids)].to_numpy()
                    weights = 1 / np.maximum(np.asarray(dist), 0.001)
                    denominator = np.sum(np.isfinite(values) * weights, axis=1)
                    numerator = np.nansum(values * weights, axis=1)
                    np.divide(numerator, denominator, out=context[:, j], where=denominator > 0)
                    coverage[:, j] = np.isfinite(values).mean(axis=1)
                    distances[link] = float(np.mean(dist))
                for offset in [0, -3, 3]:
                    shifted = pd.DataFrame(context).shift(-offset).to_numpy()
                    columns[f'mh_{ch}_{direction}_h{hop}_t{offset}'] = shifted[rr, link_positions]
                columns[f'mh_{ch}_{direction}_h{hop}_coverage'] = coverage[rr, link_positions]
                if ch == 'speed':
                    columns[f'mh_{direction}_h{hop}_distance_km'] = f.link_id.map(distances).to_numpy()
    return pd.DataFrame(columns, index=f.index)


def congestion_features(x):
    """Only masked-input features and their interactions; no truth queue labels."""
    free = x.free_speed_kmh.replace(0, np.nan)
    capacity = x.capacity_per_lane.replace(0, np.nan)
    c = {}
    for name in ['initial', 'prev', 'next', 'history']:
        c[f'cg_speed_{name}_free_ratio'] = x[f'speed_{name}'] / free
        c[f'cg_flow_{name}_capacity_ratio'] = x[f'flow_lane_{name}'] / capacity
    for ch in ['speed', 'flow_lane']:
        span = x[f'{ch}_minutes_before'] + x[f'{ch}_minutes_after']
        c[f'cg_{ch}_bracket_slope_per_min'] = (x[f'{ch}_next'] - x[f'{ch}_prev']) / span
        c[f'cg_{ch}_past_slope_per_min'] = (x[f'{ch}_offset_-3'] - x[f'{ch}_offset_-6']) / 15
        c[f'cg_{ch}_future_slope_per_min'] = (x[f'{ch}_offset_6'] - x[f'{ch}_offset_3']) / 15
        scale = free if ch == 'speed' else capacity
        c[f'cg_{ch}_up_down_difference'] = (x[f'{ch}_up_0'] - x[f'{ch}_down_0']) / scale
        for direction in ['up', 'down']:
            c[f'cg_{ch}_{direction}_change_per_min'] = (
                x[f'{ch}_{direction}_3'] - x[f'{ch}_{direction}_-3']) / (30 * scale)
    c['cg_speed_bracket_range_ratio'] = abs(x.speed_next - x.speed_prev) / free
    c['cg_initial_minus_up_speed_ratio'] = (x.speed_initial - x.speed_up_0) / free
    c['cg_initial_minus_down_speed_ratio'] = (x.speed_initial - x.speed_down_0) / free
    # These are margins from observable proxies, not official queue indicators.
    c['cg_prev_queue_margin'] = x.speed_prev / free - 0.60
    c['cg_next_queue_margin'] = x.speed_next / free - 0.60
    return pd.DataFrame(c, index=x.index)


def build_features(frame, profile, net, topology, max_gap_minutes=60, feature_set='base'):
    x, initial, history, lanes = _build_base_features(frame, profile, net, topology, max_gap_minutes)
    if feature_set == 'all':
        additions = [multihop_features(frame, net, topology, lanes), congestion_features(x)]
        x = pd.concat([x, *additions], axis=1).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    elif feature_set != 'base':
        raise ValueError('feature_set must be base or all')
    return x, initial, history, lanes
