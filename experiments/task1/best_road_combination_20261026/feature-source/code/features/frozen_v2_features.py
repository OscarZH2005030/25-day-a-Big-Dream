import json
from pathlib import Path
import numpy as np
SCHEMA=json.loads((Path(__file__).resolve().parents[2]/'base_feature_schema.json').read_text(encoding='utf-8'))

def bank(matrix):
    values = {}
    positions = {name: index for index, name in enumerate(SCHEMA['all_numeric_features'])}
    def get(name):
        return np.asarray(matrix[:, positions[name]], dtype='float32')
    def put(family, name, value):
        value = np.asarray(value, dtype='float32')
        value = np.where(np.isfinite(value), np.clip(value, -1e6, 1e6), np.nan)
        values.setdefault(family, {})['night_'+name] = value
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for channel, scale in [('speed_kmh', 25.), ('flow_vph', 600.)]:
            initial = get(channel+'_initial')
            left = get(channel+'_prev')
            right = get(channel+'_next')
            before = get(channel+'_offset_-5min')
            after = get(channel+'_offset_5min')
            past = get(channel+'_offset_-15min')
            future = get(channel+'_offset_15min')
            gap_left = np.maximum(get(channel+'_prev_gap_min'), 0)
            gap_right = np.maximum(get(channel+'_next_gap_min'), 0)
            span = gap_left+gap_right+1
            prefix = channel+'_'
            put('temporal_robust', prefix+'slope_ratio', (after-before)/(np.abs(future-past)+scale*.05))
            put('temporal_robust', prefix+'centered_bend', (after+before-2*initial)/(np.abs(after-before)+scale*.05))
            put('temporal_robust', prefix+'acceleration_asymmetry', ((after-initial)-(initial-before))/(np.abs(after-initial)+np.abs(initial-before)+scale*.05))
            put('temporal_robust', prefix+'near_far_disagreement', ((after-before)-(future-past)/3)/scale)
            put('temporal_robust', prefix+'history_logratio', np.log((np.maximum(initial, 0)+1)/(np.maximum(get(channel+'_history'), 0)+1)))
            put('gap_reliability', prefix+'weighted_endpoint_delta', ((left-initial)*(gap_right+1)+(right-initial)*(gap_left+1))/((span+1)*scale))
            put('gap_reliability', prefix+'endpoint_slope', (right-left)/((span+1)*scale))
            put('gap_reliability', prefix+'endpoint_disagreement', np.abs(right-left)/((np.sqrt(span)+1)*scale))
            put('gap_reliability', prefix+'near_slope_confidence', (after-before)/(scale*(1+span/10)))
            put('gap_reliability', prefix+'gap_balance_log', np.log1p(gap_left)-np.log1p(gap_right))
        speed = np.maximum(get('speed_kmh_initial'), 0)
        lanes = np.maximum(get('lanes'), 1)
        flow = np.maximum(get('flow_vph_initial'), 0)/lanes
        free = np.maximum(get('free_speed_kmh'), 1)
        capacity = np.maximum(get('capacity_vph')/lanes, 1)
        occupancy = (get('occupancy_offset_-5min')+get('occupancy_offset_5min'))/2
        congestion = np.clip(1-speed/free, 0, 1)
        density = flow/(speed+5)
        quality = np.clip(get('pct_observed')/100, 0, 1)
        put('traffic_physics', 'log_density', np.log1p(density))
        put('traffic_physics', 'density_squared', density*density/10000)
        put('traffic_physics', 'utilization_speed_balance', flow/capacity-speed/free)
        put('traffic_physics', 'congested_occupancy', occupancy*congestion)
        put('traffic_physics', 'occupancy_density_ratio', occupancy/(density+1))
        put('traffic_physics', 'demand_headroom', (capacity-flow)/(speed+5))
        for prefix in ['speed', 'flow_per_lane']:
            lap = get('eng_'+prefix+'_spatial_laplacian')
            grad = get('eng_'+prefix+'_spatial_gradient')
            spread = np.abs(get('eng_'+prefix+'_ensemble_spread'))
            put('spatial_nonlinear', prefix+'_signed_log_laplacian', np.sign(lap)*np.log1p(np.abs(lap)))
            put('spatial_nonlinear', prefix+'_laplacian_signal_noise', lap/(spread+.05))
            put('spatial_nonlinear', prefix+'_gradient_balance', grad/(np.abs(lap)+.05))
            put('spatial_nonlinear', prefix+'_congested_gradient', grad*congestion*congestion)
            put('spatial_nonlinear', prefix+'_quality_laplacian', lap*quality*quality)
        weekday = get('weekday')
        put('calendar_quality', 'weekend_congestion', (weekday>=5)*congestion)
        put('calendar_quality', 'slot_sin_congestion', get('slot_sin')*congestion)
        put('calendar_quality', 'slot_cos_demand', get('slot_cos')*flow/capacity)
        put('calendar_quality', 'quality_density', quality*np.log1p(density))
        put('calendar_quality', 'imputation_density', get('is_imputed')*np.log1p(density))
        put('calendar_quality', 'quality_change', get('pct_observed_offset_5min')-get('pct_observed_offset_-5min'))
        for channel, scale in [('speed_kmh', 25.), ('flow_vph', 600.)]:
            offsets = np.column_stack([get(channel+'_offset_'+str(offset)+'min') for offset in [-30, -15, -5, 5, 15, 30]])
            center = get(channel+'_initial')
            median = np.nanmedian(offsets, axis=1)
            lower = np.nanpercentile(offsets, 25, axis=1)
            upper = np.nanpercentile(offsets, 75, axis=1)
            deviation = np.nanmedian(np.abs(offsets-median[:, None]), axis=1)
            put('multiscale_robust', channel+'_median_residual', (median-center)/scale)
            put('multiscale_robust', channel+'_iqr', (upper-lower)/scale)
            put('multiscale_robust', channel+'_mad', deviation/scale)
            put('multiscale_robust', channel+'_robust_z', (center-median)/(deviation+scale*.05))
            put('multiscale_robust', channel+'_lower_tail', (lower-center)/scale)
            put('multiscale_robust', channel+'_upper_tail', (upper-center)/scale)
            valid_fraction = np.isfinite(offsets).mean(axis=1)
            put('quality_conditioned', channel+'_robust_support', valid_fraction*(median-center)/scale)
            put('quality_conditioned', channel+'_quality_bend', quality*(get(channel+'_offset_5min')+get(channel+'_offset_-5min')-2*center)/scale)
            put('quality_conditioned', channel+'_sparse_history', (1-quality)*(get(channel+'_history')-center)/scale)
            put('quality_conditioned', channel+'_quality_trend', quality*(get(channel+'_offset_15min')-get(channel+'_offset_-15min'))/scale)
            left_gap = np.maximum(get(channel+'_prev_gap_min'), 0)
            right_gap = np.maximum(get(channel+'_next_gap_min'), 0)
            left_weight = 1/(1+left_gap)
            right_weight = 1/(1+right_gap)
            endpoint = (left_weight*get(channel+'_prev')+right_weight*get(channel+'_next'))/(left_weight+right_weight)
            put('endpoint_physics', channel+'_inverse_gap_delta', (endpoint-center)/scale)
            put('endpoint_physics', channel+'_inverse_gap_congestion', (endpoint-center)*congestion/scale)
            put('endpoint_physics', channel+'_endpoint_history_disagreement', (endpoint-get(channel+'_history'))/scale)
            put('endpoint_physics', channel+'_gap_reliability_density', np.log1p(density)*(left_weight+right_weight)/2)
        put('bounded_interactions', 'bounded_density_congestion', density/(density+20)*congestion)
        put('bounded_interactions', 'bounded_capacity_pressure', np.tanh(flow/capacity)*congestion)
        put('bounded_interactions', 'occupancy_utilization_balance', occupancy/(np.abs(occupancy)+1)-flow/(flow+capacity))
        put('bounded_interactions', 'geometry_time_congestion', get('length_km')/(speed+5)*congestion)
        put('bounded_interactions', 'speed_occupancy_mismatch', speed/free*occupancy)
        put('bounded_interactions', 'quality_capacity_pressure', quality*flow/(flow+capacity))
    return values

def transform(matrix, spec, channel=None):
    keep = [index for index in range(328) if index not in spec['removed']]
    base = np.asarray(matrix[:, keep], dtype='float32', order='C')
    masked = sorted(set(spec['masked']) | set(spec.get('channel_masked', {}).get(channel, [])))
    if masked:
        base = base.copy()
        for index in masked:
            if index in keep:
                base[:, keep.index(index)] = 0
    names = [SCHEMA['input_names'][index] for index in keep]
    if spec['added']:
        generated = bank(matrix)
        extra = {name: value for family in sorted(spec['added']) for name, value in generated[family].items()}
        numeric = np.column_stack(list(extra.values()))
        base = np.concatenate([base, numeric, np.isfinite(numeric).astype('float32')], axis=1)
        names += list(extra)+[name+'__finite' for name in extra]
    assert len(names) == base.shape[1] and len(names) == len(set(names))
    assert not np.isinf(base).any()
    return base, names
