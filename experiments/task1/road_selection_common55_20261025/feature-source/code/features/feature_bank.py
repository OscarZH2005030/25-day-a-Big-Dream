"""Feature families derived exclusively from the existing masked-input cache."""
import numpy as np


def generate(raw,names):
    index={name:i for i,name in enumerate(names)}
    def c(name):return raw[...,index[name]]
    v=c('speed_kmh_initial');q=c('flow_vph_initial')
    vf=c('free_speed_kmh');lanes=c('lanes');cap=c('capacity_vph')
    ratio=v/np.maximum(vf,1);util=q*lanes/np.maximum(cap,1)
    gp=c('speed_kmh_prev_gap_min');gn=c('speed_kmh_next_gap_min')
    gap=gp+gn;confidence=1/(1+gap/30.)
    quality=c('pct_observed')/100.
    congestion=np.maximum(.6-ratio,0)
    density=q/np.maximum(v,5)/100.
    data={};groups={}
    def add(group,name,value):
        name='eng_'+name;data[name]=value;groups.setdefault(group,[]).append(name)
    for channel,scale in [('speed_kmh',25.),('flow_vph',600.)]:
        left=c(channel+'_offset_-5min');right=c(channel+'_offset_5min')
        center=c(channel+'_initial')
        add('dynamics',channel+'_slope10',(right-left)/scale)
        add('dynamics',channel+'_slope30',(c(channel+'_offset_15min')-c(channel+'_offset_-15min'))/scale)
        add('dynamics',channel+'_curvature',(right+left-2*center)/scale)
        add('dynamics',channel+'_history_deviation',(center-c(channel+'_history'))/scale)
    add('dynamics','occupancy_change',c('occupancy_offset_5min')-c('occupancy_offset_-5min'))
    add('reliability','gap_hours',gap/60.)
    add('reliability','gap_asymmetry',(gn-gp)/(gap+5))
    add('reliability','short_gap_confidence',confidence)
    add('reliability','quality_confidence',confidence*quality)
    add('reliability','gap_imputed',gap/60.*c('is_imputed'))
    for channel in ['speed','flow_per_lane']:
        left=c(f'new_left_{channel}_residual');right=c(f'new_right_{channel}_residual')
        up=c(f'new_upstream_{channel}_residual');down=c(f'new_downstream_{channel}_residual')
        add('gradients',channel+'_spatial_laplacian',(left+right)/2)
        add('gradients',channel+'_spatial_gradient',right-left)
        add('gradients',channel+'_topology_gradient',down-up)
        add('gradients',channel+'_geometry_topology_disagreement',(left+right-up-down)/2)
        add('fusion',channel+'_confident_spatial',(left+right)*confidence/2)
        add('fusion',channel+'_congested_topology',(up+down)*congestion)
        add('fusion',channel+'_boundary_strength',np.abs(up-down)*np.maximum(1-ratio,0))
    add('physics','density_proxy',density)
    add('physics','congestion_capacity',congestion*util)
    add('physics','density_gap',density*gap/60.)
    add('physics','flow_density_speed',density*ratio)
    add('physics','capacity_margin_congestion',np.maximum(util-.7,0)*np.maximum(1-ratio,0))
    add('fusion','spatial_availability',c('new_left_speed_available')+c('new_right_speed_available'))
    add('fusion','topology_availability',c('new_upstream_speed_available')+c('new_downstream_speed_available'))
    add('fusion','quality_congestion',quality*congestion)
    # These operators require a full daily grid. No wraparound across midnight.
    if raw.shape[0]!=288:raise ValueError('Feature bank requires full-day context')
    def shift(value,offset):
        result=np.full_like(value,np.nan)
        if offset>0:result[:-offset]=value[offset:]
        else:result[-offset:]=value[:offset]
        return result
    for channel,scale in [('speed',25.),('flow_per_lane',600.)]:
        own=c('new_bounded_temporal_'+channel)
        sources=[own]
        neighbor={}
        for direction in ['upstream','downstream']:
            neighbor[direction]=np.where(c(f'new_{direction}_{channel}_available')>0,
                                          c(f'new_{direction}_{channel}_residual')+own,np.nan)
            sources.append(neighbor[direction])
            for offset in [-6,-3,-1,1,3,6]:
                aligned=shift(neighbor[direction],offset)
                add('propagation',f'{direction}_{channel}_shift{offset*5}_contrast',aligned-own)
                if abs(offset)==3:sources.append(aligned)
        stack=np.stack(sources,axis=-1)
        # A median candidate resists one noisy detector or one sharp interpolation.
        valid=np.isfinite(stack);count=valid.sum(axis=-1)
        sorted_values=np.sort(np.where(valid,stack,np.inf),axis=-1)
        midpoint=np.maximum((count-1)//2,0)
        median=np.take_along_axis(sorted_values,midpoint[...,None],axis=-1)[...,0]
        median=np.where(count>0,median,np.nan)
        spread=np.nanmax(np.where(valid,stack,-np.inf),axis=-1)-np.nanmin(np.where(valid,stack,np.inf),axis=-1)
        spread=np.where(count>0,spread,np.nan)
        add('robust_estimators',channel+'_median_contrast',median-own)
        add('robust_estimators',channel+'_ensemble_spread',spread)
        add('robust_estimators',channel+'_median_confidence',(median-own)*confidence)
        legacy='speed_kmh' if channel=='speed' else 'flow_vph'
        previous=c(legacy+'_prev');following=c(legacy+'_next')
        weight=gp/np.maximum(gap,1)
        log_estimate=np.exp((1-weight)*np.log(np.maximum(previous,.01))+weight*np.log(np.maximum(following,.01)))/scale
        add('robust_estimators',channel+'_geometric_contrast',log_estimate-own)
    own_v=c('new_bounded_temporal_speed');own_q=c('new_bounded_temporal_flow_per_lane')
    up_v=np.where(c('new_upstream_speed_available')>0,c('new_upstream_speed_residual')+own_v,np.nan)*25
    down_v=np.where(c('new_downstream_speed_available')>0,c('new_downstream_speed_residual')+own_v,np.nan)*25
    up_q=np.where(c('new_upstream_flow_per_lane_available')>0,c('new_upstream_flow_per_lane_residual')+own_q,np.nan)*600
    down_q=np.where(c('new_downstream_flow_per_lane_available')>0,c('new_downstream_flow_per_lane_residual')+own_q,np.nan)*600
    up_density=up_q/np.maximum(up_v,5);down_density=down_q/np.maximum(down_v,5)
    dk=down_density-up_density
    wave=np.divide(down_q-up_q,dk,out=np.full_like(dk,np.nan),where=np.abs(dk)>1)
    add('shock_proxies','density_gradient',dk/100.)
    add('shock_proxies','shock_velocity',np.clip(wave,-40,120)/40.)
    add('shock_proxies','shock_gap_interaction',np.clip(wave,-40,120)/40.*gap/60.)
    add('shock_proxies','congested_density_gradient',dk/100.*congestion)
    add('shock_proxies','upstream_downstream_density_product',up_density*down_density/10000.)
    add('shock_proxies','density_gradient_confidence',dk/100.*confidence)
    # Hypotheses from the January single-feature screen, not accepted winners.
    disagreement=data['eng_speed_geometry_topology_disagreement']
    flow_deviation=data['eng_flow_vph_history_deviation']
    speed_spread=data['eng_speed_ensemble_spread']
    speed_laplacian=data['eng_speed_spatial_laplacian']
    topology_support=np.minimum(c('new_upstream_speed_available'),c('new_downstream_speed_available'))
    geometry_support=np.minimum(c('new_left_speed_available'),c('new_right_speed_available'))
    for name,value in [('disagreement_confidence',disagreement*confidence),
                       ('disagreement_congestion',disagreement*np.maximum(1-ratio,0)),
                       ('disagreement_quality',disagreement*quality),
                       ('disagreement_complete_neighbors',disagreement*topology_support*geometry_support),
                       ('disagreement_gap',disagreement*np.minimum(gap/60.,3)),
                       ('disagreement_density',disagreement*np.minimum(density,3)),
                       ('flow_deviation_confidence',flow_deviation*confidence),
                       ('flow_deviation_congestion',flow_deviation*np.maximum(1-ratio,0)),
                       ('flow_deviation_utilization',flow_deviation*np.clip(util,0,2)),
                       ('speed_spread_confidence',speed_spread*confidence),
                       ('speed_spread_congestion',speed_spread*np.maximum(1-ratio,0)),
                       ('disagreement_flow_interaction',disagreement*np.clip(flow_deviation,-3,3)),
                       ('disagreement_spread_interaction',disagreement*np.minimum(speed_spread,3)),
                       ('laplacian_spread_interaction',speed_laplacian*np.minimum(speed_spread,3))]:
        add('winner_interactions',name,value)
    refined=[('refined_disagreement_gap_saturation',disagreement*(1-np.exp(-np.minimum(gap,360)/30.))),
             ('refined_disagreement_density_saturation',disagreement*density/(.2+np.maximum(density,0))),
             ('refined_disagreement_positive_congestion',np.maximum(disagreement,0)*np.maximum(1-ratio,0)),
             ('refined_disagreement_negative_congestion',np.minimum(disagreement,0)*np.maximum(1-ratio,0)),
             ('refined_disagreement_noise_normalized',np.clip(disagreement/(.1+speed_spread),-5,5)),
             ('refined_disagreement_loggap_quality',disagreement*np.log1p(np.minimum(gap/5.,72))*quality),
             ('refined_laplacian_density_confidence',speed_laplacian*density/(.2+np.maximum(density,0))*confidence)]
    for name,value in refined:add('refined_interactions',name,value)
    add('calibration','constant',np.ones_like(v))
    result=np.concatenate([raw,np.stack(list(data.values()),axis=-1)],axis=-1)
    all_names=list(names)+list(data)
    groups['bounded_time']=[name for name in names if name.startswith('new_bounded_')]
    groups['spatial']=[name for name in names if name.startswith(('new_left_','new_right_'))]
    groups['topology']=[name for name in names if name.startswith(('new_upstream_','new_downstream_'))]
    robust=groups['robust_estimators']
    groups['robust_speed']=[x for x in robust if 'speed_' in x]
    groups['robust_flow']=[x for x in robust if 'flow_per_lane_' in x]
    groups['robust_median']=[x for x in robust if x.endswith('_median_contrast')]
    groups['robust_geometric']=[x for x in robust if '_geometric_contrast' in x]
    groups['shock_density']=[x for x in groups['shock_proxies'] if '_shock_' not in x]
    groups['shock_velocity']=[x for x in groups['shock_proxies'] if '_shock_' in x]
    for family in ['calibration','bounded_time','physics','robust_estimators','robust_speed','robust_flow','robust_median','robust_geometric','shock_density']:
        groups['output_'+family]=list(groups[family])
    if np.isinf(result).any():raise ValueError('Infinite engineered feature')
    return result,all_names,groups
