"""Append genuinely new channels while keeping all 51 legacy features intact."""
import numpy as np
import pandas as pd
from spatial_experiments import Features,design
from temporal_features import replace_temporal


def append_features(experiment,date,features,obs):
    e=experiment
    if not hasattr(e,'_extra_feature_builder'):
        e._extra_feature_builder=Features(e.panel,obs,include_ramps=False)
    links=e._extra_feature_builder.links
    stamps=sorted(obs.timestamp.astype(str).unique())
    if len(stamps)!=288:raise ValueError('Expected 288 five-minute slots')
    station_by_link=dict(zip(e.links,e.stations))
    targets=pd.DataFrame({'timestamp':np.repeat(stamps,len(links)),
                          'link_id':np.tile(links,len(stamps))})
    targets['station_id']=targets.link_id.map(station_by_link)
    old_index={x:i for i,x in enumerate(e.links)}
    mapping=np.array([old_index[x] for x in links])
    weekday=pd.Timestamp(date).weekday()
    historical=e.history[:,:,weekday]
    raw=np.stack([historical[0,mapping].T,
                  (historical[1,mapping]*e.lanes[mapping,None]).T],axis=-1).reshape(-1,2)
    bounded,_=replace_temporal(obs,targets,raw)
    groups,names,norm=e._extra_feature_builder.get(obs,targets,bounded)
    # The legacy 51 channels already contain the four static road parameters.
    # Keep only the eight new multihop neighbor channels from this group.
    groups['topology']=groups['topology'][:,:8]
    names['topology']=names['topology'][:8]
    extra=design(groups,['spatial','topology'])[:,1:]
    extra=np.column_stack([extra,bounded/norm])
    labels=[f'new_{name}' for group in ['spatial','topology'] for name in names[group]]
    labels+=['new_bounded_temporal_speed','new_bounded_temporal_flow_per_lane']
    inverse=np.array([links.index(x) for x in e.links])
    tensor=extra.reshape(288,len(links),-1).transpose(1,0,2)[inverse]
    for j,label in enumerate(labels):features[label]=tensor[:,:,j]
    if len(features)!=69:raise ValueError(f'Expected 69 raw channels, got {len(features)}')
    return features
