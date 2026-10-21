"""Additional observed-only occupancy and sensor-quality inputs."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from prepare_trafficflow_data import grid
from prepare_features import neighbor

QUALITY_FEATURES=['occupancy','occupancy_available','occupancy_prev5','occupancy_prev5_available',
    'occupancy_next5','occupancy_next5_available','occupancy_up','occupancy_up_available',
    'occupancy_down','occupancy_down_available','is_imputed','is_imputed_prev5','is_imputed_next5',
    'quality_prev5','quality_next5','is_observed']

def shifted(x,offset,fill=0):
    out=np.full_like(x,fill)
    if offset>0:out[offset:]=x[:-offset]
    else:out[:offset]=x[-offset:]
    return out

def quality_features(frame,links,graph):
    speed,flow,quality,occupancy,imputed,observed=grid(frame,links,
        ['speed_kmh','flow_vph','pct_observed','occupancy','is_imputed','is_observed'])
    valid=np.isfinite(speed)&np.isfinite(flow)&np.isfinite(occupancy)&(quality>=75)
    occ=np.where(valid,occupancy/.3,0).astype(np.float32)
    up,um=neighbor(occ[...,None],valid,graph['up_index'],graph['up_weight'])
    down,dm=neighbor(occ[...,None],valid,graph['down_index'],graph['down_weight'])
    imputed=np.nan_to_num(imputed.astype(np.float32),nan=1.)
    quality=np.nan_to_num(quality.astype(np.float32))/100
    values=[occ,valid,shifted(occ,1),shifted(valid,1),shifted(occ,-1),shifted(valid,-1),
        up[...,0],um,down[...,0],dm,imputed,shifted(imputed,1,1),shifted(imputed,-1,1),
        shifted(quality,1),shifted(quality,-1),np.nan_to_num(observed.astype(np.float32))]
    result=np.stack(values).astype(np.float16)
    assert len(result)==len(QUALITY_FEATURES) and np.isfinite(result).all()
    return result
