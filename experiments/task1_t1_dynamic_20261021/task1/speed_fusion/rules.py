from __future__ import annotations

import numpy as np
import pandas as pd
from numbers import Real
from ..data import KEY, TARGETS
from ..fusion_experiments import aligned_prediction
from ..metrics import validate_predictions

CONTROL='Current_best_wave_hybrid'
NEURAL='Long_context_neural'
WEIGHTS={'Speed_gap15_60_a10':.10,'Speed_gap15_60_a25':.25,'Speed_gap15_60_a50':.50}
PRIMARY='Speed_gap15_60_a25'
METHODS=[CONTROL,*WEIGHTS]
SIDES=['speed_minutes_before','speed_minutes_after']
RULE='finite positive before/after; 15 < before+after <= 60: blend speed only; all flow and other speed copied from current best'


def gate_from_sides(before,after):
    before,after=np.asarray(before,dtype=float),np.asarray(after,dtype=float)
    if before.ndim!=1 or after.shape!=before.shape:
        raise ValueError('Side times must be aligned one-dimensional vectors')
    for side in [before,after]:
        if np.isneginf(side).any() or np.any(np.isfinite(side)&(side<=0)):
            raise ValueError('Invalid observable side time')
    supported=np.isfinite(before)&np.isfinite(after)
    with np.errstate(over='ignore',invalid='ignore'):gap=before+after
    if np.any(supported&~np.isfinite(gap)):
        raise ValueError('Observable time overflow')
    return supported&(gap>15)&(gap<=60),gap


def context_for(expected,diagnostics):
    context=diagnostics[KEY+SIDES].copy()
    if context[KEY].isna().any().any() or context.duplicated(KEY).any():
        raise ValueError('Null or duplicate context keys')
    check=expected[KEY].merge(context[KEY],on=KEY,how='outer',indicator=True,validate='one_to_one')
    if not check['_merge'].eq('both').all():
        raise ValueError('Context target coverage differs')
    context=expected[KEY].merge(context,on=KEY,sort=False,validate='one_to_one')
    gate,gap=gate_from_sides(*(context[k].to_numpy() for k in SIDES))
    context['gap_minutes']=gap
    context['gap']=np.asarray(pd.cut(gap,[0,15,30,60,np.inf],labels=['<=15min','15-30min','30-60min','>60min']).astype(str),dtype=object)
    context.loc[~np.isfinite(gap),'gap']='one_side_or_none'
    context['blend_speed']=gate
    return context


def fuse_speed(expected,current,neural,diagnostics,alpha):
    """No truth, regime branch or congestion label enters the gate or blend."""
    if isinstance(alpha,(bool,np.bool_)) or not isinstance(alpha,Real) or not np.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('Neural weight must be finite and between zero and one')
    current=aligned_prediction(expected,current);neural=aligned_prediction(expected,neural)
    context=context_for(expected,diagnostics);gate=context.blend_speed.to_numpy()
    result=current.copy()
    # Preserve all flow values/dtype; converting speed to float64 is exact for float32 sources too.
    result['speed_kmh']=result.speed_kmh.astype(float)
    result.loc[gate,'speed_kmh']=(1-alpha)*current.speed_kmh.to_numpy(float)[gate]+alpha*neural.speed_kmh.to_numpy(float)[gate]
    validate_predictions(expected,result)
    return result


def audit(expected,current,neural,prediction,context,alpha):
    current=aligned_prediction(expected,current);neural=aligned_prediction(expected,neural)
    prediction=aligned_prediction(expected,prediction);context=context_for(expected,context)
    gate=context.blend_speed.to_numpy()
    flow_equal=np.array_equal(prediction.flow_vph.to_numpy(),current.flow_vph.to_numpy())
    outside_equal=np.array_equal(prediction.speed_kmh.to_numpy()[~gate],current.speed_kmh.to_numpy()[~gate])
    desired=(1-alpha)*current.speed_kmh.to_numpy(float)[gate]+alpha*neural.speed_kmh.to_numpy(float)[gate]
    inside_equal=np.array_equal(prediction.speed_kmh.to_numpy()[gate],desired)
    if not flow_equal or not outside_equal or not inside_equal:
        raise ValueError('Speed-only preservation/formula invariant failed')
    return dict(passed=True,alpha=alpha,target_rows=len(expected),eligible_rows=int(gate.sum()),
        preserved_speed_rows=int((~gate).sum()),all_flow_values_equal=True,all_outside_speed_values_equal=True,
        all_inside_speed_values_match_formula=True,changed_speed_rows=int(np.count_nonzero(prediction.speed_kmh.to_numpy()!=current.speed_kmh.to_numpy())))
