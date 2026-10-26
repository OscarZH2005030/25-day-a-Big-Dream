"""仅推理特征运行时：独立于任何旧训练目录，不含训练入口。"""
import sys,json
from pathlib import Path
import numpy as np
BUNDLE=Path(__file__).resolve().parent/'feature-source'
sys.path[:0]=[str(BUNDLE/'code/original'),str(BUNDLE/'code/features')]
from task1_baseline import Experiment,KEYS
from augment_features import append_features
from feature_bank import generate
SCHEMA=json.loads((BUNDLE/'feature_schema.json').read_text(encoding='utf8'))
BASE=json.loads((BUNDLE/'base_feature_schema.json').read_text(encoding='utf8'))
PANELS=json.loads((BUNDLE/'split.json').read_text(encoding='utf8'))['panels']

def expand_fast(matrix):
 """Only the two selected families, vectorized six-point NaN quantiles."""
 assert SCHEMA['frozen_spec']=={'masked':[],'removed':[],'added':['gap_reliability','multiscale_robust'],'channel_masked':{'speed':[],'flow_per_lane':[]}}
 ix={n:i for i,n in enumerate(BASE['all_numeric_features'])};extra={}
 def get(n):return matrix[:,ix[n]]
 def put(n,v):extra['night_'+n]=np.where(np.isfinite(v),np.clip(v,-1e6,1e6),np.nan).astype('float32')
 def quantiles(a,qs):
  valid=np.isfinite(a);count=valid.sum(1);s=np.sort(np.where(valid,a,np.inf),axis=1);res=[]
  for q in qs:
   rank=np.maximum(count-1,0)*q;lo=np.floor(rank).astype(int);hi=np.ceil(rank).astype(int);frac=rank-lo
   left=s[np.arange(len(a)),lo];right=s[np.arange(len(a)),hi]
   v=left.astype('float64')+(right.astype('float64')-left)*frac;v[count==0]=np.nan;res.append(v.astype('float32'))
  return res
 with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
  for ch,scale in [('speed_kmh',25.),('flow_vph',600.)]:
   initial=get(ch+'_initial');left=get(ch+'_prev');right=get(ch+'_next');before=get(ch+'_offset_-5min');after=get(ch+'_offset_5min');gl=np.maximum(get(ch+'_prev_gap_min'),0);gr=np.maximum(get(ch+'_next_gap_min'),0);span=gl+gr+1;p=ch+'_'
   put(p+'weighted_endpoint_delta',((left-initial)*(gr+1)+(right-initial)*(gl+1))/((span+1)*scale));put(p+'endpoint_slope',(right-left)/((span+1)*scale));put(p+'endpoint_disagreement',np.abs(right-left)/((np.sqrt(span)+1)*scale));put(p+'near_slope_confidence',(after-before)/(scale*(1+span/10)));put(p+'gap_balance_log',np.log1p(gl)-np.log1p(gr))
  for ch,scale in [('speed_kmh',25.),('flow_vph',600.)]:
   a=np.column_stack([get(ch+'_offset_'+str(o)+'min') for o in [-30,-15,-5,5,15,30]]);median,lower,upper=quantiles(a,[.5,.25,.75]);mad=quantiles(np.abs(a-median[:,None]),[.5])[0];center=get(ch+'_initial');p=ch+'_'
   for n,v in [('median_residual',(median-center)/scale),('iqr',(upper-lower)/scale),('mad',mad/scale),('robust_z',(center-median)/(mad+scale*.05)),('lower_tail',(lower-center)/scale),('upper_tail',(upper-center)/scale)]:put(p+n,v)
 numeric=np.column_stack(list(extra.values()));return np.concatenate([matrix,numeric,np.isfinite(numeric)],axis=1).astype('float32'),BASE['input_names']+list(extra)+[n+'__finite' for n in extra]
