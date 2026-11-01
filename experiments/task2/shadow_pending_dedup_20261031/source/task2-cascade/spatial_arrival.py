"""Low-frequency public-position field for prospective boundary arrivals."""
import argparse,json
from pathlib import Path
import numpy as np
from core import r
import structured_boundary as sb
from history_aligned_phase import load_aligned
from relative_support import relative_features

HERE=sb.HERE/'spatial';HERE.mkdir(exist_ok=True)

def field_features(s,g):
    n=s['x'].shape[1];assert g.shape==(n,359);field=g[:,:312].reshape(n,13,3,8).sum(3);coverage=g[:,312:351].reshape(n,13,3);support=np.any(g[:,351:]!=0,axis=1);names=[];cols=[];x=s['x']
    def add(name,v):names.append(name);cols.append(v)
    for ch,label in enumerate(['speed','flow','occupancy']):
        v=field[:,:,ch];valid=(x[:,:,4+ch]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);latest=np.zeros(n);age=np.full(n,13,int)
        for t in range(13):latest[valid[t]]=x[t,valid[t],ch];age[valid[t]]=12-t
        index=np.maximum(12-age,0);lastfit=v[np.arange(n),index];seen=valid.any(0);resid=x[:,:,ch].T-v;cnt=valid.sum(0)
        for span in [1,3,6,13]:add(label+f'_field_mean_{span}',v[:,-span:].mean(1))
        for lag in [3,6,12]:add(label+f'_field_change_{lag}',v[:,-1]-v[:,-1-lag])
        add(label+'_latest_observed_minus_field',np.where(seen,latest-lastfit,0));add(label+'_local_field_residual_mae',np.divide(np.where(valid.T,abs(resid),0).sum(1),cnt,out=np.zeros(n),where=cnt>0));add(label+'_field_coverage_now',coverage[:,-1,ch]);add(label+'_field_coverage_history',coverage[:,:,ch].mean(1))
        for direction,src,dst in [('upstream',s['network']['edges'][0],s['network']['edges'][1]),('downstream',s['network']['edges'][1],s['network']['edges'][0])]:
            level=np.zeros(n);trend=np.zeros(n);count=np.zeros(n)
            for a,j in zip(src,dst):
                if support[a] and support[j]:level[j]+=v[a,-1]-v[j,-1];trend[j]+=(v[a,-1]-v[a,-4])-(v[j,-1]-v[j,-4]);count[j]+=1
            add(label+'_'+direction+'_field_level_difference',np.divide(level,count,out=np.zeros(n),where=count>0));add(label+'_'+direction+'_field_trend_difference',np.divide(trend,count,out=np.zeros(n),where=count>0))
    add('position_field_supported',support);a=np.column_stack(cols).astype('float32');return np.tile(a[:,None,:],(1,6,1)),names

def extra(backend,pop,regime):
    dest=HERE/f'{regime}-{pop}-context.npz'
    if dest.exists():return dict(np.load(dest))
    meta,z,c=load_aligned(pop,regime);out={}
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        ri=c[f'{i}_ri'];hi=c[f'{i}_hi']
        if len(ri):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));base=backend[2](s);f,names=field_features(s,base[:,-359:]);rank,rnames=relative_features(z[f'{i}_base']);out[f'{i}_D']=np.column_stack([f[ri,hi],rank[ri,hi]])
        else:out[f'{i}_D']=None
        if (i+1)%512==0:print(json.dumps(dict(phase='spatial_arrival',pop=pop,done=i+1)),flush=True)
    for k,v in out.items():
        if v is None:out[k]=np.zeros((0,len(names)+len(rnames)),dtype='float32')
    np.savez_compressed(dest,**out);r.write(HERE/'context-schema.json',dict(names=names+rnames,source='degree0..7 projection of valid history onto explicit public order_index; no invented geographic distance or edge',future_observations_used=False));return out

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);sb.HERE=HERE;sb.extra=extra
    for pop in ['october','november']:extra(backend,pop,'early')
    models,theta=sb.train(['october'],backend,'probe');variants=['base','semantic_pointwise','structured_marginal','structured_map'];internal={k:sb.evaluate(backend,'november','early',models,theta,k)[0] for k in variants};b=internal['base'];eligible=[k for k in variants[1:] if internal[k]['ongoing_macro']>b['ongoing_macro'] and internal[k]['FP']<=b['FP'] and internal[k]['TP']>=b['TP']];selected=max(eligible,key=lambda k:internal[k]['ongoing_macro']) if eligible else 'base';r.write(args.output/'task2_spatial_arrival_internal.json',dict(scores=internal,selected=selected,later_month_selection=False,design_informed_by_reused_cases=True,inherited_routing_selected_later=True,internal_not_blind=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,later_month_selection=False,thresholds=dict(parent=.25,proposal=.5,verifier=.5)));models,theta=sb.train(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        extra(backend,pop,'full');reports[pop]={}
        for k in variants:
            scores,rows=sb.evaluate(backend,pop,'full',models,theta,k);reports[pop][k]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_spatial_arrival_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN','known_changed_cells']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
