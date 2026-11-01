"""Inference-only replay of four frozen policies on frozen extra origins.

No fitting functions. First reconstruct and verify OLD cached inference, then
write target-free predictions for each new origin before accessing its labels.
"""
import argparse,json,hashlib,copy
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import lightgbm as lgb
from core import ROOT,r,Context,softmax,verifier_features
from run import Generator
import onset_phase as phase
from joint import joint_features
from propagation_sequence import features as observed_features,horizon_features
from arrival_prefix import propose
from boundary_context import directed_features
from spatial_arrival import field_features
from arrival_unit import unit_proposal,unit_features
from arrival_wait import timing_features
from arrival_unit_directional import models as primary_models
from arrival_wait_combined import load as retained_model
from margin import Margin
import arrival_wait as wait
import arrival_context as context
from episodes import load as old_load
from phase_summary import date_ci

HERE=ROOT/'work/task2-cascade/origin_stress';HERE.mkdir(exist_ok=True)
NAMES=['R127','retained','accepted_alternative','sequence_control']
INPUT_KEYS=['x','network','panel','origin','condition','role','feature_names']

def prediction_input(s):return {k:s[k] for k in INPUT_KEYS if k in s}

class FrozenPipeline:
    def __init__(self,args,backend):
        self.backend=backend;self.parent=backend[0];self.ctx=Context(args,backend,ROOT/'work/task2-rescue-v2/profiles',r.read,r.write,lambda *a,**kw:None);self.gen=Generator('full',self.ctx,backend[2]);self.phase=lgb.Booster(model_file=str(phase.HERE/'full-phase.txt'));self.margin=Margin(lgb.Booster(model_file=str(phase.BASE/'margin-full.txt')));self.primary=primary_models('full')['committee_advance'][0];self.retained=retained_model('full')['pair'];self.alt=lgb.Booster(model_file=str(wait.HERE/'observed_sequence/action_population/full-accepted--1.txt'));self.control=[lgb.Booster(model_file=str(wait.HERE/f'observed_sequence/full-current-{k}.txt')) for k in range(3)]
    def predict(self,sample,debug=False):
        s=prediction_input(sample);b=np.asarray(self.parent.probability(s),dtype='float32');parent=b>=(.5 if s['condition']==0 else .25);pred={n:parent.copy() for n in NAMES};diag=dict(b=b,rescue=np.zeros_like(parent),unit={},timing={},observed={},decisions={})
        if s['condition']!=0:return pred,diag
        ep,q,pool=self.gen.predict(s,b);vf0,_=verifier_features(s,self.ctx,ep,q,b,'full');ph,_,_=joint_features(s);ri,hi=np.nonzero(((q>=.5)&(b<.5)).T);oldX=np.column_stack([vf0[ri,hi],ph[ri,hi]]);take=self.margin.predict(oldX,num_threads=4)>=.5 if len(ri) else np.zeros(0,bool);rescue=np.zeros_like(parent);rescue[hi[take],ri[take]]=True
        for n in NAMES[1:]:pred[n]|=rescue
        x=self.backend[2](s);extra,future=self.ctx.extra(s,'full');xx=np.column_stack([x,extra]);ids=np.flatnonzero(pool);prob=np.zeros((len(x),6,3),dtype='float32');prob[:,:,0]=1
        if len(ids):
            matrix=np.column_stack([np.tile(xx[ids],(6,1)),np.repeat(np.eye(6,dtype='float32'),len(ids),axis=0),future[ids].transpose(1,0,2).reshape(6*len(ids),10)]);prob[ids]=softmax(self.phase.predict(matrix,raw_score=True,num_threads=4)).reshape(6,len(ids),3).transpose(1,0,2)
        conditional,joint,first,before,after=phase.decode(prob,ep);conditional=conditional.T;joint=joint.T;conditional[:,~pool]=0;joint[:,~pool]=0;use=self.backend[-1](s);shaped=phase.reshape_parent(b,conditional,use);changed=((shaped>=.5)!=(b>=.5)).any(0);vf,_=verifier_features(s,self.ctx,ep,conditional,b,'full');directed,_=directed_features(s,b);field,_=field_features(s,x[:,-359:]);raw=observed_features(s)[0]
        ids,direction,prefix,newfirst,oldfirst,rank=propose(b,first,conditional,use)
        for row,link in enumerate(ids):
            if direction[row]!=0 or not changed[link]:continue
            action=unit_proposal(b[:,link],prefix[row],0)
            if action is None:continue
            h,value=action
            if rescue[h,link]:continue
            oldbits=b[:,link]>=.5;newbits=shaped[:,link]>=.5;oldpos=np.flatnonzero(oldbits);newpos=np.flatnonzero(newbits)
            summary=np.r_[vf[link].mean(0),vf[link,:,103:].min(0),vf[link,:,103:].max(0),ph[link].mean(0),b[:,link],conditional[:,link],joint[:,link],first[link],before[link],after[link],oldbits,newbits,oldpos[0]/5,newpos[0]/5,oldbits.mean(),newbits.mean()].astype('float32');assert len(summary)==308
            pref=np.r_[summary,np.eye(6)[newfirst[row]],np.eye(6)[oldfirst[row]],prefix[row],rank[row],direction[row]].astype('float32');assert len(pref)==351
            c=np.r_[pref,directed[link].mean(0),field[link].mean(0),directed[link,newfirst[row]]].astype('float32');assert len(c)==511
            u=unit_features(c,b[:,link],rescue[:,link],first[link],before[link],after[link],conditional[:,link],joint[:,link],h,value);t=timing_features(u,b,first,link,h,s['network']['edges']);f=np.r_[t,horizon_features(raw[link],h)].astype('float32');key=(int(link),int(h));diag['unit'][key]=u;diag['timing'][key]=t;diag['observed'][key]=f
            accepted=self.primary.predict(u[None],num_threads=4)[0]>=.5;veto=self.retained.predict(t[None],num_threads=4)[0]>=.5;alternative=self.alt.predict(f[None],num_threads=4)[0]>=.5;control=all(m.predict(f[None],num_threads=4)[0]>=.5 for m in self.control);diag['decisions'][key]=[bool(accepted),bool(veto),bool(alternative),bool(control)]
            if accepted and not veto:
                pred['retained'][h,link]=True
                if not alternative:pred['accepted_alternative'][h,link]=True
                if not control:pred['sequence_control'][h,link]=True
        for n in NAMES[1:]:assert not (parent&~pred[n]).any() and not (rescue&~pred[n]).any()
        diag.update(rescue=rescue,raw_ri=ri,raw_hi=hi,rawX=oldX,conditional=conditional,joint=joint,first=first,before=before,after=after)
        return pred,diag

def check_models(protocol):
    changed=[]
    for path,expected in protocol['model_sha256'].items():
        p=Path(path)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append(path)
    assert not changed,changed[:8]
    return len(protocol['model_sha256'])

def validate_old(pipeline,backend):
    reports=[]
    for pop,mode in [('december','first'),('december','alternative_veto'),('december','retained_veto'),('january','first'),('february_new','critical'),('february_old','critical')]:
        meta,z,protected,cases=wait.candidates(backend,pop,'full','full');catalog=r.read(wait.HERE/f'combined/cases-full-{pop}.json')
        if mode=='alternative_veto':target=next(v for v in catalog if v['panel']=='D7_I405_S' and v['link_index']==5 and v['horizon_minutes']==25 and v['origin'].startswith('2030-12-19'))
        elif mode=='retained_veto':target=next(v for v in catalog if v['veto']['multi'])
        elif pop=='february_new':target=next(v for v in catalog if v['panel']=='D7_I210_W' and v['link_index']==25 and v['horizon_minutes']==25)
        elif pop=='february_old':target=next(v for v in catalog if v['panel']=='D7_I405_N' and v['link_index']==260)
        else:target=catalog[0]
        i=next(i for i,e in enumerate(meta) if e['panel']==target['panel'] and e['origin']==target['origin']);e=meta[i];slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));p,d=pipeline.predict(s,True)
        assert np.allclose(d['b'],z[f'{i}_base'],atol=2e-6,rtol=0),('base',pop,float(abs(d['b']-z[f'{i}_base']).max()))
        assert np.array_equal(d['rescue'],protected[i]),('rescue',pop)
        _,old=old_load(pop,'full');assert np.array_equal(d['raw_ri'],old[f'{i}_ri']) and np.array_equal(d['raw_hi'],old[f'{i}_hi'])
        assert np.allclose(d['rawX'],old[f'{i}_Z'],atol=2e-5,rtol=0),('raw margin features',pop)
        expected=[c for c in cases if c['i']==i];keys={(c['link'],c['h']) for c in expected};assert keys==set(d['unit']),('proposal',pop,keys,set(d['unit']))
        with np.load(wait.HERE/f'observed_sequence/raw-full-{pop}.npz') as raw:cached_raw={c['link']:raw[f"{i}_{c['link']}"] for c in expected}
        maxerror=0
        for c in expected:
            key=(c['link'],c['h']);a=d['unit'][key];b=c['x'];error=float(abs(a-b).max());maxerror=max(maxerror,error);assert np.allclose(a,b,atol=2e-5,rtol=0),('unit',pop,key,error,int(np.argmax(abs(a-b))))
            assert np.allclose(d['timing'][key],c['f'],atol=2e-5,rtol=0),('timing',pop,key)
            assert np.allclose(d['observed'][key],np.r_[c['f'],horizon_features(cached_raw[c['link']],c['h'])],atol=2e-5,rtol=0),('observed features',pop,key)
        actual_catalog={(v['link_index'],v['horizon_minutes']//5-1):v for v in catalog if v['panel']==e['panel'] and v['origin']==e['origin']}
        for key,v in actual_catalog.items():assert d['decisions'][key][0] and d['decisions'][key][1]==v['veto']['multi'],('decision',pop,key)
        altered=dict(s,y=1-s['y'],m=~s['m'],has_future_queue=not s['has_future_queue'],speed_y=np.ones_like(s['y'])*999)
        altered_pred=pipeline.predict(altered)[0];assert all(np.array_equal(p[n],altered_pred[n]) for n in NAMES),('target poisoning',pop)
        reports.append(dict(pop=pop,mode=mode,panel=e['panel'],origin=e['origin'],proposals=len(expected),max_unit_abs_error=maxerror,rescue_exact=True,decisions_exact=True,target_fields_ignored=True));print(json.dumps(dict(stage='old_reconstruction',**reports[-1])),flush=True)
    r.write(ROOT/'outputs/task2_origin_stress_reconstruction.json',dict(passed=True,cases=reports,new_origin_inference_started=False));return reports

def score(backend,protocol):
    state={month:{name:dict(rows=[],neg=defaultdict(list),TP=0,FP=0,FN=0,lost_retained_onset_TP=0,all_onset_TP=0,all_onset_FP=0,all_onset_FN=0,negative_FP_cells=0,positive_windows=0,negative_windows=0,onset_windows=0,ongoing_windows=0) for name in NAMES} for month in ['2030-12','2031-01','2031-02','all']}
    for j,e in enumerate(protocol['origins']):
        with np.load(HERE/f'pred-{j:04d}.npz') as saved:pred={n:saved[n] for n in NAMES}
        values=backend[1].load_day(e['panel'],e['date']);sample=backend[1].make_sample(values,backend[3][e['panel']],e['origin_slot'],e['panel']);y=sample['y'].astype(bool);m=sample['m'];condition=int(sample['condition']);positive=sample['has_future_queue']
        for month in [e['date'][:7],'all']:
            for name,p in pred.items():
                a=state[month][name];a['positive_windows']+=int(positive);a['negative_windows']+=int(not positive);a['onset_windows']+=int(condition==0);a['ongoing_windows']+=int(condition==1)
                if condition==0:
                    tp=int((p&y&m).sum());fp=int((p&~y&m).sum());fn=int((~p&y&m).sum());a['all_onset_TP']+=tp;a['all_onset_FP']+=fp;a['all_onset_FN']+=fn;a['lost_retained_onset_TP']+=int((pred['retained']&~p&y&m).sum())
                    if positive:a['TP']+=tp;a['FP']+=fp;a['FN']+=fn
                    else:a['negative_FP_cells']+=fp;a['neg'][e['panel']].append(bool((p&m).any()))
                if positive:a['rows'].append(dict(panel=e['panel'],condition=condition,window_id=e['origin'],date=e['date'],iou=backend[4](y,p,m)))
    results={}
    for month,models in state.items():
        results[month]={}
        for name,a in models.items():
            rows=a.pop('rows');neg=a.pop('neg');s=backend[5](rows);on=[v for k,v in s['by_panel_condition'].items() if k.endswith('queue_onset')];s.update(a,onset_macro=float(np.mean(on)) if on else None,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None);results[month][name]=dict(scores=s,windows=rows)
        before=results[month]['R127']['windows']
        for name in NAMES[1:]:
            after=results[month][name]['windows']
            try:results[month][name]['vs_R127_onset_interval']=date_ci(before,after,0)
            except (ValueError,ZeroDivisionError):results[month][name]['vs_R127_onset_interval']=None
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();protocol=r.read(ROOT/'outputs/task2_origin_stress_frozen.json');check_models(protocol);setup=r.read(ROOT/'outputs/task2_research_handoff.json')['args'];cfg=SimpleNamespace(**{k:Path(v) for k,v in setup.items()});backend=r.setup(cfg);pipeline=FrozenPipeline(cfg,backend);validate_old(pipeline,backend)
    if args.validate_only:return
    marker=ROOT/'outputs/task2_origin_stress_progress.json';r.write(marker,dict(status='inference',completed=0,total=len(protocol['origins']),roster_sha256=protocol['roster_sha256']))
    for j,e in enumerate(protocol['origins']):
        path=HERE/f'pred-{j:04d}.npz'
        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if path.exists():
            prior=r.read(HERE/f'pred-{j:04d}.json');assert prior['roster_sha256']==protocol['roster_sha256'] and prior['implementation_sha256']==code_hash and prior['panel']==e['panel'] and prior['origin']==e['origin']
        else:
            values=backend[1].load_day(e['panel'],e['date']);s=backend[1].make_sample(values,backend[3][e['panel']],e['origin_slot'],e['panel']);pred,debug=pipeline.predict(s);np.savez_compressed(path,**pred)
            # Save prediction-only data before any label-based scoring.
            r.write(HERE/f'pred-{j:04d}.json',dict(origin=e['origin'],panel=e['panel'],condition=int(s['condition']),roster_sha256=protocol['roster_sha256'],implementation_sha256=code_hash,target_fields_in_prediction_file=False))
        if (j+1)%25==0:r.write(marker,dict(status='inference',completed=j+1,total=len(protocol['origins']),roster_sha256=protocol['roster_sha256']));print(json.dumps(dict(stage='frozen_origin_inference',completed=j+1,total=len(protocol['origins']))),flush=True)
    check_models(protocol);results=score(backend,protocol);r.write(ROOT/'outputs/task2_origin_stress_results.json',dict(reports=results,roster_sha256=protocol['roster_sha256'],model_hashes_verified_before_after=True,no_refitting=True,same_development_months=True,not_independent_new_month=True,official_truth_available=False,policy_selection_using_results=False));r.write(marker,dict(status='complete',completed=len(protocol['origins']),total=len(protocol['origins'])));print(json.dumps({m:{n:{k:v['scores'][k] for k in ['proxy_IoU','onset_macro','TP','FP','FN','lost_retained_onset_TP','negative_false_any']} for n,v in a.items()} for m,a in results.items()}),flush=True)

if __name__=='__main__':main()
