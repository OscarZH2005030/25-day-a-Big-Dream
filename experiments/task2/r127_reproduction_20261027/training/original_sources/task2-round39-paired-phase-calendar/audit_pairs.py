"""Source-neutral phase trios sharing a first-positive observed physical cell."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;PROJECT=ROOT.parents[1]
sys.path[1:1]=[str(OUT/n) for n in ['task2-round33-conditional-start','task2-round10-phase-coverage','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from run_round2 import arrays,sample,PANELS,OLD
from timing_core import unseen_queue
from task2q.cache import atomic_json
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
    d=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',d);atomic_json(OUT/'overnight-status.json',dict(d,training=False,script=str(Path(__file__))));print(json.dumps(d),flush=True)
def guard(stage):
    m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(h),hashes=h));return h
def main():
    started=time.time();before=guard('before');save('experiment.json',dict(hypothesis='R36 found origin phase associated with R10 sampling source; collect all-three-phase matched views of the SAME first-positive timestamp/link observation, before any phase-feature trial',scope='complete JuneNov public fit calendar only, no Dec/Jan/Feb data',label='complete six future labels, positive event, own link no observed eligible queue in released13frames; whole window Onset',pair_unit='same panel/day/link/first-observed-future-positive timestamp; NOT an independent physical event ID',selection='from every physical observation group, earliest available triple of consecutive5min origins; optionally latest triple only if disjoint from earliest; all legal origins have same Onset/conditional target rules',phase='each triple contains exactly one of each origin slot modulo3, no mod3 feature is added',augmentation_plan='retain original conditional rows; append balanced trio views (including labeled duplicates if already selected); original window raw mass1 and each added physical target group raw mass1, normalize globally; not an unbiased official macroIoU loss',gate='>=150 paired groups in JuneSep, >=50 in OctNov, plus >=20/10 per applicable direction; exact original-target replay and equal phase counts required',future='future quality/coverage/group IDs and target timestamps used ONLY for label/sampling; no inference features',CPU_threads=4,new_models=0,Task1='92files immutable, historical .942828633 not remeasured',official_S_queue=None))
    entries=read(OUT/'task2-round10-phase-coverage/expanded-fit.json')['samples'];manifest=read(OLD/'datasets/fit/index.json');netkeys={p:next(e['network_key'] for e in entries if e['panel']==p) for p in PANELS};nets={p:arrays(k) for p,k in netkeys.items()};sources={p:manifest['assets']['networks'][k]['metadata']['sources'] for p,k in netkeys.items()};origins={}
    for e in entries:
        if e['condition']==0:origins.setdefault((e['panel'],e['day_key']),set()).add(e['origin_slot'])
    original=[];paired=[];groups=[];stats={p:{r:dict(days=0,available_target_groups=0,paired_target_groups=0,paired_views=0,overlap_original_views=0,phase_counts=[0,0,0],first_step_hist=[0]*6) for r in ['early','held']} for p in PANELS};checked=[]
    for i,(key,rec) in enumerate(manifest['assets']['days'].items()):
        src=rec['metadata']['sources'];matched=[p for p,s in sources.items() if all(src.get(a)==b for a,b in s.items())];assert len(matched)==1;p=matched[0];net=nets[p];v=arrays(key);n=len(net['link_ids']);date=pd.Timestamp(int(v['times_ns'][0]),tz='UTC').strftime('%Y-%m-%d');assert '2030-06-01'<=date<='2030-11-30';period='early' if date<'2030-10-01' else 'held';a=stats[p][period];a['days']+=1
        q=np.isfinite(v['speed'])&(v['speed']<=net['threshold'])&v['eligible'];good=v['eligible']&np.isfinite(v['speed'])&np.isfinite(v['quality'])&(v['quality']>=75);positive=q&good;origin_q=np.isfinite(v['origin_speed'])&(v['origin_speed']<=net['threshold'])&v['origin_eligible'];pq=np.concatenate([np.zeros((1,n),int),np.cumsum(q,axis=0)]);pg=np.concatenate([np.zeros((1,n),int),np.cumsum(good,axis=0)]);pp=np.concatenate([np.zeros((1,n),int),np.cumsum(positive,axis=0)]);by_target={};old=origins.get((p,key),set());day_original=set()
        for slot in range(12,282):
            past=pq[slot]-pq[slot-12]
            if (past>=2).any():continue
            use=(past==0)&~origin_q[slot]&((pg[slot+7]-pg[slot+1])==6)&((pp[slot+7]-pp[slot+1])>0)
            if not use.any():continue
            first=np.argmax(positive[slot+1:slot+7],axis=0)
            for link in np.flatnonzero(use):
                h=int(first[link]);j=int(link);target=slot+h+1;by_target.setdefault((target,j),[]).append(slot)
                if slot in old:
                    row=dict(panel=p,day_key=key,network_key=netkeys[p],date=date,period=period,origin_slot=int(slot),origin=pd.Timestamp(int(v['times_ns'][slot]),tz='UTC').isoformat(),target_link_index=j,target_link_id=str(net['link_ids'][j]),first_step=h,phase=int(slot%3),condition=0,source='original_conditional');original.append(row);day_original.add((slot,j))
        a['available_target_groups']+=len(by_target)
        for (target,j),slots in sorted(by_target.items()):
            available=set(slots);triples=[(s,s+1,s+2) for s in sorted(slots) if s+1 in available and s+2 in available]
            if not triples:continue
            selected=list(triples[0]);disjoint=[t for t in triples if not(set(t)&set(selected))]
            if disjoint:selected.extend(disjoint[-1])
            assert len(selected) in [3,6] and all(selected.count(s)==1 for s in selected);phasecounts=np.bincount(np.asarray(selected)%3,minlength=3);assert len(set(phasecounts.tolist()))==1
            gid=f'{p}/{date}/{j}/{target}';a['paired_target_groups']+=1;group=dict(panel=p,date=date,period=period,group=gid,target_link_index=j,target_first_positive_slot=int(target),available_origins=len(slots),selected_origins=[int(s) for s in selected],views=len(selected),raw_group_mass=1.);groups.append(group)
            for slot in selected:
                h=target-slot-1;assert 0<=h<6;overlap=(slot,j) in day_original;row=dict(panel=p,day_key=key,network_key=netkeys[p],date=date,period=period,origin_slot=int(slot),origin=pd.Timestamp(int(v['times_ns'][slot]),tz='UTC').isoformat(),target_link_index=j,target_link_id=str(net['link_ids'][j]),first_step=int(h),phase=int(slot%3),condition=0,source='paired_three_phase',label_side_group=gid,group_views=len(selected),raw_weight=1./len(selected),already_in_original=bool(overlap));paired.append(row);a['paired_views']+=1;a['overlap_original_views']+=int(overlap);a['phase_counts'][slot%3]+=1;a['first_step_hist'][h]+=1
        # Independent replay check on a paired view from each direction/period.
        if (p,period) not in checked:
            candidates=[r for r in paired if r['panel']==p and r['date']==date]
            if candidates:
                row=candidates[0];s=sample(row);j=row['target_link_index'];assert s['condition']==0 and unseen_queue(s)[j] and s['m'][:,j].all() and s['y'][:,j].any();assert int(np.argmax(s['y'][:,j]))==row['first_step'];checked.append((p,period))
        if (i+1)%100==0:progress('paired_calendar_scan',days=i+1,total=len(manifest['assets']['days']),paired_views=len(paired))
    original.sort(key=lambda r:(r['panel'],r['origin'],r['target_link_index']));paired.sort(key=lambda r:(r['panel'],r['origin'],r['target_link_index'],r['label_side_group']))
    early=[r for r in original if r['period']=='early'];full_y=np.asarray([r['first_step'] for r in original]);early_y=np.asarray([r['first_step'] for r in early]);assert np.array_equal(early_y,np.load(PROJECT/'work/round33-conditional-start/y.npy'));assert np.array_equal(full_y,np.load(PROJECT/'work/round34-conditional-start-full/y.npy'));assert len(checked)==16
    early_groups=sum(stats[p]['early']['paired_target_groups'] for p in PANELS);held_groups=sum(stats[p]['held']['paired_target_groups'] for p in PANELS);global_gate=bool(early_groups>=150 and held_groups>=50);roads={p:bool(global_gate and stats[p]['early']['paired_target_groups']>=20 and stats[p]['held']['paired_target_groups']>=10) for p in PANELS};assert all(stats[p]['early']['days']==122 and stats[p]['held']['days']==61 for p in PANELS)
    save('original-conditional-targets.json',original);save('paired-targets.json',paired);save('paired-groups.json',groups);save('calendar-pair-summary.json',dict(by_panel=stats,days=len(manifest['assets']['days']),complete_183day_calendar=True,original_early_rows=len(early),original_full_rows=len(original),original_label_replay_exact=True,paired_early_groups=early_groups,paired_held_groups=held_groups,paired_views=len(paired),label_overlap_not_independent_new_data=True));save('training-gates.json',dict(roads=roads,passed_roads=int(sum(roads.values())),global_gate=global_gate,training_early_and_held_data_only=True,no_December_January_February_used=True));assert guard('after')==before;save('final-verification.json',dict(passed=True,diagnostic_only=True,new_models=0,complete_calendar=True,original_target_replay_exact=True,paired_sample_replay_groups=16,source_neutral_equal_phase_counts_per_group=True,Task1_92files_unchanged=True,official_S_queue=None));(ROOT/'第三十九轮诊断说明.txt').write_text('同一first-positive物理观测的连续三起点视角，全部phase均衡；不是独立物理事件。原conditional目标逐行标签重放一致，未来信息仅标签侧使用。\n'+json.dumps(dict(stats=stats,early_groups=early_groups,held_groups=held_groups,roads=roads),ensure_ascii=False,indent=2)+'\n',encoding='utf8');progress('paired_calendar_complete',paired_views=len(paired),passed_roads=int(sum(roads.values())),seconds=time.time()-started)
if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
