"""Known-latent small-network posterior decisions, matched and wrong assumptions."""
import common as d
import task4_gaussian_posterior_pilot as pilot
import task4_span_shrink_stress as base
import numpy as np,pandas as pd,json,msvcrt,traceback,itertools,hashlib
from scipy.stats import truncnorm
from datetime import datetime,timezone
O=d.OUT/'task4_smallnet_posterior_stress';SEED=2026110311
SCENARIOS=['matched_conditional','near_sparse_censored','positive_background','correlated_multiplicative']
METHODS=['weak_prior','baseline','posterior_mean','median','projected_median','relative_L1_median','projected_relative_L1_median']

def geometry(nnode):
 pairs=np.array(list(itertools.combinations(range(nnode),2)));oi,di=pairs.T;Q=((oi[None,:]<=np.arange(nnode-1)[:,None])&(di[None,:]>np.arange(nnode-1)[:,None])).astype(float);lookup=np.full((nnode,nnode),-1,int);lookup[oi,di]=np.arange(len(oi));triples=np.array(list(itertools.combinations(range(nnode),3)));return Q,lookup,triples,oi,di

def latent_case(oi,di,scenario,noise_factor,rng):
 span=(di-oi)/di.max();sigma=5.;true_sigma=sigma*noise_factor
 if scenario=='matched_conditional':
  b=rng.gamma(2.,5.,len(oi))*np.exp(-3*span)+1.;truth=truncnorm.rvs(-b/true_sigma,np.inf,loc=b,scale=true_sigma,random_state=rng)
 else:
  truth=rng.gamma(2.,5.,len(oi))*np.exp(-3*span)+.05
  if scenario=='near_sparse_censored':truth=np.where(rng.random(len(oi))<.12,truth*5,.02);b=np.maximum(truth+rng.normal(0,true_sigma,len(oi)),1e-6)
  elif scenario=='positive_background':truth=np.where(rng.random(len(oi))<.2,truth*3,.02);b=truth*np.exp(rng.normal(-.25**2/2,.25,len(oi)))+.4*noise_factor*truth.mean()*rng.lognormal(-.7**2/2,.7,len(oi))
  else:s=.4*noise_factor;b=truth*np.exp(rng.normal(0,s,oi.max()+1)[oi]+rng.normal(0,s,di.max()+1)[di]-s*s)
 return truth,b,sigma

def weighted_median(samples):
 weights=1/np.maximum(samples.sum(axis=1),1e-12);order=np.argsort(samples,axis=0);sortedvalues=np.take_along_axis(samples,order,axis=0);sortedweights=weights[order];position=np.argmax(np.cumsum(sortedweights,axis=0)>=.5*weights.sum(),axis=0);return sortedvalues[position,np.arange(samples.shape[1])]

def actions(Q,c,b,draws):
 samples=draws.reshape(-1,len(b));mean=samples.mean(axis=0);median=np.median(samples,axis=0);relative=weighted_median(samples);baseline,st0=base.ridge(Q,c,b);projected,st1=base.ridge(Q,c,median);projectedrelative,st2=base.ridge(Q,c,relative);return dict(weak_prior=(b,True),baseline=(baseline,st0['success']),posterior_mean=(mean,True),median=(median,True),projected_median=(projected,st1['success']),relative_L1_median=(relative,True),projected_relative_L1_median=(projectedrelative,st2['success']))

def main():
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='A converged conditional posterior is not necessarily a correct model; compare loss-motivated mean/median decisions against KNOWN artificial latent flows under matched and misspecified assumptions.',seed=SEED,network_nodes=[6,9],scenarios=SCENARIOS,true_noise_factors=[.5,1.,2.],replicates=4,case_count=96,chains=4,burn_sweeps=200,sample_sweeps=800,assumed_sigma=5.,methods=METHODS,relative_L1_median='Coordinate weighted median with posterior sample weights1/sum(flow), motivated by UNCLIPPED relativeL1; not claimed optimal for clipped/composite score.',matched_control='Generate supplied b first; draw latent f* from independent nonnegative Gaussian centered at b, then c=A*f*. Only factor1 exactly matches assumed posterior. Counterexamples draw sparse/structured f* first and corrupt b with wrong noise.',convergence_gate='Every path and global diagnostics raw split-Rhat<1.05, approximate ESS>=100. Posterior method scores are withheld if gate fails; paired comparisons only share eligible cases and failure rates are reported.',selection='No method/assumption selection on artificial scores. All seven actions and all failure cases retained. This is not real OD validation.',source_sha256=d.sha(__file__),sampler_source_sha256=d.sha(d.WORK/'task4_gaussian_posterior_pilot.py'),scope='Controlled small synthetic networks only; matched-case wins are calibration checks, not independent evidence for real traffic. Official Task4 baseline unchanged, no Kaggle upload.')
 file=O/'protocol.json'
 if file.exists():old=json.loads(file.read_text());protocol['registered_utc']=old['registered_utc'];assert old==protocol
 else:d.save(file,protocol)
 rows=pd.read_csv(O/'all_cases.csv').to_dict('records') if (O/'all_cases.csv').exists() else [];diagnostics=json.loads((O/'convergence_diagnostics.json').read_text()) if (O/'convergence_diagnostics.json').exists() else [];counts=pd.DataFrame(rows).groupby('case').size().to_dict() if rows else {};diagnosed={r['case'] for r in diagnostics};done={case for case,n in counts.items() if n==len(METHODS) and case in diagnosed};rows=[r for r in rows if r['case'] in done];diagnostics=[r for r in diagnostics if r['case'] in done];context={};original_state=d.state
 d.state=lambda **kw:original_state(**{**context,**kw})
 for nodes in protocol['network_nodes']:
  Q,lookup,triples,oi,di=geometry(nodes)
  for scenario in SCENARIOS:
   for factor in protocol['true_noise_factors']:
    for rep in range(protocol['replicates']):
     case=f'n{nodes}/{scenario}/{factor}/{rep}'
     if case in done:continue
     seed=int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8],'little')%(2**63);rng=np.random.default_rng(seed);truth,b,sigma=latent_case(oi,di,scenario,factor,rng);c=Q@truth;context.update(study='smallnet_posterior_stress',case=case);d.state(status='running',stage='smallnet_posterior_case',next='All-path convergence before known-latent synthetic scoring; no real hidden labels or selected methods');pilot.SEED=seed;draws,diag=pilot.sample(Q,c,b,lookup,triples,sigma,1.,protocol);pathdiag=[dict(path_index=i,raw_split_Rhat=pilot.raw_rhat(draws[:,:,i]),approximate_ESS=pilot.rough_ess(draws[:,:,i])) for i in range(len(b))];eligible=bool(diag['ranking_eligible'] and max(z['raw_split_Rhat'] for z in pathdiag)<1.05 and min(z['approximate_ESS'] for z in pathdiag)>=100);diag.update(case=case,all_path_diagnostics=pathdiag,ranking_eligible=eligible);diagnostics.append(diag)
     decisions=actions(Q,c,b,draws) if eligible else {'weak_prior':(b,True),'baseline':base.ridge(Q,c,b)}
     # Convert fallback solver metadata to the same Boolean action flag.
     if not eligible:decisions['baseline']=(decisions['baseline'][0],decisions['baseline'][1]['success'])
     artifact=d.WORK/'task4_smallnet_posterior_cases'/(hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz');artifact.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(artifact,artificial_truth=truth,artificial_prior=b,artificial_counts=c,posterior_samples=draws);digest=d.sha(artifact)
     for name in METHODS:
      pred,success=decisions.get(name,(None,False));row=dict(case=case,nodes=nodes,scenario=scenario,true_noise_factor=factor,replicate=rep,seed=seed,method=name,posterior_converged=eligible,action_available=bool(success),artifact_sha256=digest)
      if pred is not None and success:row.update(base.metrics(pred,truth,b,Q,c,di))
      rows.append(row)
     pd.DataFrame(rows).to_csv(O/'all_cases.csv',index=False);d.save(O/'convergence_diagnostics.json',diagnostics);print('SMALLNET POSTERIOR',case,eligible,flush=True)
 df=pd.DataFrame(rows);scores=['synthetic_S_od','synthetic_S_link','synthetic_S_dev','synthetic_S_attr','synthetic_S_ODME'];df.groupby(['scenario','true_noise_factor','method'])[scores].mean().to_csv(O/'by_assumption.csv');df.groupby('method').agg(total_cases=('case','count'),action_available=('action_available','sum'),converged_cases=('posterior_converged','sum')).to_csv(O/'availability.csv');baseline=df[df.method=='baseline'].set_index('case');risk=[]
 for name in METHODS:
  r=df[df.method==name].set_index('case');good=r.action_available&baseline.action_available&r.posterior_converged;delta=r.loc[good,'synthetic_S_ODME']-baseline.loc[good,'synthetic_S_ODME'];od=r.loc[good,'synthetic_S_od']-baseline.loc[good,'synthetic_S_od'];risk.append(dict(method=name,paired_converged_cases=len(delta),mean_synthetic_score_delta=float(delta.mean()) if len(delta) else None,worst_synthetic_score_delta=float(delta.min()) if len(delta) else None,score_harmed_cases=int((delta<0).sum()),mean_synthetic_OD_delta=float(od.mean()) if len(od) else None,OD_harmed_cases=int((od<0).sum())))
 pd.DataFrame(risk).to_csv(O/'paired_risk.csv',index=False);d.state=original_state;d.state(status='stage_complete',stage='task4_smallnet_posterior_stress_complete',next='Separate matched-model calibration from wrong-assumption harms and convergence failures. Do not recommend real posterior actions from self-model or artificial scores; preserve official Task4 baseline.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_smallnet_posterior_stress',error=repr(e),next='Repair fixed small-network study without deleting failed cases or altering convergence gates, scenarios or method grid.');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
