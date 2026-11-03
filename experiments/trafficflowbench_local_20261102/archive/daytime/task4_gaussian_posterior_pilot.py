"""Count-conserving Gaussian-prior posterior pilot; convergence before ranking."""
import common as d
import task4_span_shrink_stress as base
import numpy as np,pandas as pd,json,msvcrt,traceback,itertools
from scipy.linalg import qr
from scipy.optimize import linprog
from scipy.stats import truncnorm,kstest
from datetime import datetime,timezone
O=d.OUT/'task4_gaussian_posterior_pilot';SEED=2026110309

def raw_rhat(x):
 m,n=x.shape;n2=n//2;x=np.concatenate([x[:,:n2],x[:,-n2:]],axis=0);W=float(np.var(x,axis=1,ddof=1).mean());B=n2*float(np.var(x.mean(axis=1),ddof=1));return float(np.sqrt(((n2-1)*W/n2+B/n2)/max(W,1e-20)))

def rough_ess(x):
 m,n=x.shape;z=x-x.mean(axis=1,keepdims=True);fft=np.fft.rfft(z,n=2*n,axis=1);ac=np.fft.irfft(fft*np.conj(fft),n=2*n,axis=1)[:,:n]/np.arange(n,0,-1);rho=ac.mean(axis=0)/max(float(ac[:,0].mean()),1e-20);s=0.
 for lag in range(1,n-1,2):
  pair=float(rho[lag]+rho[lag+1])
  if pair<=0:break
  s+=pair
 return float(min(m*n,m*n/max(1+2*s,1)))

def small_check():
 rng=np.random.default_rng(SEED);c0,c1=5.,4.;b=np.array([1.,2.,1.]);sigma=1.3;loc=(b[1]+c0-b[0]+c1-b[2])/3;sd=sigma/np.sqrt(3);a=(0-loc)/sd;bb=(min(c0,c1)-loc)/sd;z0=np.linspace(.1,3.9,12000);xleft=c0-z0;xright=c1-z0;step_loc=((b[1]-z0)-(b[0]-xleft)-(b[2]-xright))/3;t=truncnorm.rvs((-z0-step_loc)/sd,(np.minimum(xleft,xright)-step_loc)/sd,loc=step_loc,scale=sd,random_state=rng);draw=z0+t;stat=kstest(draw,lambda value:truncnorm.cdf(value,a,bb,loc=loc,scale=sd));expected=truncnorm.mean(a,bb,loc=loc,scale=sd);assert abs(draw.mean()-expected)<.03 and stat.statistic<.025;d.save(O/'one_dimensional_check.json',dict(seed=SEED,samples=len(draw),sample_mean=float(draw.mean()),analytic_mean=float(expected),KS_statistic=float(stat.statistic),all_nonnegative=bool((draw>=0).all()),scope='Exact one-dimensional conditional-distribution test, not a hidden-OD check.'))

def initialize(Q,c,lookup,nnode,rng):
 adj=lookup[np.arange(nnode-1),np.arange(1,nnode)];B=Q[:,adj];assert np.all(B.sum(axis=1)==1),'Counted rows are not simple interval cuts';cuts=np.array([np.mean(c[B[:,k]>0]) for k in range(nnode-1)]);assert np.max(np.abs(B@cuts-c))<1e-7;cross=(np.arange(nnode-1)+1)*(nnode-1-np.arange(nnode-1));floor=.02*float(np.min(cuts/cross));interior=np.full(Q.shape[1],floor);interior[adj]+=cuts-floor*cross;assert interior.min()>0 and np.max(np.abs(Q@interior-c))<1e-7;vertices=[]
 for objective in [np.ones(Q.shape[1]),-np.ones(Q.shape[1]),rng.normal(size=Q.shape[1])]:
  result=linprog(objective,A_eq=Q,b_eq=c,bounds=(0,None),method='highs');assert result.success;vertices.append(.98*result.x+.02*interior)
 return np.vstack([interior]+vertices)

def sample(Q,c,mu,lookup,triples,sigma,multiplier,protocol):
 rng=np.random.default_rng(SEED+int(multiplier*100));chains=initialize(Q,c,lookup,lookup.shape[0],rng);nchains=len(chains);sweep_steps=int(np.ceil(Q.shape[1]/3));total=(protocol['burn_sweeps']+protocol['sample_sweeps'])*sweep_steps;adjacent=np.flatnonzero((triples[:,1]-triples[:,0]==1)|(triples[:,2]-triples[:,1]==1));uniform=rng.integers(0,len(triples),size=(total,nchains));local=adjacent[rng.integers(0,len(adjacent),size=(total,nchains))];choices=np.where(rng.random((total,nchains))<.75,local,uniform);samples=[];observables=[];chainindex=np.arange(nchains);sd=sigma/np.sqrt(3);skipped=0;maxdrift=0.;selected=np.linspace(0,Q.shape[1]-1,20,dtype=int)
 for step in range(total):
  node=triples[choices[step]];long=lookup[node[:,0],node[:,2]];left=lookup[node[:,0],node[:,1]];right=lookup[node[:,1],node[:,2]];fl=chains[chainindex,long];fa=chains[chainindex,left];fb=chains[chainindex,right];low=-fl;high=np.minimum(fa,fb);loc=((mu[long]-fl)-(mu[left]-fa)-(mu[right]-fb))/3;usable=high-low>1e-12;move=np.zeros(nchains);skipped+=int((~usable).sum())
  if usable.any():move[usable]=truncnorm.rvs((low[usable]-loc[usable])/sd,(high[usable]-loc[usable])/sd,loc=loc[usable],scale=sd,random_state=rng)
  chains[chainindex,long]+=move;chains[chainindex,left]-=move;chains[chainindex,right]-=move
  if (step+1)%sweep_steps:continue
  sweep=(step+1)//sweep_steps
  if sweep%25==0:
   d.check_deadline();drift=float(np.abs(chains@Q.T-c).max());maxdrift=max(maxdrift,drift);assert chains.min()>-1e-7 and drift<1e-6;d.state(status='running',stage='task4_gaussian_posterior_sampling',sigma_multiplier=multiplier,sweep=sweep,next='Check raw split-Rhat/ESS and conservation; never rank unconverged samples or treat posterior assumptions as hidden truth')
  if sweep>protocol['burn_sweeps']:
   samples.append(chains.copy());D=np.abs(chains-mu).sum(axis=1);logdensity=-np.sum((chains-mu)**2,axis=1)/(2*sigma*sigma);observables.append(np.column_stack([chains.sum(axis=1),D,logdensity,chains[:,selected]]))
 draws=np.transpose(np.array(samples),(1,0,2));obs=np.transpose(np.array(observables),(1,0,2));diag=[]
 for k in range(obs.shape[2]):diag.append(dict(observable=['total_flow','prior_L1','log_density'][k] if k<3 else 'path_'+str(selected[k-3]),raw_split_Rhat=raw_rhat(obs[:,:,k]),approximate_ESS=rough_ess(obs[:,:,k])))
 eligible=bool(max(r['raw_split_Rhat'] for r in diag)<1.05 and min(r['approximate_ESS'] for r in diag)>=100);return draws,dict(sigma_multiplier=multiplier,sigma=sigma,max_count_drift=maxdrift,minimum_draw=float(draws.min()),degenerate_move_fraction=skipped/(total*nchains),ranking_eligible=eligible,diagnostics=diag)

def main():
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),panel='D12_I405_N',split='validation',seed=SEED,chains=4,burn_sweeps=100,sample_sweeps=400,sigma_multipliers=[.5,1.,2.],posterior='exp(-||f-b||^2/(2 sigma^2)) on f>=0 and A*f=c. A Gaussian prior-error assumption, not proved by observed count covariance. No learned global beta is applied to the center.',moves='Random state-independent triangle directions: long(i,k) +t, short(i,j)-t, short(j,k)-t.75% adjacent-split triples,25% uniform triples; exact truncated-normal line conditional. Counted operator must satisfy tested path concatenation identities.',convergence_gate='Raw split-Rhat<1.05 and approximate autocorrelation ESS>=100 for total, priorL1,logdensity and20 fixed path entries. These diagnostics do not establish model correctness.',ranking='No decision ranking unless gate passes. No hidden-reference score is evaluated; model posterior expectations, if later computed, are assumptions only.',source_sha256=d.sha(__file__),scope='Single-panel sampling correctness/mixing pilot, not a new official candidate or full-system validation. Frozen baseline unchanged, no Kaggle uploads.')
 d.save(O/'protocol.json',protocol);small_check();paths,Q,c,b,hashes=base.load(protocol['panel'],protocol['split']);oi=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();di=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();nnode=int(max(di))+1;assert len(paths)==nnode*(nnode-1)//2;lookup=np.full((nnode,nnode),-1,int);lookup[oi,di]=np.arange(len(paths));triples=np.array(list(itertools.combinations(range(nnode),3)));assert np.max(np.abs(Q[:,lookup[triples[:,0],triples[:,2]]]-Q[:,lookup[triples[:,0],triples[:,1]]]-Q[:,lookup[triples[:,1],triples[:,2]]]))==0;_,R,piv=qr(Q.T,mode='economic',pivoting=True);rank=int((np.abs(np.diag(R))>max(abs(R[0,0]),1)*1e-10).sum());idx=piv[:rank];covfile=d.OUT/'task4_covariance_audit/full_train_frozen_models.json';cov=json.loads(covfile.read_text())['models']['path_additive'];loaded=(Q@b)[idx];sigma0=np.sqrt(cov['variance'])*float(np.median(loaded[loaded>0]))/np.sqrt(np.mean(np.sum(Q[idx]**2,axis=1)));d.save(O/'input_sha256.json',dict(files=hashes,covariance_model_sha256=d.sha(covfile),triangle_identities_checked=len(triples),rank=rank,sigma_base=sigma0));summaries=[]
 for multiplier in protocol['sigma_multipliers']:
  d.state(status='running',stage='task4_gaussian_posterior_start',sigma_multiplier=multiplier,next='Single-road four-chain conditional sampler, convergence before interpreting posterior means')
  draws,result=sample(Q,c,b,lookup,triples,sigma0*multiplier,multiplier,protocol);path=d.WORK/'task4_gaussian_posterior_samples'/('sigma'+str(multiplier)+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,samples=draws);result['samples_sha256']=d.sha(path);result['posterior_score_evaluated']=False;summaries.append(result);d.save(O/'diagnostics.json',summaries);print('POSTERIOR PILOT',multiplier,result['ranking_eligible'],max(z['raw_split_Rhat'] for z in result['diagnostics']),flush=True)
 d.state(status='stage_complete',stage='task4_gaussian_posterior_pilot_complete',next='Inspect conservation and mixing diagnostics; no posterior decisions or hidden scores have been ranked. If mixing fails, preserve failure and improve sampler rather than promote biased chain summaries.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_gaussian_posterior_pilot',error=repr(e),next='Repair sampling math or implementation preserving target distribution and pre-registered diagnostics; never claim unconverged samples improve hidden OD.');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
