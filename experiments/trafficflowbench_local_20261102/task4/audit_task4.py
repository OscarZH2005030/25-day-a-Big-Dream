"""Reproduce public Task4 identifiability, demand bounds and invariant witnesses.

No hidden truth, credentials, Kaggle client or submission code is used.
"""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')

def operator(net,cf):
 paths=pd.read_csv(net/'path_set.csv',dtype=str)
 inc=pd.read_csv(net/'path_link_incidence.csv',dtype=str)
 pi={v:i for i,v in enumerate(paths.path_id)}
 li={v:i for i,v in enumerate(cf.link_id)}
 sel=inc[inc.link_id.isin(li)]
 A=sparse.coo_matrix((np.ones(len(sel)),(sel.link_id.map(li),sel.path_id.map(pi))),shape=(len(cf),len(paths))).tocsr()
 A.data[:]=1
 codes=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
 groups=[sparse.coo_matrix((np.ones(len(paths)),(code,np.arange(len(paths)))),shape=(code.max()+1,len(paths))).tocsr() for code in codes]
 return paths,A,groups

def information(paths,A,groups):
 dense=A.toarray()
 _,s,vh=np.linalg.svd(dense,full_matrices=False)
 rank=int((s>max(s[0],1)*1e-10).sum())
 basis=vh[:rank]
 remainder=[]
 for group in groups:
  G=group.toarray();remainder.append(G-(G@basis.T)@basis)
 extra=int(np.linalg.matrix_rank(np.concatenate(remainder),tol=1e-8))
 return dict(paths=len(paths),distinct_OD_pairs=len(paths[['origin_zone','destination_zone']].drop_duplicates()),count_rank=rank,path_nullity=len(paths)-rank,origin_and_destination_extra_rank=extra,nullity_even_if_all_marginals_known=len(paths)-rank-extra,total_flow_identified=bool(np.linalg.norm(np.ones(len(paths))-(np.ones(len(paths))@basis.T)@basis)<1e-8))

def bounds(A,c):
 low=linprog(np.ones(A.shape[1]),A_eq=A,b_eq=c,bounds=(0,None),method='highs')
 high=linprog(-np.ones(A.shape[1]),A_eq=A,b_eq=c,bounds=(0,None),method='highs')
 row=dict(min_success=bool(low.success),max_success=bool(high.success),min_message=low.message,max_message=high.message)
 if low.success and high.success:
  lo=float(low.x.sum());hi=float(high.x.sum())
  row.update(min_total=lo,max_total=hi,max_min_ratio=hi/lo,min_count_NMAE=float(np.abs(A@low.x-c).sum()/c.sum()),max_count_NMAE=float(np.abs(A@high.x-c).sum()/c.sum()))
 return row

def invariant(A,c,b,base,groups,rng,objectives):
 sign=np.sign(base-b)
 H=sparse.vstack([A,*groups,sparse.csr_matrix(sign[None,:])],format='csr')
 rhs=H@base
 limits=[(float(bj),None) if sj>0 else (0,float(bj)) if sj<0 else (float(bj),float(bj)) for bj,sj in zip(b,sign)]
 d0=float(np.abs(base-b).sum());attr0=groups[1]@base;attr0/=attr0.sum()
 baseline_link=max(0,1-float(np.abs(A@base-c).sum()/c.sum()))
 rows=[]
 for rep in range(objectives):
  result=linprog(rng.normal(size=len(base)),A_eq=H,b_eq=rhs,bounds=limits,method='highs')
  row=dict(replicate=rep,success=bool(result.success),message=result.message)
  if result.success:
   f=result.x;attr=groups[1]@f;attr/=attr.sum()
   row.update(OD_relative_L1_change=float(np.abs(f-base).sum()/base.sum()),maximum_count_change=float(np.abs(A@(f-base)).max()),prior_L1_relative_change=float(abs(np.abs(f-b).sum()-d0)/max(d0,1)),destination_TV_change=float(.5*np.abs(attr-attr0).sum()),baseline_S_link=baseline_link,witness_S_link=max(0,1-float(np.abs(A@f-c).sum()/c.sum())))
   assert row['maximum_count_change']<1e-5 and row['prior_L1_relative_change']<1e-9 and row['destination_TV_change']<1e-9
  rows.append(row)
 return rows

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--release-root',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--panels',nargs='+')
 parser.add_argument('--splits',nargs='+',default=['validation','private'],choices=['train','validation','private'])
 parser.add_argument('--baseline-path-flow',nargs='+',type=Path)
 parser.add_argument('--seed',type=int,default=2026110301)
 parser.add_argument('--objectives',type=int,default=8)
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 allpanels=sorted(p.name for p in (args.release_root/'task4').iterdir() if p.is_dir())
 panels=args.panels or allpanels
 baseline=pd.concat([pd.read_csv(p,dtype={'panel':str,'path_id':str,'departure_time':str}) for p in args.baseline_path_flow],ignore_index=True) if args.baseline_path_flow else None
 manifest={};info=[];bb=[];ww=[]
 for panel in panels:
  for split in args.splits:
   folder=args.release_root/'task4'/panel/split;net=args.release_root/'corridors'/panel/'network'
   cf=pd.read_csv(folder/'synthetic_link_counts.csv',dtype={'link_id':str})
   paths,A,groups=operator(net,cf);c=cf['count'].to_numpy(float)
   info.append(dict(panel=panel,split=split,**information(paths,A,groups)))
   bb.append(dict(panel=panel,split=split,**bounds(A,c)))
   if baseline is not None:
    prior=pd.read_csv(folder/'synthetic_weak_prior.csv',dtype={'path_id':str,'departure_time':str})
    b=prior.set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float)
    selected=baseline[(baseline.panel==panel)&(baseline.departure_time==prior.departure_time.iloc[0])]
    if selected.path_id.duplicated().any():raise ValueError('Duplicate baseline path keys')
    f=selected.set_index('path_id').reindex(paths.path_id).path_flow.to_numpy(float)
    if not np.isfinite(f).all() or (f<0).any():raise ValueError('Missing/invalid baseline rows')
    rng=np.random.default_rng(args.seed+allpanels.index(panel)*100+['validation','private'].index(split)) if split!='train' else np.random.default_rng(args.seed+allpanels.index(panel)*100+2)
    ww.extend(dict(panel=panel,split=split,**row) for row in invariant(A,c,b,f,groups,rng,args.objectives))
   for source in [net/'path_set.csv',net/'path_link_incidence.csv',folder/'synthetic_link_counts.csv',folder/'synthetic_weak_prior.csv']:manifest[str(source.relative_to(args.release_root))]=sha(source)
   pd.DataFrame(info).to_csv(args.output/'identifiability.csv',index=False)
   pd.DataFrame(bb).to_csv(args.output/'bounds.csv',index=False)
   if ww:pd.DataFrame(ww).to_csv(args.output/'witnesses.csv',index=False)
   print(panel,split,bb[-1],flush=True)
 save(args.output/'provenance.json',dict(seed=args.seed,objectives=args.objectives,input_sha256=manifest,baseline_sha256=[sha(p) for p in args.baseline_path_flow or []],source_sha256=sha(__file__),official_score=None,note='Public structural diagnostics only. Alternative feasible flows are not submissions or hidden-score improvements.'))
if __name__=='__main__':main()
