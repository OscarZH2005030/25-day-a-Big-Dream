"""What counted links identify about OD/destination structure; no model fitting."""
import common as d
import numpy as np,pandas as pd,json,msvcrt
from datetime import datetime,timezone
O=d.OUT/'task4_information_audit'

def main():
 d.save(O/'protocol.json',dict(utc=datetime.now(timezone.utc).isoformat(),scope='Linear identifiability of released measurement operator, not hidden-score estimation',source_sha256=d.sha(__file__),tolerance=1e-8,split='validation',no_submission=True))
 rows=[]
 for panel in d.night.predictor.load_routes():
  net=d.DATA/'corridors'/panel/'network';paths=pd.read_csv(net/'path_set.csv',dtype=str);inc=pd.read_csv(net/'path_link_incidence.csv',dtype=str);counts=pd.read_csv(d.DATA/'task4'/panel/'validation/synthetic_link_counts.csv',dtype={'link_id':str});ids=paths.path_id.tolist();pi={v:i for i,v in enumerate(ids)};links=counts.link_id.tolist();li={v:i for i,v in enumerate(links)};matched=inc[inc.link_id.isin(li)];A=np.zeros((len(links),len(ids)));A[matched.link_id.map(li).to_numpy(),matched.path_id.map(pi).to_numpy()]=1
  u,s,vh=np.linalg.svd(A,full_matrices=False);rank=int((s>max(s[0],1)*1e-10).sum());basis=vh[:rank];origins=sorted(paths.origin_zone.unique());dest=sorted(paths.destination_zone.unique());Gd=np.array([(paths.destination_zone==z).to_numpy(float) for z in dest]);Go=np.array([(paths.origin_zone==z).to_numpy(float) for z in origins]);D=Gd-(Gd@basis.T)@basis;Q=Go-(Go@basis.T)@basis;combined=np.concatenate([D,Q]);increment=int(np.linalg.matrix_rank(combined,tol=1e-8));row=dict(panel=panel,paths=len(ids),distinct_OD_pairs=len(paths[['origin_zone','destination_zone']].drop_duplicates()),counted_links=len(links),count_rank=rank,path_nullity=len(ids)-rank,origin_zones=len(origins),destination_zones=len(dest),destination_constraints_extra_rank=int(np.linalg.matrix_rank(D,tol=1e-8)),origin_constraints_extra_rank=int(np.linalg.matrix_rank(Q,tol=1e-8)),origin_and_destination_extra_rank=increment,nullity_even_if_all_marginals_known=len(ids)-rank-increment,total_flow_identified=bool(np.linalg.norm(np.ones(len(ids))-(np.ones(len(ids))@basis.T)@basis)<1e-8),path_set_sha256=d.sha(net/'path_set.csv'),incidence_sha256=d.sha(net/'path_link_incidence.csv'),counts_sha256=d.sha(d.DATA/'task4'/panel/'validation/synthetic_link_counts.csv'));rows.append(row);pd.DataFrame(rows).to_csv(O/'identifiability.csv',index=False);print(row,flush=True)
 b=np.full((2,2),250.);truth=np.array([[450.,50.],[50.,450.]]);wrong=np.array([[50.,450.],[450.,50.]])
 def score(f):
  od=max(0,1-np.abs(f-truth).sum()/truth.sum());link=max(0,1-abs(f.sum()-truth.sum())/truth.sum());dev=np.exp(-abs(np.abs(f-b).sum()/np.abs(truth-b).sum()-1));attr=max(0,1-.5*np.abs(f.sum(0)/f.sum()-truth.sum(0)/truth.sum()).sum());return dict(S_od=float(od),S_link=float(link),S_dev=float(dev),S_attr=float(attr),S_ODME=float(.45*od+.25*link+.15*dev+.15*attr))
 d.save(O/'counterexample.json',dict(scope='Toy illustration, not real benchmark data',prior=b.tolist(),truth=truth.tolist(),opposite_matching=wrong.tolist(),true_submission=score(truth),wrong_submission=score(wrong)));d.state(status='stage_complete',stage='task4_information_audit_complete',next='Prioritize hidden OD matching/deviation/attraction structure and identifiability, not squeezing S_link; no upload or artificial official-score claim.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
