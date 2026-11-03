"""Exact count-feasible total-demand bounds; diagnostic extreme solutions only."""
import common as d
import numpy as np,pandas as pd,json,msvcrt,traceback
from scipy import sparse
from scipy.optimize import linprog
from datetime import datetime,timezone
O=d.OUT/'task4_demand_bounds'

def main():
 d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Saturated link fit can leave large uncertainty in number of trips and mean trip length. Quantify exact feasible bounds, rather than tuning link error.',splits=['train','validation','private'],optimization='min/max sum(f) subject released counted-link A*f=c and f>=0; no weak-prior bound imposed. Report infeasibility without silently replacing c.',extremes='Artificial feasible endpoints, never recommended submissions or hidden-score estimates',source_sha256=d.sha(__file__),scope='No hidden path labels, no candidate replacement, no Kaggle upload. Full Task4 and S_od/S_dev/S_attr remain unknown.'))
 rows=[];hashes=[]
 for panel in d.night.predictor.load_routes():
  net=d.DATA/'corridors'/panel/'network';paths=pd.read_csv(net/'path_set.csv',dtype=str);inc=pd.read_csv(net/'path_link_incidence.csv',dtype=str);links=pd.read_csv(net/'links.csv',dtype={'link_id':str}).set_index('link_id');pi={x:i for i,x in enumerate(paths.path_id)};seqs=paths.link_seq.str.split(';');lengths=np.array([sum(float(links.loc[x,'length_km']) for x in seq if x in links.index) for seq in seqs]);complete=all(all(x in links.index for x in seq) for seq in seqs)
  for source in [net/'path_set.csv',net/'path_link_incidence.csv',net/'links.csv']:hashes.append(dict(path=str(source),sha256=d.sha(source)))
  for split in ['train','validation','private']:
   d.check_deadline();d.state(status='running',stage='task4_demand_feasible_bounds',panel=panel,split=split,next='Exact public count-feasible trip-total bounds and length tradeoff; do not turn extremes into a submission')
   folder=d.DATA/'task4'/panel/split;cf=pd.read_csv(folder/'synthetic_link_counts.csv',dtype={'link_id':str});bf=pd.read_csv(folder/'synthetic_weak_prior.csv',dtype={'path_id':str}).set_index('path_id').reindex(paths.path_id);li={x:i for i,x in enumerate(cf.link_id)};matched=inc[inc.link_id.isin(li)];A=sparse.coo_matrix((np.ones(len(matched)),(matched.link_id.map(li).to_numpy(),matched.path_id.map(pi).to_numpy())),shape=(len(cf),len(paths))).tocsr();A.data[:]=1;c=cf['count'].to_numpy(float);b=bf.path_flow.to_numpy(float);row=dict(panel=panel,split=split,paths=len(paths),counted_links=len(cf),prior_total=float(b.sum()),complete_path_lengths=complete)
   low=linprog(np.ones(len(paths)),A_eq=A,b_eq=c,bounds=(0,None),method='highs');high=linprog(-np.ones(len(paths)),A_eq=A,b_eq=c,bounds=(0,None),method='highs');row.update(min_success=bool(low.success),max_success=bool(high.success),min_status=low.message,max_status=high.message)
   if low.success and high.success:
    fmin=low.x;fmax=high.x;lo=float(fmin.sum());hi=float(fmax.sum());den=max(float(c.sum()),1e-9);errlo=float(np.abs(A@fmin-c).sum()/den);errhi=float(np.abs(A@fmax-c).sum()/den);assert errlo<1e-7 and errhi<1e-7 and fmin.min()>-1e-7 and fmax.min()>-1e-7
    row.update(min_total=lo,max_total=hi,max_min_ratio=hi/lo,min_count_NMAE=errlo,max_count_NMAE=errhi,min_solution_prior_L1=float(np.abs(fmin-b).sum()),max_solution_prior_L1=float(np.abs(fmax-b).sum()),prior_relative_position=(float(b.sum())-lo)/max(hi-lo,1e-9),min_total_mean_path_km=float(lengths@fmin/lo) if complete else None,max_total_mean_path_km=float(lengths@fmax/hi) if complete else None,min_total_vehicle_km=float(lengths@fmin) if complete else None,max_total_vehicle_km=float(lengths@fmax) if complete else None,hidden_S_od=None,hidden_S_dev=None,hidden_S_attr=None)
    dest=paths.destination_zone;lowattr=pd.Series(fmin).groupby(dest).sum();highattr=pd.Series(fmax).groupby(dest).sum();row['extreme_attraction_TV']=float(.5*np.abs(lowattr/lo-highattr/hi).sum());witness=d.WORK/'task4_demand_bounds'/panel/split;witness.mkdir(parents=True,exist_ok=True);pd.DataFrame(dict(path_id=paths.path_id,min_total_extreme=fmin,max_total_extreme=fmax)).to_parquet(witness/'diagnostic_extremes_NOT_A_SUBMISSION.parquet',index=False);row['witness_sha256']=d.sha(witness/'diagnostic_extremes_NOT_A_SUBMISSION.parquet')
   rows.append(row);pd.DataFrame(rows).to_csv(O/'bounds.csv',index=False)
   for source in [folder/'synthetic_link_counts.csv',folder/'synthetic_weak_prior.csv']:hashes.append(dict(path=str(source),sha256=d.sha(source)))
   d.save(O/'input_sha256.json',hashes);print('DEMAND BOUNDS',panel,split,row.get('min_total'),row.get('max_total'),row.get('max_min_ratio'),flush=True)
 d.state(status='stage_complete',stage='task4_demand_bounds_complete',next='Analyze total-demand/mean-trip-length ambiguity and prior placement; investigate structured weak prior corrections without treating count-fit as hidden success.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='task4_demand_bounds',error=repr(e),next='Repair bound audit preserving exact public counts, nonnegativity and all failures');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
