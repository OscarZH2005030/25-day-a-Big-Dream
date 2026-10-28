"""Audit matching before transferring observed OD/trip structure."""
import sys,json
from pathlib import Path
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));from run_task4 import REL
def lengths(panel,paths):
 links=pd.read_csv(REL/'corridors'/panel/'network/links.csv').set_index('link_id').length_km.to_dict();seq=paths.link_seq.str.split(';');known=np.array([all(l in links for l in s) for s in seq]);km=np.array([sum(links[l] for l in s) if ok else np.nan for s,ok in zip(seq,known)]);return known,km
if __name__=='__main__':
 rows=[]
 for d in sorted((REL/'corridors').iterdir()):
  p=pd.read_csv(d/'network/path_set.csv');known,km=lengths(d.name,p);columns={f.name:pd.read_csv(f,nrows=0).columns.tolist() for f in (d/'network').glob('*.csv')};geo=[f for f,c in columns.items() if any(x.lower() in ['latitude','longitude','lat','lon','geometry','x_coord','y_coord'] for x in c)];rows.append(dict(panel=d.name,paths=len(p),unique_OD_pairs=len(p[['origin_zone','destination_zone']].drop_duplicates()),complete_length_paths=int(known.sum()),missing_length_paths=int((~known).sum()),geographic_csv_files=';'.join(geo),verified_geographic_OD_zone_matches=0))
 pd.DataFrame(rows).to_csv(O/'zone_path_matching_audit.csv',index=False);j=json.loads((REL/'config/synthetic_release_v1.json').read_text());(O/'release_type_correction.json').write_text(json.dumps({'historical_source_rows_in_release':j['traffic_policy']['historical_source_rows_in_release'],'competition_counts':'synthetic released counts; previous reports calling these real count CV should be read as released synthetic count CV','true_external_sources':['Census LODES residence/workplace linkages','NHTS household travel diary data','previous public Caltrans/PeMS external sources'],'zone_match':'none verified; overlapping IDs in a third-party sample were insufficient and showed inconsistent lengths','missing_link_lengths':'do not impute from another road or interpret partial sum as complete distance'},indent=2),encoding='utf-8');print(pd.DataFrame(rows).to_string(index=False))
