import argparse,json
from pathlib import Path
import pandas as pd
from task2q.data import PANELS,released_sample
from task2q.submission import validate_queue
from structured_runtime import StructuredPredictor
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--split',choices=['validation','private'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cpu');args=ap.parse_args();model=StructuredPredictor(device=args.device);frames=[]
 for panel in PANELS:
  d=args.data_root/'task2'/panel/args.split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
  for row in index.itertuples():
   s=released_sample(args.data_root,panel,args.split,row.window_id);p=model.predict(s);keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=p[h,keys.link_id.map(ii).to_numpy()];parts.append(keys)
  frame=pd.concat(parts,ignore_index=True);validate_queue(frame,template);frame['panel']=panel;frames.append(frame)
 result=pd.concat(frames,ignore_index=True);assert len(result)==87000;args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False);print(json.dumps(dict(rows=len(result),official_S_queue=None,Task2_only=True)))
if __name__=='__main__':main()
