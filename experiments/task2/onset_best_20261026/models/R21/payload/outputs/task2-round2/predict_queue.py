"""冻结Task2模型推理入口；只读发布history和masked T，不读代理未来。"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from task2q.data import PANELS,FEATURE_NAMES,released_sample
from task2q.baselines import tree_features
from task2q.model import QueueForecaster
from task2q.smoke import tensor_input
from task2q.submission import validate_queue
from causal_features import enhanced

class FrozenPredictor:
    def __init__(self,bundle=None,device='cuda'):
        self.root=Path(bundle or Path(__file__).resolve().parent);self.device=device
        self.rule=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'))['selected'];name=self.rule['model']
        self.alpha=float(name.split('-')[1]) if name.startswith('blend-') else (1. if name=='new_neural' else 0.)
        self.neural=None;self.tree=None
        if self.alpha:
            ck=torch.load(self.root/'best.pt',map_location=device,weights_only=False);self.neural=QueueForecaster(len(FEATURE_NAMES),**ck['config']).to(device);self.neural.load_state_dict(ck['model']);self.neural.eval()
        if self.alpha<1:
            self.tree=lgb.Booster(model_file=str(self.root/('fallback-first-round-tree.txt' if name=='old_tree' else 'selected-tree.txt')))
        self.fn=tree_features if name=='old_tree' else enhanced
    @torch.inference_mode()
    def predict(self,s):
        p=0
        if self.neural is not None:p=self.alpha*self.neural(**tensor_input(s,self.device))['logits'].sigmoid()[0].cpu().numpy()
        if self.tree is not None:p=p+(1-self.alpha)*self.tree.predict(self.fn(s,s['network']),num_threads=4).reshape(6,-1)
        return (p>=self.rule['thresholds'][s['condition']]).astype(np.int8)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--split',choices=['validation','private'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args();torch.set_num_threads(4);predictor=FrozenPredictor(device=args.device);frames=[]
    for panel in PANELS:
        directory=args.data_root/'task2'/panel/args.split;index=pd.read_csv(directory/'window_index.csv');template=pd.read_csv(directory/'sample_submission_queue.csv');parts=[]
        for row in index.itertuples():
            s=released_sample(args.data_root,panel,args.split,row.window_id);pred=predictor.predict(s);keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=pred[h,keys.link_id.map(ii).to_numpy()];parts.append(keys)
        frame=pd.concat(parts,ignore_index=True);validate_queue(frame,template);frame['panel']=panel;frames.append(frame)
    result=pd.concat(frames,ignore_index=True);assert len(result)==87000;args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(json.dumps(dict(rows=len(result),official_S_queue=None,Task2_only=True)))
if __name__=='__main__':main()
