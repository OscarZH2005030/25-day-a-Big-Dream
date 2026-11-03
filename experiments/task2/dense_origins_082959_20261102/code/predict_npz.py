"""X274/panel/R127 probability NPZ inference, no labels."""
import argparse,json
from pathlib import Path
import numpy as np
from dense_runtime import Portable
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--candidate',choices=['dense_100','dense_50'],default='dense_100');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();c=json.loads((a.root/'profile.json').read_text())['candidates'][a.candidate];rt=Portable(a.root)
with np.load(a.features) as z:
 x=z['X'];labels=z['panel'];parent=z['parent'];p=np.empty(parent.shape,dtype='float32')
 for panel in np.unique(labels):
  ids=np.flatnonzero(labels==panel);p[ids]=rt.predict(str(panel),x[ids],parent[ids],c['alpha'])
t=np.asarray(c['thresholds']);np.savez_compressed(a.output,probability=p,prediction=p>=t,thresholds=t)
