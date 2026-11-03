import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'work/daytime';OUT=ROOT/'outputs/daytime';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'work/overnight'))
import state_experiment as night
DATA=night.DATA;KEYS=night.KEYS;DEADLINE=datetime(2026,11,3,8,tzinfo=timezone.utc)
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(p)
def state(**kw):save(WORK/'STATE.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),deadline_utc=DEADLINE.isoformat(),automation_id='trafficflowbench',kaggle_upload_authorized=False,remaining_opportunities_user_reported=2,**kw))
def check_deadline():
 if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Daytime cutoff reached')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
