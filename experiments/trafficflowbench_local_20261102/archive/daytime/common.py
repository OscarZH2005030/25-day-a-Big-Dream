import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,hashlib,time,uuid
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'work/daytime';OUT=ROOT/'outputs/daytime';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'work/overnight'))
import state_experiment as night
DATA=night.DATA;KEYS=night.KEYS
# Latest explicit user instruction: continue after breakthroughs and GitHub pushes.
# An optional environment deadline remains available for a bounded manual run.
deadline_setting=os.environ.get('TFB_RESEARCH_DEADLINE_UTC')
DEADLINE=datetime.fromisoformat(deadline_setting.replace('Z','+00:00')) if deadline_setting else None
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
 # Windows readers may temporarily deny replacement while a status view is open.
 # Retry the atomic replace; never truncate the last complete destination file.
 for attempt in range(24):
  try:tmp.replace(p);return
  except PermissionError:
   if attempt==23:raise
   time.sleep(min(.025*2**min(attempt,4),.25))
def state(**kw):save(WORK/'STATE.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),deadline_utc=DEADLINE.isoformat() if DEADLINE else None,continuation='until_user_stops',automation_id='trafficflowbench',github_push_authorized_repository='https://github.com/OscarZH2005030/25-day-a-Big-Dream',kaggle_upload_authorized=False,remaining_opportunities_user_reported=2,**kw))
def check_deadline():
 if DEADLINE is not None and datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('Configured research cutoff reached')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
