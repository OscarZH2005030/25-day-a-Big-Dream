import json,time,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
W=Path(__file__).resolve().parent
while datetime.now(timezone.utc)<datetime(2026,11,3,8,tzinfo=timezone.utc):
 s=json.loads((W/'STATE.json').read_text())
 if s.get('stage')=='pseudo_augmentation_ready' and s.get('status')=='stage_complete':
  with (W/'augmentation_fit.log').open('ab') as log:r=subprocess.run([sys.executable,'-X','utf8',str(W/'augmentation_fit.py')],stdout=log,stderr=subprocess.STDOUT)
  raise SystemExit(r.returncode)
 if s.get('status')=='failed_or_cutoff':raise SystemExit('Data generation requires recovery')
 time.sleep(10)
