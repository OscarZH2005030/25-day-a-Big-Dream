"""只用Python标准库核对模型包清单、SHA256和相对路径，不训练、不联网。"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 manifest=json.loads((ROOT/'package-manifest.json').read_text(encoding='utf8'));failed=[]
 for name,expected in manifest['files'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT):raise ValueError('包内路径越界：'+name)
  if not path.is_file():failed.append((name,'missing'));continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
  if h.hexdigest()!=expected['sha256'] or path.stat().st_size!=expected['bytes']:failed.append((name,'hash_or_size_mismatch'))
 if failed:print(json.dumps(failed,ensure_ascii=False));return 1
 routes=json.loads((ROOT/'selected-road-models.json').read_text(encoding='utf8'))['routes'];assert len(routes)==10
 for route in routes.values():
  for path in list(route['paths'].values())+([route['history']] if 'history' in route else []):
   assert not Path(path).is_absolute() and (ROOT/path).resolve().is_relative_to(ROOT)
 print('PASS: '+str(len(manifest['files']))+' files / 10 road model routes');return 0
if __name__=='__main__':sys.exit(main())
