"""Verify the sealed payload. Local reports/new_runs are outside the manifest."""
import hashlib,json,sys,time
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'HANDOFF_MANIFEST.json').read_text(encoding='utf-8'))
bad=[];start=time.time()
for index,item in enumerate(manifest['files'],1):
    relative=PurePosixPath(item['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
    p=ROOT.joinpath(*relative.parts)
    if not p.is_file() or p.stat().st_size!=item['bytes']:bad.append({'path':item['path'],'reason':'missing or wrong size'});continue
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=item['sha256']:bad.append({'path':item['path'],'reason':'SHA256 mismatch'})
    if index%3000==0:print(f'Checked {index}/{len(manifest["files"])} files',flush=True)
report={'passed':not bad,'files':len(manifest['files']),'bytes':manifest['payload_bytes'],'mismatches':bad,'seconds':round(time.time()-start,2)}
folder=ROOT/'local_reports';folder.mkdir(exist_ok=True)
(folder/'integrity_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if report['passed'] else 1)
