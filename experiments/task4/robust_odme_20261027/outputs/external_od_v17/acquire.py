"""Anonymous official OD sources; raw data outside deliverables, hashes recorded."""
from pathlib import Path
import urllib.request,json,hashlib,time,gzip
O=Path(__file__).resolve().parent;ROOT=O.parents[1];W=ROOT/'work/external_od_v17';W.mkdir(parents=True,exist_ok=True)
SOURCES=[('ca_xwalk.csv.gz','https://lehd.ces.census.gov/data/lodes/LODES8/ca/ca_xwalk.csv.gz'),('lodes_ca.sha256sum','https://lehd.ces.census.gov/data/lodes/LODES8/ca/lodes_ca.sha256sum'),('ca_od_main_JT01_2023.csv.gz','https://lehd.ces.census.gov/data/lodes/LODES8/ca/od/ca_od_main_JT01_2023.csv.gz'),('LODESTechDoc.pdf','https://lehd.ces.census.gov/doc/help/onthemap/LODESTechDoc.pdf'),('nhts2017_Csv.zip','https://nhts.ornl.gov/assets/2016/download/Csv.zip')]
def fetch(name,url):
 p=W/name;start=time.time()
 if not p.exists():
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 Task4 research public data'})
  with urllib.request.urlopen(req,timeout=60) as r,p.with_suffix(p.suffix+'.part').open('wb') as f:
   while True:
    b=r.read(1024*1024)
    if not b:break
    f.write(b)
  p.with_suffix(p.suffix+'.part').replace(p)
 raw=p.read_bytes();rec={'file':name,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'anonymous':True,'seconds':time.time()-start}
 pin=O/'pinned_sources.json'
 if pin.exists():
  expected={r['file']:r for r in json.loads(pin.read_text(encoding='utf-8'))}
  if name in expected:assert expected[name]['sha256']==rec['sha256'],('Pinned source changed',name)
 print(name,len(raw),flush=True);return rec
if __name__=='__main__':
 records=[]
 for name,url in SOURCES:
  records.append(fetch(name,url));(O/'sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 checks=(W/'lodes_ca.sha256sum').read_text();verified=[]
 for r in records:
  if r['file'].endswith('.csv.gz'):
   lines=[x for x in checks.splitlines() if x.split()[-1].lstrip('*').split('/')[-1]==r['file'][:-3]];assert len(lines)==1,(r['file'],len(lines));h=hashlib.sha256()
   with gzip.open(W/r['file'],'rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   assert lines[0].split()[0]==h.hexdigest();verified.append({'file':r['file'],'uncompressed_sha256':h.hexdigest()})
 (O/'official_checksum_verification.json').write_text(json.dumps({'verified':verified,'source':'official Census SHA256 file covers uncompressed CSV'},indent=2),encoding='utf-8')
