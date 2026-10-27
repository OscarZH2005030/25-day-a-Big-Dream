"""只下载不可变发布包中的models；标准库实现，校验ZIP并避免Windows深层整包解压。"""
from pathlib import Path
import argparse,urllib.request,hashlib,shutil,zipfile,os
URL='https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/download/best-road-combination-20261026/TrafficFlowBench-Best-20261026.zip'
DIGEST='efbc9223b2d45c131840e6c64f460e90b92a6a70700e0bc6a36d5480e0f2a847'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();cache=root/'release-cache';cache.mkdir(parents=True,exist_ok=True);archive=cache/'TrafficFlowBench-Best-20261026.zip'
 if not archive.exists():
  request=urllib.request.Request(URL,headers={'User-Agent':'TrafficFlowBench-Reproduction'});tmp=archive.with_suffix('.zip.tmp')
  with urllib.request.urlopen(request,timeout=300) as source,tmp.open('wb') as target:shutil.copyfileobj(source,target,4*1024*1024)
  os.replace(tmp,archive)
 h=hashlib.sha256()
 with archive.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 assert h.hexdigest()==DIGEST,'下载ZIP的SHA256不匹配'
 prefix='TrafficFlowBench-Best-20261026/models/';n=0
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   if not info.filename.startswith(prefix) or info.is_dir():continue
   dest=(root/'models'/info.filename[len(prefix):]).resolve();assert dest.is_relative_to(root/'models');dest.parent.mkdir(parents=True,exist_ok=True)
   with z.open(info) as source,dest.open('wb') as target:shutil.copyfileobj(source,target,4*1024*1024)
   n+=1
 assert n;print('PASS: published models extracted, files='+str(n))
if __name__=='__main__':main()
