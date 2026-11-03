"""Verify committed artifact bytes and remote head, append idempotent publication log."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,hashlib,io,json,argparse
ROOT=Path(__file__).resolve().parents[2]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--note',default='Verified research publication, not official score improvement');parser.add_argument('--local-only',action='store_true');args=parser.parse_args();repo=ROOT/'work/daytime/publish-repo';prefix='experiments/trafficflowbench_local_20261102/';entries=[line.split('  ',1) for line in (repo/prefix/'SHA256SUMS.txt').read_text().splitlines()];requests=''.join('HEAD:'+prefix+name+'\n' for expected,name in entries).encode();stream=io.BytesIO(subprocess.check_output(['git','cat-file','--batch'],input=requests,cwd=repo))
 for expected,name in entries:
  header=stream.readline().split();assert len(header)==3 and header[1]==b'blob';payload=stream.read(int(header[2]));assert stream.read(1)==b'\n';assert hashlib.sha256(payload).hexdigest()==expected,name
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 if args.local_only:
  print(json.dumps(dict(commit=head,files_verified=len(entries),exact_committed_bytes=True)));return
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0];assert head==remote,'Remote changed; inspect without force push';path=ROOT/'outputs/daytime/GitHub发布记录.json';history=json.loads(path.read_text()) if path.exists() else dict(repository='https://github.com/OscarZH2005030/25-day-a-Big-Dream',publications=[])
 if not any(r['commit']==head for r in history['publications']):history['publications'].append(dict(utc=datetime.now(timezone.utc).isoformat(),branch='main',commit=head,commit_url='https://github.com/OscarZH2005030/25-day-a-Big-Dream/commit/'+head,study_url='https://github.com/OscarZH2005030/25-day-a-Big-Dream/tree/main/experiments/trafficflowbench_local_20261102',committed_files_verified=len(entries),exact_bytes_verified=True,remote_head_verified=True,raw_competition_observations_uploaded=False,kaggle_submissions=0,note=args.note));path.write_text(json.dumps(history,indent=2,ensure_ascii=False),encoding='utf8')
 print(json.dumps(dict(commit=head,files_verified=len(entries),remote_verified=True)))
if __name__=='__main__':main()
