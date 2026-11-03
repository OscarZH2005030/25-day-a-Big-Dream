"""Evaluate whole-road pilot replacements within the fixed ten-road field."""
import sys,json,shutil
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'work'))
import state_joint_proxy as study
SOURCE=ROOT/'outputs/day1/task13-nratio'
OUT=ROOT/'outputs/day1/task13-fullfield';OUT.mkdir(exist_ok=True)
ROADS=['D7_I210_W','D7_I405_S']
class Inputs:
    def __init__(self,arm):self.arm=arm
    def glob(self,pattern):
        original={p.name:p for p in (ROOT/'work/task1-holdout/evaluation/train').glob(pattern)}
        for road in ROADS:
            replacements=list((SOURCE/road/self.arm/'holdout-predictions').glob(pattern));assert len(replacements)==55
            for path in replacements:assert path.name in original;original[path.name]=path
        assert len(original)==550
        return list(original.values())

def main():
    assert (SOURCE/'complete.json').exists()
    protocol=dict(replacements=ROADS,arms=['state_control','joint_Nratio'],selection='Checkpoints frozen on22internaldays before55day reports; no pointwise error routing.',
        evaluation='Actual perday target state SSE and full eligible linkfield FD/truth-anchoredL1dNanalogue; unchangedother8roads, same55days andaggregationasnightbaseline.',
        caveat='This doesnot reproduceorganizerhiddenboundary/rampmasks; all55daysdeveloped. No official total or Kaggle upload.')
    p=ROOT/'work/day1/task13-fullfield-protocol.json';p.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    summaries=[]
    for arm in protocol['arms']:
        folder=OUT/arm;folder.mkdir(exist_ok=True)
        if not (folder/'complete.json').exists():
            study.PRED=Inputs(arm);study.OUT=folder;study.STATE_SCORE_FILE=None;study.PROTOCOL_FILE=p;study.prior.CONFIGS=[(arm,'baseline',0)];study.run()
        frame=pd.read_csv(folder/'summary.csv');summaries.append(frame)
    baseline=pd.read_csv(ROOT/'outputs/round9/summary.csv');baseline=baseline[baseline.method=='baseline'];summaries.append(baseline)
    result=pd.concat(summaries,ignore_index=True);result.to_csv(OUT/'summary.csv',index=False)
    p0=baseline.set_index('scope');delta=result[result.method!='baseline'].copy()
    for col in ['S_state','physics_analogue','partial_total_analogue']:delta['delta_'+col]=delta[col]-delta.scope.map(p0[col])
    delta.to_csv(OUT/'baseline-comparison.csv',index=False);shutil.copy2(p,OUT/'protocol.json');shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    (OUT/'complete.json').write_text(json.dumps(dict(complete=True,official_score=None)),encoding='utf8');print(delta.to_string(index=False),flush=True)

if __name__=='__main__':main()
