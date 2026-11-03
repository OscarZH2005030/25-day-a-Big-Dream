"""Fixed all-eight control expansion, no per-road holdout selection."""
import json,sys,shutil
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'work'))
import state_joint_proxy as study
OUT=ROOT/'outputs/day1/task13-all8-fullfield';OUT.mkdir(exist_ok=True)
SOURCES={r:ROOT/'outputs/day1/task13-nratio' for r in ['D7_I210_W','D7_I405_S']}
SOURCES.update({r:ROOT/'outputs/day1/task13-control-expansion' for r in ['D12_I405_N','D12_I5_N','D12_I5_S','D7_I10_E','D7_I10_W','D7_I405_N']})
class Inputs:
    def glob(self,pattern):
        original={p.name:p for p in (ROOT/'work/task1-holdout/evaluation/train').glob(pattern)}
        for road,source in SOURCES.items():
            replacement=list((source/road/'state_control/holdout-predictions').glob(pattern));assert len(replacement)==55
            for p in replacement:assert p.name in original;original[p.name]=p
        assert len(original)==550
        return list(original.values())
def main():
    for source in set(SOURCES.values()):assert (source/'complete.json').exists()
    protocol=dict(roads=list(SOURCES),method='All eight predetermined state_control routes, two unchanged original routes. No holdout-based per-road picking.',selection='Same 168 training and 22 internal selection days; initial checkpoint eligible; 55 developed dates report only.',caveat='Physics analogue uses observable truth-anchored change, not official hidden flux/ramp masks. No official score or upload.')
    p=OUT/'protocol-before-fullfield.json';p.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    if not (OUT/'complete.json').exists():
        study.PRED=Inputs();study.OUT=OUT;study.STATE_SCORE_FILE=None;study.PROTOCOL_FILE=p;study.prior.CONFIGS=[('all8_state_control','baseline',0)];study.run()
    result=pd.read_csv(OUT/'summary.csv');base=pd.read_csv(ROOT/'outputs/round9/summary.csv');base=base[base.method=='baseline'].set_index('scope')
    for col in ['S_state','physics_analogue','partial_total_analogue']:result['delta_'+col]=result[col]-result.scope.map(base[col])
    result.to_csv(OUT/'baseline-comparison.csv',index=False);shutil.copy2(Path(__file__),OUT/Path(__file__).name);print(result.to_string(index=False),flush=True)
if __name__=='__main__':main()
