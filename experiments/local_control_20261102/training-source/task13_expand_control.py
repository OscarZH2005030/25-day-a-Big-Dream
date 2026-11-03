"""Apply the frozen control recipe to the other six existing Jayden routes."""
import sys,json,shutil,hashlib
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import task13_nratio as study
base=study.base;ROOT=base.ROOT
ROADS=['D12_I405_N','D12_I5_N','D12_I5_S','D7_I10_E','D7_I10_W','D7_I405_N']
OUT=ROOT/'outputs/day1/task13-control-expansion';OUT.mkdir(exist_ok=True)
def main():
    protocol=dict(roads_fixed_before_new_holdout=ROADS,recipe='Exactly same state_control warm-start budget/seed and INNER state+Nratio checkpoint criterion as two-road pilot. Keep initial checkpoint eligible.',
        selection='Train168days,select22internaldays,freeze each checkpoint before preparing55developedholdoutdays. Report allsix roads, not just winners; retainpriorwhole-comboifaggregateworse.',
        justification='Two-road fullfield state_control improved state and conservation analogue; jointNratio loss worsened holdout, so do not expand that arm.',
        caveat='All dates previously developed; whole-road fixed routes, no per-point/per-case routing. No official Sphysics/total and no upload.')
    (OUT/'expansion-protocol-before-holdout.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    routes=json.loads((base.BUNDLE/'selected-road-models.json').read_text())['routes'];m=base.runtime()
    for road in ROADS:
        route=routes[road];assert route['model']=='jayden_186'
        history=base.RUNTIME/'history'/road;history.mkdir(parents=True,exist_ok=True);shutil.copy2(base.BUNDLE/route['paths']['history'],history/'history.npz')
        keys=base.WORK/'task13-shared/cache'/road/'keys.parquet';keys.parent.mkdir(parents=True,exist_ok=True)
        if not keys.exists():pd.read_csv(base.DATA/'task1'/road/'train/sample_submission_state.csv',dtype=str).to_parquet(keys,index=False)
        data=m.Data(road)
        for i,date in enumerate(m.TRAIN+m.INNER):
            data.prepare(date)
            if i%20==0:print(road,'preparing',i+1,190,flush=True)
        shutil.copy2(base.BUNDLE/route['paths']['normalization'],data.folder/'normalization.npz')
        del data
    study.OUT=OUT;study.ARMS=['state_control'];base.ROADS=ROADS;study.main()
    shutil.copy2(Path(__file__),OUT/Path(__file__).name)
if __name__=='__main__':main()
