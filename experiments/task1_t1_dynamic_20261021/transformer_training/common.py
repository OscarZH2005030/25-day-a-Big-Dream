from pathlib import Path
from task1.cache import snapshot
from task1.data import sha256
from task1.transformer_priority.common import FROZEN

VARIANTS=['A_dynamic','B_regime']
CONFIG=dict(FROZEN,eval_batch=2)
AUGMENT=dict(hidden_fraction=.10,extra_supervision_fraction=.20,
             patterns=['point','temporal','spatial'],temporal_lengths=[6,12],
             spatial_links=4,spatial_ticks=6)
CONTROLS=['T0','T1','Current_equal','Current_fusion']
METHODS=CONTROLS+['A_dynamic','A_dynamic_fixed','B_regime','B_regime_fixed']
EXPERIMENT='t1_training_ablations_v1'


def code_identity():
    return [dict(name=p.name,sha256=sha256(p)) for p in sorted(Path(__file__).parent.glob('*.py'))]


def implementation():
    return snapshot(Path(__file__).parent.glob('*.py'))
