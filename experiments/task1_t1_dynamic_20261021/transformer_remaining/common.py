from pathlib import Path
from task1.cache import snapshot
from task1.transformer_priority.common import frozen_plan
from transformer_cross_corridor.common import CONFIG, AUGMENT, VARIANTS, dependencies as previous_dependencies

BATCHES = [
    dict(name='i210', family='D7_I210', folder='01_D7_I210', panels=['D7_I210_E','D7_I210_W']),
    dict(name='i10', family='D7_I10', folder='02_D7_I10', panels=['D7_I10_E','D7_I10_W']),
    dict(name='i405', family='D12_I405', folder='03_D12_I405', panels=['D12_I405_N','D12_I405_S']),
]
PANELS = [p for b in BATCHES for p in b['panels']]
BATCH_NAMES = [b['name'] for b in BATCHES]
EXPERIMENT = 'remaining_t1_dynamic_batch_v1'
QUEUE_EXPERIMENT = 'remaining_t1_dynamic_queue_v1'
SUMMARY_METHODS = ['T0','T1','Current_fusion','A_dynamic','A_dynamic_fixed']


def batch_for(name):
    b=next(b for b in BATCHES if b['name']==name)
    return dict(b,panels=b['panels'].copy())


def selected_batches(names):
    expected=[b['name'] for b in BATCHES if b['name'] in names]
    if list(names)!=expected or not expected:
        raise ValueError('Choose unique batches in the fixed i210, i10, i405 order')
    return [batch_for(n) for n in names]


def plan(panels=None):
    result=frozen_plan()
    result['panels']=list(PANELS if panels is None else panels)
    return result


def methods(mode):
    if mode!='rebuild_t0':
        raise ValueError('Only the frozen 45/45/10 fusion is authorized for this experiment')
    return ['T0','LightGBM','T1','Current_fusion','A_dynamic','A_dynamic_fixed']


def recipes(mode):
    methods(mode)
    return dict(beta=.5,speed=.1,flow=.1)


def dependencies():
    from task1.transformer_priority.common import HERE
    static=[HERE/n for n in ['reference_split.json','reference_feature_names.json','reference_routing_counts.csv','upstream_sha256.json']]
    return snapshot([Path(x['path']) for x in previous_dependencies()]+list(Path(__file__).parent.glob('*.py'))+static)


def portable_dependencies():
    return sorted([dict(name=Path(x['path']).parent.name+'/'+Path(x['path']).name,sha256=x['sha256'])
                   for x in dependencies()],key=lambda x:x['name'])


def validate_config(cfg):
    if cfg['panels'] not in [b['panels'] for b in BATCHES]:
        raise ValueError('Unexpected batch direction scope')
    if (cfg['experiment']!=EXPERIMENT or cfg['variants']!=VARIANTS or cfg['augmentation']!=AUGMENT
        or cfg['split']!=plan(cfg['panels']) or cfg['methods']!=methods(cfg['fusion_mode'])
        or cfg['fixed_fusion']!=recipes(cfg['fusion_mode'])):
        raise ValueError('Frozen batch protocol differs')
    expected=dict(CONFIG)
    if cfg['smoke']:
        expected.update(epochs=2,steps_per_epoch=4,patience=2)
    if cfg['training']!=expected:
        raise ValueError('Training budget differs')
