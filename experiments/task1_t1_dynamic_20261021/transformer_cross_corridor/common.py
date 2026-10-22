from pathlib import Path
from task1.cache import snapshot
from task1.data import sha256
from task1.transformer_priority.common import FROZEN, HERE, frozen_plan, implementation as priority_implementation
from transformer_training.common import AUGMENT

PANELS = ['D12_I5_N', 'D12_I5_S']
VARIANTS = ['T1', 'A_dynamic']
CONFIG = dict(FROZEN, eval_batch=2)
EXPERIMENT = 'd12_i5_t1_dynamic_replication_v1'
MODES = ['rebuild_t0', 'tree_only']


def plan():
    result = frozen_plan()
    result['panels'] = PANELS.copy()
    return result


def dependencies():
    import transformer_training
    import transformer_fusion
    files = [Path(x['path']) for x in priority_implementation()]
    for package in [Path(__file__).parent, Path(transformer_training.__file__).parent, Path(transformer_fusion.__file__).parent]:
        files.extend(package.glob('*.py'))
    return snapshot(files)


def portable_dependencies():
    # Hash names and bytes, so deployment does not invalidate an identical model.
    return sorted([dict(name=Path(x['path']).parent.name + '/' + Path(x['path']).name,
                        sha256=x['sha256']) for x in dependencies()], key=lambda x: x['name'])


def methods(mode):
    if mode not in MODES:
        raise ValueError('An explicit fusion mode is required')
    return (['T0'] if mode == 'rebuild_t0' else []) + ['LightGBM', 'T1', 'Current_fusion', 'A_dynamic', 'A_dynamic_fixed']


def recipes(mode):
    methods(mode)
    return dict(beta=.5 if mode == 'rebuild_t0' else 1., speed=.1, flow=.1)


def validate_config(cfg):
    if (cfg['experiment'] != EXPERIMENT or cfg['panels'] != PANELS or cfg['variants'] != VARIANTS
        or cfg['augmentation'] != AUGMENT or cfg['split'] != plan()
        or cfg['methods'] != methods(cfg['fusion_mode']) or cfg['fixed_fusion'] != recipes(cfg['fusion_mode'])):
        raise ValueError('Frozen experiment protocol changed')
    expected = dict(CONFIG)
    if cfg['smoke']:
        expected.update(epochs=2, steps_per_epoch=4, patience=2)
    if cfg['training'] != expected:
        raise ValueError('Training budget differs')
