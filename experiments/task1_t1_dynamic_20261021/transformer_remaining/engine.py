"""Load the immutable two-direction engine into an isolated module per batch."""
import importlib.util
from pathlib import Path
from . import common
from .logging import append_batch_log


def engine_for(name):
    import transformer_cross_corridor.runner as original
    batch=common.batch_for(name)
    source=Path(original.__file__)
    spec=importlib.util.spec_from_file_location('transformer_remaining._engine_'+name,source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PANELS=batch['panels'].copy()
    module.plan=lambda:common.plan(batch['panels'])
    def validate(cfg):
        common.validate_config(cfg)
        if cfg['panels']!=batch['panels']:
            raise ValueError('Result does not belong to requested batch')
    module.validate_config=validate
    def log_results(out,project):
        cfg,table=module.verify_complete(Path(out))
        return append_batch_log(Path(out),Path(project),cfg,table)
    module.log_results=log_results
    return module
