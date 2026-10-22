from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import os
import subprocess
import sys
import pandas as pd
from task1.cache import snapshot
from task1.data import save_json,sha256
from task1.transformer_priority.common import read,upstream,versions
from transformer_training.train import write_pointer
from transformer_fusion.source import verify
from .common import (BATCHES,BATCH_NAMES,QUEUE_EXPERIMENT,CONFIG,AUGMENT,dependencies,
                     selected_batches,recipes,plan)
from .engine import engine_for
from .logging import append_summary_log
from .summary import old_sources,anchor_sources,summarize_queue,summarize_smoke,aggregate_ten,verify_source


def build_config(args):
    specs=old_sources(args.project)
    training=dict(CONFIG)
    if args.smoke: training.update(epochs=2,steps_per_epoch=4,patience=2)
    return dict(experiment=QUEUE_EXPERIMENT,action=args.action,project=str(args.project),
        release_root=str(args.release_root),cache_dir=str(args.cache_dir),device=args.device,smoke=args.smoke,
        batches=selected_batches(args.batches),training=training,augmentation=AUGMENT,
        split=plan(),fixed_fusion=recipes('rebuild_t0'),implementation=dependencies(),versions=versions(),
        old_sources=specs,old_source_anchors=anchor_sources(specs))


def validate_queue_config(cfg):
    names=[b['name'] for b in cfg['batches']]
    expected=dict(CONFIG)
    if cfg['smoke']: expected.update(epochs=2,steps_per_epoch=4,patience=2)
    if (cfg['experiment']!=QUEUE_EXPERIMENT or cfg['batches']!=selected_batches(names)
        or cfg['training']!=expected or cfg['augmentation']!=AUGMENT or cfg['split']!=plan()
        or cfg['fixed_fusion']!=recipes('rebuild_t0') or cfg['action'] not in ['run','prepare']):
        raise ValueError('Frozen queue protocol differs')


def child_command(args,batch,folder,resume):
    cmd=[sys.executable,'-u','-B','-m','transformer_remaining','batch','--batch',batch['name'],
         '--batch-action',args.action,'--project',str(args.project),'--release-root',str(args.release_root),
         '--cache-dir',str(args.cache_dir),'--output',str(folder),'--device',args.device]
    if args.smoke: cmd.append('--smoke')
    if resume: cmd.append('--resume')
    return cmd


def spawn_child(args,batch,folder,resume):
    env=os.environ.copy()
    package_root=str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH']=os.pathsep.join([package_root,str(args.project),env.get('PYTHONPATH','')])
    env['PYTHONIOENCODING']='utf-8'
    process=subprocess.Popen(child_command(args,batch,folder,resume),cwd=package_root,env=env)
    try:
        code=process.wait()
    except KeyboardInterrupt:
        # Ctrl+C normally reaches parent and child in the same Windows console.
        try:
            process.wait(timeout=10)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        raise
    if code:
        raise RuntimeError(f'Batch {batch["name"]} exited with code {code}; queue stopped')


def inspect_child(folder,batch,prepare=False):
    engine=engine_for(batch['name'])
    if not prepare:
        cfg,_=engine.verify_complete(folder)
        return cfg
    cfg=read(folder/'run_config.json')
    engine.validate_config(cfg)
    if cfg['action']!='prepare' or read(folder/'status.json')['stage']!='prepared':
        raise ValueError('Input preparation incomplete')
    verify(cfg['implementation']);verify(read(folder/'source_manifest.json'))
    preflight=read(folder/'preflight.json')
    if not preflight['passed'] or preflight['holdout_labels_read'] or preflight['formal_training_started']:
        raise ValueError('Preparation phase touched training or holdout labels')
    for panel in batch['panels']:
        info=preflight['panels'][panel]
        if info['holdout_labels_read']:
            raise ValueError('Preparation read holdout labels')
        if not cfg['smoke']:
            for role,field in [('training','training_dates'),('internal','internal_validation_dates'),('holdout','holdout_dates')]:
                if info['dates'][role]!=cfg['split'][field]:
                    raise ValueError('Prepared dates differ')
    return cfg


def child_finished(folder,prepare):
    if prepare:
        return (folder/'status.json').exists() and read(folder/'status.json').get('stage')=='prepared'
    return (folder/'COMPLETE.json').exists()


def root_manifest(out):
    excluded={'execution.lock','status.json','queue_progress.json','artifact_manifest.json','COMPLETE.json',
              'SMOKE_COMPLETE.json','PREPARED.json','BATCHES_COMPLETE.json','log_update.json'}
    files=[p for p in out.iterdir() if p.is_file() and p.name not in excluded]
    files += [p for p in (out/'batches').glob('*/*') if p.name in ['COMPLETE.json','artifact_manifest.json']]
    return snapshot(files)


def verify_queue_complete(out,deep=True):
    cfg=read(out/'queue_config.json')
    validate_queue_config(cfg)
    if cfg['smoke'] or cfg['action']!='run' or [b['name'] for b in cfg['batches']]!=BATCH_NAMES:
        raise ValueError('Not a full formal queue')
    complete=read(out/'COMPLETE.json')
    if not complete['integrity_passed'] or complete['experiment']!=QUEUE_EXPERIMENT:
        raise ValueError('Full queue incomplete')
    if sha256(out/'artifact_manifest.json')!=complete['artifact_manifest_sha256']:
        raise ValueError('Queue manifest changed')
    verify(read(out/'artifact_manifest.json'));verify(cfg['implementation']);verify(cfg['old_source_anchors'])
    sources=read(out/'summary_sources.json')
    verify(sources['anchors'])
    if deep:
        for spec in sources['specs']: verify_source(spec)
    integrity=read(out/'summary_integrity.json')
    if not integrity['passed'] or integrity['panels']!=10 or integrity['families']!=5 or not integrity['all_sources_formal']:
        raise ValueError('Ten-direction scope incomplete')
    ref=pd.read_csv(Path(cfg['project'])/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    rebuilt=aggregate_ten(pd.read_csv(out/'ten_direction_metrics_by_panel_regime.csv'),ref).set_index('method')
    table=pd.read_csv(out/'ten_direction_comparison.csv')
    if table.method.duplicated().any() or set(table.method)!=set(rebuilt.index):
        raise ValueError('Ten-direction methods differ')
    for r in table.itertuples(index=False):
        for field in ['S_state','delta_vs_T0','delta_vs_T1','delta_vs_current_fusion','delta_vs_historical']:
            if abs(getattr(r,field)-rebuilt.loc[r.method,field])>1e-12:
                raise ValueError('Ten-direction aggregate differs')
    return cfg,table


def finish(out,project,cfg):
    if cfg['action']=='prepare':
        save_json(out/'PREPARED.json',dict(passed=True,formal_training_started=False,holdout_labels_read=False,
            batches=[b['name'] for b in cfg['batches']]))
        return 'prepared'
    if cfg['smoke']:
        summarize_smoke(out,cfg)
        save_json(out/'SMOKE_COMPLETE.json',dict(passed=True,formal_training_started=False,formal_ten_direction_score=False,
            batches=[b['name'] for b in cfg['batches']]))
        return 'smoke_completed'
    if [b['name'] for b in cfg['batches']]!=BATCH_NAMES:
        save_json(out/'BATCHES_COMPLETE.json',dict(passed=True,scope='Selected formal batches only',
            batches=[b['name'] for b in cfg['batches']],ten_direction_complete=False))
        return 'selected_batches_completed'
    summarize_queue(out,project,cfg)
    verify(cfg['implementation'])
    manifest=root_manifest(out)
    save_json(out/'artifact_manifest.json',manifest)
    save_json(out/'COMPLETE.json',dict(experiment=QUEUE_EXPERIMENT,integrity_passed=True,panels=10,families=5,
        artifact_manifest_sha256=sha256(out/'artifact_manifest.json')))
    _,table=verify_queue_complete(out,deep=False)
    append_summary_log(out,project,table)
    print(table.to_string(index=False),flush=True)
    return 'completed'


def run_queue(args):
    out=args.output
    if out==args.cache_dir or out.is_relative_to(args.cache_dir) or args.cache_dir.is_relative_to(out):
        raise ValueError('Output and cache must be separate')
    if out.is_relative_to(args.release_root) or args.cache_dir.is_relative_to(args.release_root):
        raise ValueError('Cannot write into released inputs')
    cfg=build_config(args)
    validate_queue_config(cfg)
    for spec in cfg['old_sources']:
        protected=Path(spec['path']).resolve()
        for target in [out,args.cache_dir]:
            if target.is_relative_to(protected) or protected.is_relative_to(target):
                raise ValueError('Queue output/cache overlaps a completed source experiment')
    if out.exists() and not (out/'queue_config.json').exists():
        if any(p.name!='execution.lock' for p in out.iterdir()):
            raise ValueError('Use an empty queue output or the original top-level queue directory')
    out.mkdir(parents=True,exist_ok=True)
    with upstream().lock(out/'execution.lock'):
        if (out/'queue_config.json').exists():
            if not args.resume or read(out/'queue_config.json')!=cfg:
                raise ValueError('Existing queue/config changed; unchanged -Resume or new output required')
            if (out/'COMPLETE.json').exists():
                _,table=verify_queue_complete(out)
                print(append_summary_log(out,args.project,table),flush=True)
                return 0
        save_json(out/'queue_config.json',cfg)
        completed=[]
        current=None
        def status(**values):
            write_pointer(out/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**values))
        try:
            for batch in cfg['batches']:
                current=batch['name']
                verify(cfg['implementation']);verify(cfg['old_source_anchors'])
                folder=out/'batches'/batch['folder']
                status(stage='batch',batch=current,panels=batch['panels'],completed_batches=completed,
                       child_status=str(folder/'status.json'))
                prepare=args.action=='prepare'
                if child_finished(folder,prepare):
                    inspect_child(folder,batch,prepare)
                    print('[queue verified and skipped]',current,flush=True)
                else:
                    resume=(folder/'run_config.json').exists()
                    spawn_child(args,batch,folder,resume)
                    inspect_child(folder,batch,prepare)
                completed.append(current)
                write_pointer(out/'queue_progress.json',dict(completed_batches=completed))
            status(stage='summarizing' if args.action=='run' else 'finishing_preparation',completed_batches=completed)
            stage=finish(out,args.project,cfg)
            status(stage=stage,completed_batches=completed,formal_training_started=not args.smoke and args.action=='run')
            print('Queue:',stage,out,flush=True)
            return 0
        except KeyboardInterrupt:
            status(stage='interrupted',batch=current,completed_batches=completed,
                resume_output=str(out),message='Resume the same queue; unfinished epoch/stage is repeated')
            print('Queue interrupted. Resume with the same output:',out,flush=True)
            return 130
        except Exception as error:
            status(stage='failed',batch=current,completed_batches=completed,error=repr(error))
            raise


def postprocess(args):
    out=args.output
    with upstream().lock(out/'execution.lock'):
        cfg=read(out/'queue_config.json')
        validate_queue_config(cfg)
        if Path(cfg['project'])!=args.project or cfg['smoke'] or cfg['action']!='run':
            raise ValueError('A full formal queue is required')
        verify(cfg['implementation']);verify(cfg['old_source_anchors'])
        if args.action=='log':
            _,table=verify_queue_complete(out)
            print(append_summary_log(out,args.project,table),flush=True)
        else:
            if [b['name'] for b in cfg['batches']]!=BATCH_NAMES:
                raise ValueError('All three formal batches are required')
            for batch in cfg['batches']:
                inspect_child(out/'batches'/batch['folder'],batch)
            finish(out,args.project,cfg)
            write_pointer(out/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),stage='completed',
                completed_batches=BATCH_NAMES))
