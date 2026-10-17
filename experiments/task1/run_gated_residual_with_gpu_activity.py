"""Foreground GPU activity companion for the gated-residual pipeline.

Separate constant CUDA workload, parent-death pipe, automatic cleanup.
No CUDA/torch import occurs during --dry-run or --io-self-check.
"""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PIPELINE = HERE/'lgb_gated_residual_v1/run_pipeline.py'
DEFAULT_OUTPUT = HERE/'lgb_gated_residual_v1/runs/all_corridors_time80_20'


def worker(ready, size, io_only=False, import_only=False):
    def parent_gone():
        sys.stdin.buffer.read()
        os._exit(0)
    if io_only:
        threading.Thread(target=parent_gone, daemon=True).start()
        ready.write_text('IPC TEST ONLY; NO CUDA', encoding='utf-8')
        while True: time.sleep(.1)
    import faulthandler
    faulthandler.enable()
    def stage(name, message):
        if ready is not None:
            path = ready.with_name('stage.json'); tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(stage=name, message=message)), encoding='utf-8'); tmp.replace(path)
        print(message, flush=True)
        faulthandler.dump_traceback_later(60, repeat=False)
    # Load NumPy explicitly outside torch._C initialization; previous CPU checks
    # used this order. This is a mitigation, not proof of the intermittent cause.
    stage('numpy_import', 'Loading NumPy first...')
    import numpy
    stage('torch_import', 'NumPy '+numpy.__version__+' loaded. Loading PyTorch...')
    import torch
    stage('thread_setup', 'PyTorch '+torch.__version__+' loaded. Setting CPU thread count...')
    torch.set_num_threads(1)
    if import_only:
        faulthandler.cancel_dump_traceback_later()
        assert not torch.cuda.is_initialized()
        print('IMPORT CHECK PASSED; no CUDA initialization, GPU activity or training.', flush=True)
        return
    # As in the previously successful helper, start the pipe reader only after
    # native libraries finish loading. Launcher cleanup kills a timed-out import.
    threading.Thread(target=parent_gone, daemon=True).start()
    stage('cuda_initialization', 'Libraries loaded. Initializing CUDA and running first kernel...')
    with torch.inference_mode():
        a = torch.full((size, size), .01, dtype=torch.float32, device='cuda')
        b = torch.empty_like(a)
        torch.mm(a, a, out=b); torch.cuda.synchronize()
        faulthandler.cancel_dump_traceback_later()
        print('GPU activity ready:', torch.cuda.get_device_name(0), flush=True)
        print(f'Continuous {size}x{size} matrix kernels; one outstanding kernel at a time.', flush=True)
        tmp = ready.with_suffix('.tmp'); tmp.write_text('CUDA READY', encoding='utf-8'); tmp.replace(ready)
        path = ready.with_name('stage.json'); tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(stage='running', message='GPU activity running')), encoding='utf-8'); tmp.replace(path)
        iterations, heartbeat = 0, time.monotonic()
        while True:
            torch.mm(a, a, out=b)
            torch.cuda.synchronize()
            iterations += 1
            if time.monotonic()-heartbeat >= 60:
                if not torch.isfinite(b).all().item(): raise RuntimeError('Nonfinite GPU activity result')
                print('GPU activity heartbeat; kernels=', iterations, flush=True)
                heartbeat = time.monotonic()


class WatchedProcess:
    """Hold a Windows process handle so PID reuse cannot extend the watch."""
    def __init__(self, pid):
        if os.name != 'nt': raise RuntimeError('--watch-pid is supported on Windows only')
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not self.handle: raise OSError(ctypes.get_last_error(), 'Cannot watch PID '+str(pid))
    def running(self):
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == 0xffffffff: raise OSError(ctypes.get_last_error(), 'Process wait failed')
        return result == 258  # WAIT_TIMEOUT, process still running
    def close(self):
        if self.handle: self.kernel.CloseHandle(self.handle); self.handle = None


def queue_locked(output):
    path = output/'queue.lock'
    if not path.exists(): return False
    if os.name != 'nt': return False
    import msvcrt
    with path.open('r+b') as stream:
        stream.seek(0)
        try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError: return True
        stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return False


def environment():
    env = os.environ.copy(); env['PYTHONUTF8'] = '1'; env['PYTHONUNBUFFERED'] = '1'
    return env


def stop_worker(proc):
    if proc is None: return
    if proc.stdin:
        try: proc.stdin.close()
        except OSError: pass
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=5)


def io_self_check():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='gpu_ipc_test_', dir=HERE) as name:
        folder = Path(name)
        assert folder.resolve().is_relative_to(HERE.resolve())
        ready = folder/'ready.txt'
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--io-worker', str(ready)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment())
        watcher = WatchedProcess(proc.pid) if os.name == 'nt' else None
        try:
            deadline = time.monotonic()+15
            while not ready.exists() and proc.poll() is None and time.monotonic() < deadline: time.sleep(.05)
            if not ready.exists(): raise RuntimeError('IPC readiness test failed')
            if watcher is not None and not watcher.running(): raise RuntimeError('Process handle watch failed')
            proc.stdin.close(); proc.wait(timeout=5)
            if proc.returncode != 0: raise RuntimeError('IPC shutdown test failed')
            if watcher is not None and watcher.running(): raise RuntimeError('Exited process incorrectly reported alive')
        finally:
            stop_worker(proc)
            if watcher is not None: watcher.close()
        if os.name == 'nt':
            import msvcrt
            (folder/'queue.lock').write_bytes(b'0')
            with (folder/'queue.lock').open('r+b') as lock:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    if not queue_locked(folder): raise RuntimeError('Active queue lock was missed')
                finally:
                    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            if queue_locked(folder): raise RuntimeError('Released lock reported busy')
    print('PASS: readiness, pipe-EOF shutdown, process-handle watch and queue lock; no torch, CUDA, training or fitting.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--watch-pid', type=int, help='Attach activity only; never launch/stop the existing trainer')
    p.add_argument('--startup-timeout', type=float, default=300)
    p.add_argument('--matrix-size', type=int, default=512)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--io-self-check', action='store_true')
    p.add_argument('--import-check', action='store_true', help='Import NumPy/PyTorch in worker order, without CUDA or training')
    p.add_argument('--activity-worker', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--io-worker', type=Path, help=argparse.SUPPRESS)
    p.add_argument('pipeline_args', nargs=argparse.REMAINDER, help='Additional pipeline hyperparameters after --')
    args = p.parse_args()
    if args.activity_worker: worker(args.activity_worker, args.matrix_size); return 0
    if args.io_worker: worker(args.io_worker, 0, io_only=True); return 0
    if args.io_self_check: io_self_check(); return 0
    if args.import_check: worker(None, 0, import_only=True); return 0
    if args.startup_timeout <= 0 or args.matrix_size < 32 or args.matrix_size > 1024:
        p.error('startup timeout must be positive; matrix size must be 32..1024')
    if args.watch_pid is not None and args.watch_pid <= 0: p.error('watch PID must be positive')
    extra = args.pipeline_args
    if extra[:1] == ['--']: extra = extra[1:]
    for option in extra:
        if option.split('=')[0] in ['--output', '--stage', '--device', '--resume']:
            p.error('Use the wrapper output option; stage/device/resume are fixed to all/cuda/resume')
    command = [sys.executable, '-u', str(PIPELINE), '--stage', 'all', '--device', 'cuda',
               '--output', str(args.output.resolve()), '--resume']+extra
    if args.dry_run:
        print('DRY RUN: no CUDA initialization, subprocess training or GPU activity.')
        print('Watch PID: '+str(args.watch_pid) if args.watch_pid else subprocess.list2cmdline(command))
        return 0
    if not PIPELINE.exists(): raise FileNotFoundError(PIPELINE)
    if not args.watch_pid and queue_locked(args.output):
        print('This output already has a running queue. Use --watch-pid with status.json pid; no second queue was started.', flush=True)
        return 1
    watcher = WatchedProcess(args.watch_pid) if args.watch_pid else None
    if watcher is not None and not watcher.running():
        watcher.close(); print('Watched process has already exited. Nothing started.', flush=True); return 0
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = HERE/'gpu_activity_logs'/('gated_residual_'+stamp); folder.mkdir(parents=True)
    ready, logfile = folder/'ready.txt', folder/'activity.log'
    activity = queue = None
    print('Activity log:', logfile, flush=True)
    try:
        if os.name == 'nt':
            if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
                raise RuntimeError('Windows system-sleep inhibition failed')
        with logfile.open('w', encoding='utf-8') as log:
            activity = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--activity-worker', str(ready), '--matrix-size', str(args.matrix_size)],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, env=environment())
            deadline, next_update, printed = time.monotonic()+args.startup_timeout, time.monotonic()+10, 0
            while not ready.exists():
                content = logfile.read_text(encoding='utf-8', errors='replace')
                if len(content) > printed: print(content[printed:], end='', flush=True); printed = len(content)
                if watcher is not None and not watcher.running(): return 0
                if activity.poll() is not None: raise RuntimeError('GPU activity initialization failed')
                state_path = ready.with_name('stage.json')
                state = json.loads(state_path.read_text(encoding='utf-8'))['stage'] if state_path.exists() else 'worker_startup'
                if time.monotonic() > deadline:
                    detail = 'CUDA was not reached' if state in ['numpy_import', 'torch_import', 'thread_setup', 'worker_startup'] else 'first CUDA operation did not finish'
                    raise RuntimeError(f'GPU helper timed out in stage={state}; {detail}; no new training was started')
                if time.monotonic() >= next_update:
                    print(f'Waiting for GPU helper; stage={state}; {max(0, int(deadline-time.monotonic()))} seconds remain...', flush=True)
                    next_update = time.monotonic()+10
                time.sleep(.2)
            print('GPU activity ready. System automatic sleep inhibited.', flush=True)
            if watcher is None:
                print('Starting the gated-residual pipeline in foreground...', flush=True)
                queue = subprocess.Popen(command, cwd=PIPELINE.parent, env=environment())
            else:
                print('Watching existing PID', args.watch_pid, '; no second queue started.', flush=True)
            warned = False
            while (queue.poll() is None if queue is not None else watcher.running()):
                if activity.poll() is not None and not warned:
                    print('GPU activity exited unexpectedly. Training continues; inspect activity.log.', flush=True)
                    warned = True
                time.sleep(1)
            return queue.returncode if queue is not None else 0
    except KeyboardInterrupt:
        print('Stopping companion'+(' and the queue it launched.' if queue is not None else '; existing training is left running.'), flush=True)
        return 130
    except (OSError, RuntimeError) as error:
        print(str(error), flush=True)
        if logfile.exists(): print('Recent worker log:\n'+logfile.read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
        return 1
    finally:
        # Only terminate a queue this invocation created; never kill an attached PID.
        if queue is not None and queue.poll() is None:
            try: queue.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == 'nt': subprocess.run(['taskkill', '/PID', str(queue.pid), '/T', '/F'], capture_output=True)
                else: queue.terminate()
                try: queue.wait(timeout=5)
                except subprocess.TimeoutExpired: queue.kill(); queue.wait()
        stop_worker(activity)
        if watcher is not None: watcher.close()
        if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__': sys.exit(main())
