"""Bounded CUDA/VRAM stress test. No overclocking, driver changes or network use.
Parent watchdog stops the child on temperature >=86 C, AC loss, monitoring
failure, or stalled progress. This is a compute test, not a 3D rendering test.
"""
import argparse
import csv
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SMI = r"C:\Windows\System32\nvidia-smi.exe"
FIELDS = ["temperature.gpu", "utilization.gpu", "utilization.memory",
          "memory.used", "memory.total", "power.draw", "clocks.current.graphics",
          "clocks.current.memory", "pstate",
          "clocks_throttle_reasons.hw_thermal_slowdown",
          "clocks_throttle_reasons.sw_thermal_slowdown"]


def save(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ac_connected():
    class Status(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte),
                    ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                    ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    result = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(result)):
        raise RuntimeError("Unable to read AC power status")
    return result.ac == 1


def telemetry():
    p = subprocess.run([SMI, "--query-gpu=" + ",".join(FIELDS),
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=8,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    line = next(csv.reader(p.stdout.strip().splitlines()))
    result = dict(zip(FIELDS, [v.strip() for v in line]))
    for key in FIELDS[:8]:
        result[key] = float(result[key])
    result["timestamp"] = dt.datetime.now().astimezone().isoformat()
    result["ac_connected"] = ac_connected()
    return result


def worker(args):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261020)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out = Path(args.out)
    start = time.monotonic()
    checks = 0

    def progress(phase, **kw):
        value = dict(phase=phase, elapsed_seconds=time.monotonic() - start,
                     correctness_checks=checks, **kw)
        save(out / "progress.json", value)
        print(json.dumps(value), flush=True)

    prop = torch.cuda.get_device_properties(0)
    progress("initializing", torch_version=torch.__version__,
             cuda_version=torch.version.cuda, gpu=prop.name,
             total_vram_bytes=prop.total_memory)
    free, total = torch.cuda.mem_get_info()
    chunk_words = 64 * 1024 * 1024  # 256 MiB per int32 buffer
    count = int(min(10 * 1024**3, free * .70) // (chunk_words * 4))
    if count < 4:
        raise RuntimeError("Less than 1 GiB is available for the memory test")
    buffers = [torch.empty(chunk_words, dtype=torch.int32, device="cuda") for _ in range(count)]
    tested_bytes = count * chunk_words * 4
    patterns = [0, -1, 1431655765, -1431655766, 2147483647, -2147483648]
    deadline = time.monotonic() + args.vram_seconds
    rounds = 0
    while time.monotonic() < deadline or rounds < 1:
        for pattern in patterns:
            for i, buf in enumerate(buffers):
                buf.fill_(pattern ^ i)
            for i, buf in enumerate(buffers):
                mismatches = int(torch.count_nonzero(buf != (pattern ^ i)).item())
                checks += 1
                if mismatches:
                    raise RuntimeError(f"VRAM mismatch: buffer={i}, count={mismatches}")
            progress("vram_patterns", rounds=rounds, pattern=pattern,
                     tested_vram_bytes=tested_bytes)
        rounds += 1
    del buf, buffers
    torch.cuda.empty_cache()

    # +/-1 inputs give exactly representable integer sums. Compare a 16x16
    # subset against a CPU reference on every monitoring/verification interval.
    n = args.matrix_size
    a = torch.randint(0, 2, (n, n), device="cuda", dtype=torch.float32).mul_(2).sub_(1)
    b = torch.randint(0, 2, (n, n), device="cuda", dtype=torch.float32).mul_(2).sub_(1)
    c = torch.empty_like(a)
    ah, bh = a.half(), b.half()
    ch = torch.empty_like(ah)
    rows = torch.linspace(0, n - 1, 16, device="cuda").long()
    cols = torch.linspace(0, n - 1, 16, device="cuda").long()
    reference = a[rows].cpu() @ b[:, cols].cpu()
    free, _ = torch.cuda.mem_get_info()
    pad_count = int(min(8 * 1024**3, free * .60) // (chunk_words * 4))
    pads = [torch.full((chunk_words,), 1431655765 ^ i, dtype=torch.int32, device="cuda")
            for i in range(pad_count)]
    progress("compute_start", matrix_size=n, vram_pattern_rounds=rounds,
             tested_vram_bytes=tested_bytes, padding_bytes=pad_count*chunk_words*4)
    deadline = time.monotonic() + args.compute_seconds
    mode_start = time.monotonic()
    mode = "fp32"
    counts = {"fp32": 0, "fp16": 0}
    active_seconds = {"fp32": 0.0, "fp16": 0.0}
    last_report = 0.0
    while time.monotonic() < deadline:
        tick = time.monotonic()
        if tick - mode_start >= 45:
            mode = "fp16" if mode == "fp32" else "fp32"
            mode_start = tick
        aa, bb, cc = (a, b, c) if mode == "fp32" else (ah, bh, ch)
        for _ in range(8):
            torch.mm(aa, bb, out=cc)
        torch.cuda.synchronize()
        active_seconds[mode] += time.monotonic() - tick
        counts[mode] += 8
        if time.monotonic() - last_report >= 5:
            observed = cc[rows[:, None], cols[None, :]].cpu().float()
            if not torch.equal(observed, reference):
                error = float((observed-reference).abs().max())
                raise RuntimeError(f"{mode} matrix reference mismatch: max error={error}")
            checks += 1
            progress("matrix_compute", mode=mode, operations=counts.copy(),
                     max_sample_error=0.0, tested_vram_bytes=tested_bytes)
            last_report = time.monotonic()
    for i, pad in enumerate(pads):
        mismatches = int(torch.count_nonzero(pad != (1431655765 ^ i)).item())
        checks += 1
        if mismatches:
            raise RuntimeError(f"VRAM retention mismatch: buffer={i}, count={mismatches}")
    result = dict(status="passed", gpu=prop.name, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic()-start,
                  vram_tested_bytes=tested_bytes, vram_pattern_rounds=rounds,
                  matrix_size=n, matrix_operations=counts, active_seconds=active_seconds,
                  correctness_checks=checks, observed_mismatches=0,
                  peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                  note="Full allocated-buffer pattern checks; sampled matrix checks. Not all physical VRAM or 3D rendering paths tested.")
    save(out / "worker-result.json", result)
    progress("complete")


def supervise(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError("Use a new output directory for each test")
    initial = telemetry()
    if not initial["ac_connected"]:
        raise RuntimeError("Connect AC power before testing")
    if initial["temperature.gpu"] >= 65:
        raise RuntimeError("GPU is already warm; wait until below 65 C")
    if initial["utilization.gpu"] >= 40:
        raise RuntimeError("Another GPU workload appears active; stop it first")
    command = [sys.executable, __file__, "--worker", "--out", str(out),
               "--vram-seconds", str(args.vram_seconds),
               "--compute-seconds", str(args.compute_seconds),
               "--matrix-size", str(args.matrix_size)]
    started = dt.datetime.now().astimezone().isoformat()
    t0 = time.monotonic()
    samples = []
    reason = None
    errors = 0
    child = None
    try:
        with (out / "worker.log").open("w", encoding="utf-8") as log, \
             (out / "telemetry.csv").open("w", newline="", encoding="utf-8") as tf:
            writer = csv.DictWriter(tf, fieldnames=["timestamp", "elapsed_seconds", "ac_connected"] + FIELDS)
            writer.writeheader()
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            while child.poll() is None:
                try:
                    sample = telemetry()
                    sample["elapsed_seconds"] = round(time.monotonic()-t0, 2)
                    samples.append(sample)
                    writer.writerow(sample)
                    tf.flush()
                    errors = 0
                    if sample["temperature.gpu"] >= args.stop_temperature:
                        reason = f"Protective temperature stop: {sample['temperature.gpu']} C"
                    elif not sample["ac_connected"]:
                        reason = "AC disconnected"
                    elif (out / "STOP").exists():
                        reason = "User stop file detected"
                except Exception as exc:
                    errors += 1
                    if errors >= 3:
                        reason = "Telemetry unavailable for three readings: " + str(exc)
                pf = out / "progress.json"
                age = time.time()-pf.stat().st_mtime if pf.exists() else time.monotonic()-t0
                if age > 120:
                    reason = "No worker progress for 120 seconds"
                if time.monotonic()-t0 > args.vram_seconds + args.compute_seconds + 240:
                    reason = "Maximum test duration exceeded"
                if reason:
                    child.terminate()
                    break
                time.sleep(2)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
    except BaseException:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        raise
    finally:
        result_file = out / "worker-result.json"
        worker_result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
        report = dict(started=started, ended=dt.datetime.now().astimezone().isoformat(),
                      duration_seconds=time.monotonic()-t0,
                      status="passed" if child and child.returncode == 0 and worker_result else "stopped_or_failed",
                      stop_reason=reason, child_exit_code=child.returncode if child else None,
                      planned_vram_seconds=args.vram_seconds, planned_compute_seconds=args.compute_seconds,
                      stop_temperature_c=args.stop_temperature, samples=len(samples),
                      peak_temperature_c=max((s["temperature.gpu"] for s in samples), default=None),
                      peak_power_w=max((s["power.draw"] for s in samples), default=None),
                      peak_vram_mib=max((s["memory.used"] for s in samples), default=None),
                      mean_gpu_utilization_percent=sum(s["utilization.gpu"] for s in samples)/max(1,len(samples)),
                      thermal_slowdown_samples=sum(
                          s[FIELDS[9]]=="Active" or s[FIELDS[10]]=="Active" for s in samples),
                      worker=worker_result)
        save(out / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--vram-seconds", type=int, default=120)
    parser.add_argument("--compute-seconds", type=int, default=900)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--stop-temperature", type=float, default=86)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except BaseException:
            traceback.print_exc()
            sys.exit(1)
    else:
        sys.exit(supervise(args))
