"""Final fit on all 273 public training days. No held-out score is claimed."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_stress_test import FIELDS, save, telemetry


def worker(args):
    import numpy as np
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.manual_seed(20261020)
    np.random.seed(20261020)
    rng = np.random.default_rng(20261020)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    # Bound the allocator while leaving VRAM for Windows and monitoring.
    torch.cuda.set_per_process_memory_fraction(.89)
    out = args.out
    start = time.monotonic()

    def progress(phase, **details):
        value = dict(phase=phase, elapsed_seconds=time.monotonic()-start)
        value.update(details)
        save(out / "progress.json", value)
        print(json.dumps(value), flush=True)

    metadata = json.loads((args.cache / "metadata.json").read_text(encoding="utf-8"))
    panels = []
    for p in metadata["panels"]:
        root = args.cache / p["panel"]
        panels.append(dict(meta=p, x=np.load(root / "x.npy", mmap_mode="r"),
                           y=np.load(root / "y.npy", mmap_mode="r"),
                           mask=np.load(root / "target.npy", mmap_mode="r")))

    from trafficflow_model import Model

    model = Model().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(args.init_checkpoint, weights_only=True, map_location="cuda")["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * args.steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    weights = torch.tensor([.54, .46], device="cuda").reshape(1, 2, 1, 1)
    parameters = sum(p.numel() for p in model.parameters())
    progress("ready", parameters=parameters, gpu=torch.cuda.get_device_name())

    def to_device(arrays):
        ax, ay, am = arrays
        x = torch.from_numpy(ax).cuda().to(memory_format=torch.channels_last)
        y = torch.from_numpy(ay).cuda()
        mask = torch.from_numpy(am).cuda()[:, None]
        assert mask.sum().item() > 0
        return x, y, mask

    def batch(count=None, cpu_only=False):
        xs, ys, masks = [], [], []
        for _ in range(args.batch if count is None else count):
            p = panels[int(rng.integers(len(panels)))]
            d = int(rng.integers(273))
            t = int(rng.integers(288 - 96 + 1))
            n = p["x"].shape[-1]
            s = int(rng.integers(max(1, n - 48 + 1)))
            x = np.array(p["x"][d, :, t:t+96, s:s+48], dtype=np.float32)
            y = np.array(p["y"][d, :, t:t+96, s:s+48], dtype=np.float32)
            m = np.array(p["mask"][d, t:t+96, s:s+48], dtype=np.float32)
            if x.shape[-1] < 48:
                pad = 48 - x.shape[-1]
                x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="edge")
                y = np.pad(y, ((0, 0), (0, 0), (0, pad)), mode="edge")
                m = np.pad(m, ((0, 0), (0, pad)))
            xs.append(x); ys.append(y); masks.append(m)
        arrays = (np.stack(xs), np.stack(ys), np.stack(masks))
        return arrays if cpu_only else to_device(arrays)

    if args.auto_batch:
        trials = []
        selected = 32
        for candidate in [128, 256, 512, 768, 1024, 1152, 1216]:
            x = y = mask = pred = loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                x, y, mask = batch(candidate)
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x)
                    loss = (((pred-y)**2) * weights * mask).sum() / mask.sum()
                loss.backward()
                torch.cuda.synchronize()
                allocated = torch.cuda.max_memory_allocated()/1024**3
                reserved = torch.cuda.max_memory_reserved()/1024**3
                trials.append(dict(batch=candidate, peak_allocated_gib=allocated,
                                   peak_reserved_gib=reserved, status="fits"))
                selected = candidate
                progress("batch_calibration", **trials[-1])
                if reserved >= 13.2:
                    break
            except torch.OutOfMemoryError:
                trials.append(dict(batch=candidate, status="allocator_limit"))
                progress("batch_calibration", **trials[-1])
                break
            finally:
                x = y = mask = pred = loss = None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        args.batch = selected
        save(out / "batch-calibration.json", dict(selected_batch=selected, trials=trials,
             purpose="Real training activation memory; no dummy VRAM padding",
             allocator_limit_fraction=.89))
        progress("batch_selected", batch=selected)

    history = []
    training_targets_seen = 0
    from concurrent.futures import ThreadPoolExecutor
    prefetcher = ThreadPoolExecutor(max_workers=1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.
        pending = prefetcher.submit(batch, args.batch, True)
        for step in range(1, args.steps + 1):
            arrays = pending.result()
            if step < args.steps:
                pending = prefetcher.submit(batch, args.batch, True)
            x, y, mask = to_device(arrays)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
                loss = (((pred-y) ** 2) * weights * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()
            training_targets_seen += int(mask.sum().item())
            if step % 25 == 0 or step == 1:
                progress("training", epoch=epoch, epochs=args.epochs, step=step, steps=args.steps,
                         mean_loss=loss_sum/step, gradient_norm=float(grad_norm),
                         learning_rate=scheduler.get_last_lr()[0])
            del x, y, mask, pred, loss, arrays
        history.append(dict(epoch=epoch, train_loss=loss_sum/args.steps,
                            elapsed_seconds=time.monotonic()-start))
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch,
                        features=metadata["features"], train_dates=["2030-06-01", "2031-02-28"],
                        model="64-channel 5-block residual CNN",
                        note="Full-data fit; no independent performance estimate for these weights"),
                   out / "full-data-model.pt")
        save(out / "training-history.json", history)
        progress("epoch_completed", **history[-1])
    prefetcher.shutdown(wait=True)
    result = dict(status="completed", task="Final Task 1 fit",
                  elapsed_seconds=time.monotonic()-start, gpu=torch.cuda.get_device_name(),
                  epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.batch,
                  seed=20261020, training_targets_seen=training_targets_seen,
                  train_days=273, panels=len(panels), history=history,
                  warm_start=str(args.init_checkpoint.resolve()),
                  scope="All public training months. No new holdout score; original held-out results describe the earlier checkpoint.")
    save(out / "training-result.json", result)
    progress("complete", epochs=args.epochs)


def supervise(args):
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    started = dt.datetime.now().astimezone().isoformat()
    initial = telemetry()
    if not initial["ac_connected"] or initial["temperature.gpu"] >= 86:
        raise RuntimeError("Training requires AC power and GPU temperature below 86 C")
    samples, failures, reason = [], 0, None
    with (args.out / "training.log").open("w", encoding="utf-8") as log, \
            (args.out / "telemetry.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "elapsed_seconds", "ac_connected"]+FIELDS)
        writer.writeheader()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--cache", str(args.cache.resolve()), "--out", str(args.out.resolve()),
               "--epochs", str(args.epochs), "--steps", str(args.steps), "--batch", str(args.batch),
               "--init-checkpoint", str(args.init_checkpoint.resolve())]
        if args.auto_batch:
            cmd.append("--auto-batch")
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while child.poll() is None:
                try:
                    row = telemetry()
                    row["elapsed_seconds"] = round(time.monotonic()-start, 2)
                    writer.writerow(row); csvfile.flush(); samples.append(row); failures = 0
                    if row["temperature.gpu"] >= 86: reason = "GPU temperature reached 86 C"
                    if not row["ac_connected"]: reason = "AC disconnected"
                except Exception as exc:
                    failures += 1
                    if failures >= 3: reason = f"Monitoring failed: {exc}"
                progress_path = args.out / "progress.json"
                if progress_path.exists() and time.time()-progress_path.stat().st_mtime > 180:
                    reason = "No worker progress for 180 seconds"
                if time.monotonic()-start > 3600: reason = "One-hour maximum duration"
                if (args.out / "STOP").exists(): reason = "Stop requested"
                if reason:
                    child.terminate(); child.wait(timeout=20); break
                time.sleep(2)
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=20)
    result = dict(status="completed" if child.returncode == 0 and not reason else "stopped",
                  started=started, ended=dt.datetime.now().astimezone().isoformat(),
                  duration_seconds=time.monotonic()-start, child_exit_code=child.returncode,
                  stop_reason=reason, samples=len(samples),
                  peak_temperature_c=max(r["temperature.gpu"] for r in samples),
                  peak_power_w=max(r["power.draw"] for r in samples),
                  peak_vram_mib=max(r["memory.used"] for r in samples),
                  mean_gpu_utilization=sum(r["utilization.gpu"] for r in samples)/len(samples),
                  thermal_slowdown_samples=sum(any(r[k] == "Active" for k in FIELDS[-2:]) for r in samples))
    save(args.out / "monitor-summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--auto-batch", action="store_true")
    args = p.parse_args()
    try:
        worker(args) if args.worker else supervise(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
