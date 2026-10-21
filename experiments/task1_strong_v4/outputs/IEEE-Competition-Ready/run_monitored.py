"""Run the submission builder with a separate GPU/AC watchdog."""
import csv
import datetime as dt
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_stress_test import FIELDS, save, telemetry


def main():
    command = [sys.executable, "-X", "utf8", str(Path(__file__).with_name("build_submission.py"))]+sys.argv[1:]
    out = Path(sys.argv[sys.argv.index("--out")+1])
    out.mkdir(parents=True, exist_ok=True)
    initial = telemetry()
    if not initial["ac_connected"] or initial["temperature.gpu"] >= 86:
        raise RuntimeError("AC power and GPU temperature below 86 C required")
    start = time.monotonic()
    started = dt.datetime.now().astimezone().isoformat()
    samples, failures, reason = [], 0, None
    with (out/"pipeline.log").open("w", encoding="utf-8") as log, \
            (out/"pipeline-telemetry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "elapsed_seconds", "ac_connected"]+FIELDS)
        writer.writeheader()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while child.poll() is None:
                try:
                    row = telemetry()
                    row["elapsed_seconds"] = time.monotonic()-start
                    writer.writerow(row); f.flush(); samples.append(row)
                    failures = 0
                    if row["temperature.gpu"] >= 86: reason = "Temperature reached 86 C"
                    if not row["ac_connected"]: reason = "AC disconnected"
                except Exception as exc:
                    failures += 1
                    if failures >= 3: reason = str(exc)
                if time.monotonic()-start > 3600: reason = "One-hour time limit"
                if (out/"STOP").exists(): reason = "User stop"
                if reason:
                    child.terminate()
                    child.wait(timeout=20)
                    break
                time.sleep(2)
        finally:
            if child.poll() is None:
                child.terminate(); child.wait(timeout=20)
    save(out/"pipeline-monitor.json", dict(status="completed" if child.returncode == 0 and not reason else "failed",
         child_exit_code=child.returncode, stop_reason=reason, started=started,
         ended=dt.datetime.now().astimezone().isoformat(), seconds=time.monotonic()-start,
         samples=len(samples), peak_temperature_c=max(r["temperature.gpu"] for r in samples),
         peak_vram_mib=max(r["memory.used"] for r in samples),
         peak_power_w=max(r["power.draw"] for r in samples)))
    sys.exit(child.returncode)


if __name__ == "__main__":
    main()
