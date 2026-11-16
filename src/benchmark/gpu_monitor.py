import csv
import subprocess
import threading
import time
from pathlib import Path

FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
)


class GpuMonitor:
    def __init__(self, output: Path, interval_ms: int = 250):
        self.output = output
        self.interval_ms = interval_ms
        self.phase = "idle"
        self._process = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _run(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(self.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "phase", *[field.replace(".", "_") for field in FIELDS]])
            for line in self._process.stdout:
                values = [value.strip().split(" ")[0] for value in line.split(",")]
                writer.writerow([f"{time.perf_counter() - start:.2f}", self.phase, *values])

    def __enter__(self):
        self._process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
             f"-lms={self.interval_ms}"],
            stdout=subprocess.PIPE, text=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._process.terminate()
        self._process.wait()
        self._thread.join()
