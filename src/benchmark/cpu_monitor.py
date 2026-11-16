import csv
import threading
import time
from pathlib import Path

import psutil


class CpuMonitor:
    def __init__(self, output: Path, interval: float = 0.25):
        self.output = output
        self.interval = interval
        self.phase = "idle"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _run(self) -> None:
        cores = psutil.cpu_count(logical=True)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "phase", "ram_used_gb", *[f"core_{i}" for i in range(cores)]])
            psutil.cpu_percent(percpu=True)
            start = time.perf_counter()
            while not self._stop.wait(self.interval):
                per_core = psutil.cpu_percent(percpu=True)
                ram = psutil.virtual_memory().used / 2**30
                writer.writerow([f"{time.perf_counter() - start:.2f}", self.phase, f"{ram:.2f}", *per_core])

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
