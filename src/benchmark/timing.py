import csv
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseTimer:
    def __init__(self, monitor=None):
        self.records = []
        self.monitor = monitor

    @contextmanager
    def phase(self, name, **extra):
        if self.monitor is not None:
            self.monitor.set_phase(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            self.records.append({"phase": name, "seconds": seconds, **extra})
            if self.monitor is not None:
                self.monitor.set_phase("idle")
            print(f"  {name}: {seconds:.3f} s", flush=True)


def summarize(values):
    values = list(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def write_rows(path: Path, rows, append=False):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    exists = path.exists() and append
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
