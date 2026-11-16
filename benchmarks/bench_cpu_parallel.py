import argparse
import json

import numpy as np

from src.benchmark.cpu_monitor import CpuMonitor
from src.benchmark.timing import PhaseTimer, write_rows
from src.config import DATA_DIR, LOGS_DIR, RESULTS_DIR, ensure_dirs, full_dataset_dir
from src.parallel_cpu.dataset import load_pixels_parallel, load_splits_parallel
from src.sequential.dataset import list_images, load_splits, split_dataset


def check_equivalence(splits, classes, workers):
    sequential = load_splits(splits, classes)
    parallel = load_splits_parallel(splits, classes, workers)
    report = {}
    for name in sequential:
        X_seq, y_seq = sequential[name]
        X_par, y_par = parallel[name]
        report[name] = {
            "shape": list(X_par.shape),
            "dtype": str(X_par.dtype),
            "max_abs_diff": float(np.abs(X_seq - X_par).max()),
            "identical_pixels": bool(np.array_equal(X_seq, X_par)),
            "identical_labels": bool(np.array_equal(y_seq, y_par)),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="1,2,4,8,12,16,24,32")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--start-method", choices=("fork", "spawn"), default="fork")
    args = parser.parse_args()
    ensure_dirs()
    worker_counts = [int(value) for value in args.workers.split(",")]

    if args.full:
        classes, files, labels = list_images(full_dataset_dir())
        tag = "full"
    else:
        classes, files, labels = list_images(DATA_DIR / "subset")
        splits = split_dataset(files, labels)
        tag = "subset"
    if args.start_method != "fork":
        tag = f"{tag}_{args.start_method}"
    if not args.full and args.start_method == "fork":
        equivalence = check_equivalence(splits, classes, max(worker_counts))
        (RESULTS_DIR / "cpu_parallel_equivalence.json").write_text(json.dumps(equivalence, indent=2))
        print(json.dumps(equivalence, indent=2))

    load_pixels_parallel(files, max(worker_counts), start_method=args.start_method)

    rows = []
    with CpuMonitor(LOGS_DIR / f"cpu_parallel_{tag}.csv") as monitor:
        timer = PhaseTimer(monitor)
        for workers in worker_counts:
            for run in range(args.repeats):
                print(f"{workers} procesos, corrida {run + 1}/{args.repeats}", flush=True)
                timer.records.clear()
                if args.full:
                    with timer.phase(f"preparacion_datos_{workers}p"):
                        load_pixels_parallel(files, workers, start_method=args.start_method)
                else:
                    with timer.phase(f"preparacion_datos_{workers}p"):
                        load_splits_parallel(splits, classes, workers, start_method=args.start_method)
                for record in timer.records:
                    rows.append({
                        "implementation": "paralela_cpu",
                        "device": "cpu",
                        "workers": workers,
                        "start_method": args.start_method,
                        "images": len(files),
                        "run": run,
                        "phase": "preparacion_datos",
                        "seconds": record["seconds"],
                    })

    write_rows(RESULTS_DIR / f"cpu_parallel_{tag}.csv", rows)


if __name__ == "__main__":
    main()
