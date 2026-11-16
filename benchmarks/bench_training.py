import argparse
import json

from src.benchmark.cpu_monitor import CpuMonitor
from src.benchmark.gpu_monitor import GpuMonitor
from src.benchmark.timing import PhaseTimer, write_rows
from src.config import DATA_DIR, LOGS_DIR, MODELS_DIR, RESULTS_DIR, SEED, ensure_dirs, full_dataset_dir
from src.gpu.train import VARIANTS, build_model, fit, measure_transfers, predict, warm_up
from src.gpu_runtime import load_tensorflow
from src.parallel_cpu.dataset import ParallelImageLoader
from src.sequential.dataset import list_images, split_dataset


class Monitors:
    def __init__(self, *monitors):
        self.monitors = monitors

    def set_phase(self, phase):
        for monitor in self.monitors:
            monitor.set_phase(phase)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset", choices=("subset", "full"), default="subset")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    loader = ParallelImageLoader()
    tf = load_tensorflow(variant["device"])
    ensure_dirs()
    tag = f"{args.variant}_{args.dataset}"
    base = DATA_DIR / "subset" if args.dataset == "subset" else full_dataset_dir()
    as_float = not variant["pipeline"]

    rows, transfers = [], []
    cpu_log = LOGS_DIR / f"cpu_training_{tag}.csv"
    gpu_log = LOGS_DIR / f"gpu_training_{tag}.csv"
    with CpuMonitor(cpu_log) as cpu_monitor:
        monitors = [cpu_monitor]
        gpu_monitor = GpuMonitor(gpu_log) if variant["device"] == "gpu" else None
        if gpu_monitor:
            gpu_monitor.__enter__()
            monitors.append(gpu_monitor)
        try:
            timer = PhaseTimer(Monitors(*monitors))
            for run in range(args.repeats):
                print(f"{args.variant} ({args.dataset}), corrida {run + 1}/{args.repeats}", flush=True)
                timer.records.clear()
                with timer.phase("listado_split"):
                    classes, files, labels = list_images(base)
                    splits = split_dataset(files, labels)
                with timer.phase("preparacion_datos"):
                    data = loader.load_splits(splits, classes, as_float=as_float)
                if variant["device"] == "gpu":
                    transfer = measure_transfers(tf, data["train"][0])
                    timer.records.append({"phase": "transferencia_h2d", "seconds": transfer["h2d_seconds"]})
                    print(f"  transferencia_h2d: {transfer['h2d_seconds']:.3f} s", flush=True)
                    transfers.append({"variant": args.variant, "dataset": args.dataset, "run": run,
                                      "dtype": str(data["train"][0].dtype), **transfer})

                if run == 0:
                    warmup_seconds = warm_up(tf, variant, data)
                    print(f"  calentamiento (inicialización CUDA/cuDNN, no se mide): {warmup_seconds:.3f} s", flush=True)
                model = build_model(tf, variant["mixed"], SEED + run, variant["steps_per_execution"])
                with timer.phase("entrenamiento"):
                    fit(tf, model, data, args.epochs, variant["pipeline"], SEED + run)
                X_test, y_test = data["test"]
                with timer.phase("inferencia"):
                    probabilities = predict(tf, model, X_test, variant["pipeline"])
                accuracy = float((probabilities.argmax(axis=1) == y_test).mean())
                print(f"  accuracy prueba: {accuracy:.4f}", flush=True)

                for record in timer.records:
                    rows.append({
                        "implementation": args.variant,
                        "device": variant["device"],
                        "mixed_precision": variant["mixed"],
                        "tf_data": variant["pipeline"],
                        "steps_per_execution": variant["steps_per_execution"],
                        "warmup_seconds": warmup_seconds,
                        "dataset": args.dataset,
                        "images": len(files),
                        "epochs": args.epochs,
                        "run": run,
                        **record,
                        "test_accuracy": accuracy,
                    })
                if args.save_model and run == 0:
                    model.save(MODELS_DIR / f"cnn_base_{args.variant}.keras")
                del data, model, probabilities
        finally:
            if gpu_monitor:
                gpu_monitor.__exit__(None, None, None)
            loader.close()

    write_rows(RESULTS_DIR / f"training_{tag}.csv", rows)
    if transfers:
        write_rows(RESULTS_DIR / f"transfers_{tag}.csv", transfers)
        print(json.dumps(transfers[-1], indent=2))


if __name__ == "__main__":
    main()
