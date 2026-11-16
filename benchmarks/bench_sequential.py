import argparse

import numpy as np

from src.benchmark.cpu_monitor import CpuMonitor
from src.benchmark.timing import PhaseTimer, write_rows
from src.config import BATCH_SIZE, DATA_DIR, LEARNING_RATE, LOGS_DIR, RESULTS_DIR, SEED, ensure_dirs
from src.gpu_runtime import load_tensorflow
from src.sequential.dataset import list_images, load_splits, split_dataset
from src.sequential.models import build_cnn_base, compile_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    tf = load_tensorflow("cpu")
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    ensure_dirs()
    suffix = f"_{args.threads}thread" if args.threads else ""
    base = DATA_DIR / "subset"

    classes, files, labels = list_images(base)
    load_splits(split_dataset(files, labels), classes)

    rows = []
    with CpuMonitor(LOGS_DIR / f"cpu_sequential{suffix}.csv") as monitor:
        timer = PhaseTimer(monitor)
        for run in range(args.repeats):
            print(f"Corrida {run + 1}/{args.repeats}", flush=True)
            timer.records.clear()
            with timer.phase("listado_split"):
                classes, files, labels = list_images(base)
                splits = split_dataset(files, labels)
            with timer.phase("preparacion_datos"):
                data = load_splits(splits, classes)
            X_train, y_train = data["train"]
            X_val, y_val = data["val"]
            X_test, y_test = data["test"]

            tf.keras.backend.clear_session()
            tf.random.set_seed(SEED + run)
            np.random.seed(SEED + run)
            model = compile_model(build_cnn_base(), LEARNING_RATE)
            with timer.phase("entrenamiento"):
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          batch_size=BATCH_SIZE, epochs=args.epochs, verbose=0)
            with timer.phase("inferencia"):
                probabilities = model.predict(X_test, batch_size=256, verbose=0)
            accuracy = float((probabilities.argmax(axis=1) == y_test).mean())
            print(f"  accuracy prueba: {accuracy:.4f}", flush=True)

            for record in timer.records:
                rows.append({
                    "implementation": "secuencial",
                    "device": "cpu",
                    "threads": args.threads or "auto",
                    "images": len(files),
                    "epochs": args.epochs,
                    "run": run,
                    **record,
                    "test_accuracy": accuracy,
                })

    write_rows(RESULTS_DIR / f"sequential_runs{suffix}.csv", rows)


if __name__ == "__main__":
    main()
