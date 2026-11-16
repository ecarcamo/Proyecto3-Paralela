import argparse
import json
import time

import numpy as np

from src.benchmark.gpu_monitor import GpuMonitor
from src.benchmark.timing import summarize, write_rows
from src.config import DATA_DIR, LOGS_DIR, MODELS_DIR, RESULTS_DIR, ensure_dirs
from src.gpu_runtime import load_tensorflow
from src.parallel_cpu.dataset import load_splits_parallel
from src.sequential.dataset import list_images, split_dataset


def timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "cnn_base_gpu_fp32.keras"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--numpy-repeats", type=int, default=3)
    parser.add_argument("--profile-repeats", type=int, default=20)
    args = parser.parse_args()

    tf = load_tensorflow("gpu")
    from src.gpu.numba_forward import NumbaCnn, export_weights, predict_numpy

    ensure_dirs()
    classes, files, labels = list_images(DATA_DIR / "subset")
    data = load_splits_parallel(split_dataset(files, labels), classes)
    X_test, y_test = data["test"]

    model = tf.keras.models.load_model(args.model)
    weights = export_weights(model)
    reference = model.predict(X_test, batch_size=256, verbose=0)

    networks = {"numba_cuda_naive": NumbaCnn(weights, tiled=False)}
    network, compile_seconds = timed(NumbaCnn, weights)
    networks["numba_cuda_tiled"] = network
    (_, _), first_call = timed(network.predict, X_test)
    networks["numba_cuda_naive"].predict(X_test)

    rows = []
    with GpuMonitor(LOGS_DIR / "gpu_kernels.csv") as monitor:
        for name, current in networks.items():
            monitor.set_phase(name)
            for run in range(args.repeats):
                (probabilities, timings), seconds = timed(current.predict, X_test)
                rows.append({"implementation": name, "run": run, "seconds": seconds,
                             **{key.replace("_ms", "_s"): value / 1000 for key, value in timings.items()}})
        monitor.set_phase("keras_gpu")
        for run in range(args.repeats):
            _, seconds = timed(lambda: model.predict(X_test, batch_size=256, verbose=0))
            rows.append({"implementation": "keras_gpu", "run": run, "seconds": seconds})
        monitor.set_phase("numpy_cpu")
        for run in range(args.numpy_repeats):
            numpy_probabilities, seconds = timed(predict_numpy, weights, X_test)
            rows.append({"implementation": "numpy_cpu", "run": run, "seconds": seconds})

    profile = []
    for name, current in networks.items():
        current.predict(X_test)
        samples = [current.profile(X_test) for _ in range(args.profile_repeats)]
        for index, row in enumerate(samples[0]):
            profile.append({"variant": name, **row,
                            "ms": float(np.mean([sample[index]["ms"] for sample in samples]))})
    validation = {
        "images": int(len(X_test)),
        "numba_compile_and_upload_s": compile_seconds,
        "numba_first_call_with_jit_s": first_call,
        "max_abs_diff_numba_vs_keras": float(np.abs(probabilities - reference).max()),
        "max_abs_diff_numpy_vs_keras": float(np.abs(numpy_probabilities - reference).max()),
        "argmax_agreement_numba_vs_keras": float((probabilities.argmax(1) == reference.argmax(1)).mean()),
        "accuracy_keras": float((reference.argmax(1) == y_test).mean()),
        "accuracy_numba": float((probabilities.argmax(1) == y_test).mean()),
        "accuracy_numpy": float((numpy_probabilities.argmax(1) == y_test).mean()),
        "summary_seconds": {
            name: summarize(row["seconds"] for row in rows if row["implementation"] == name)
            for name in ("numba_cuda_naive", "numba_cuda_tiled", "keras_gpu", "numpy_cpu")
        },
    }
    write_rows(RESULTS_DIR / "kernels_runs.csv", rows)
    write_rows(RESULTS_DIR / "kernels_profile.csv", profile)
    (RESULTS_DIR / "kernels_validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    for row in profile:
        print(row)


if __name__ == "__main__":
    main()
