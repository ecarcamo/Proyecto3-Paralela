import json
import platform
import subprocess

import psutil

from src.config import RESULTS_DIR, ensure_dirs
from src.gpu_runtime import load_tensorflow


def cpu_model() -> str:
    for line in open("/proc/cpuinfo", encoding="utf-8"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def nvidia_smi() -> dict:
    fields = "name,driver_version,memory.total,power.limit,clocks.max.sm"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return dict(zip(fields.split(","), (value.strip() for value in output.split(","))))


def numba_device() -> dict:
    from numba import cuda

    device = cuda.get_current_device()
    free, total = cuda.current_context().get_memory_info()
    return {
        "name": device.name.decode() if isinstance(device.name, bytes) else device.name,
        "compute_capability": ".".join(map(str, device.compute_capability)),
        "multiprocessors": device.MULTIPROCESSOR_COUNT,
        "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
        "warp_size": device.WARP_SIZE,
        "max_shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
        "memory_total_mb": round(total / 2**20),
        "memory_free_mb": round(free / 2**20),
    }


def collect() -> dict:
    tf = load_tensorflow("gpu")
    gpus = tf.config.list_physical_devices("GPU")
    build = tf.sysconfig.get_build_info()
    import numba

    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": {
            "model": cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        },
        "gpu": nvidia_smi(),
        "tensorflow": {
            "version": tf.__version__,
            "cuda_build": build.get("cuda_version"),
            "cudnn_build": build.get("cudnn_version"),
            "gpus": [gpu.name for gpu in gpus],
            "gpu_details": [tf.config.experimental.get_device_details(gpu) for gpu in gpus],
        },
        "numba": {"version": numba.__version__, "device": numba_device()},
    }


def main() -> None:
    ensure_dirs()
    info = collect()
    (RESULTS_DIR / "environment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
