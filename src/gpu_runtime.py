import ctypes
import os
import site
from pathlib import Path

PRELOADED_LIBRARIES = ("cusolver/lib/libcusolver.so.11",)


def preload_cuda_libraries() -> None:
    for base in site.getsitepackages():
        for relative in PRELOADED_LIBRARIES:
            path = Path(base) / "nvidia" / relative
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def load_tensorflow(device: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "gpu":
        preload_cuda_libraries()
    else:
        raise ValueError(f"device desconocido: {device}")

    import tensorflow as tf

    if device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("TensorFlow no detecta ninguna GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    return tf
