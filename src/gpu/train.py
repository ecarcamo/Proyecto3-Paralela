import time

import numpy as np

from src.config import BATCH_SIZE, LEARNING_RATE
from src.parallel_cpu.pipeline import make_dataset
from src.sequential.models import build_cnn_base

VARIANTS = {
    "gpu_fp32": {"device": "gpu", "mixed": False, "pipeline": False, "steps_per_execution": 1},
    "gpu_fp32_tfdata": {"device": "gpu", "mixed": False, "pipeline": True, "steps_per_execution": 1},
    "gpu_mixed_tfdata": {"device": "gpu", "mixed": True, "pipeline": True, "steps_per_execution": 1},
    "gpu_mixed_tfdata_spe": {"device": "gpu", "mixed": True, "pipeline": True, "steps_per_execution": 32},
    "cpu_tfdata": {"device": "cpu", "mixed": False, "pipeline": True, "steps_per_execution": 1},
}

INFERENCE_BATCH = 256


def build_model(tf, mixed, seed, steps_per_execution=1):
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if mixed else "float32")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = build_cnn_base()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=steps_per_execution,
    )
    return model


def warm_up(tf, variant, data):
    X_train, y_train = data["train"]
    sample = {"train": (X_train[:1024], y_train[:1024]), "val": (X_train[:256], y_train[:256])}
    start = time.perf_counter()
    model = build_model(tf, variant["mixed"], 0, variant["steps_per_execution"])
    fit(tf, model, sample, 1, variant["pipeline"], 0)
    return time.perf_counter() - start


def fit(tf, model, data, epochs, pipeline, seed):
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    if pipeline:
        train = make_dataset(tf, X_train, y_train, BATCH_SIZE, shuffle=True, seed=seed)
        val = make_dataset(tf, X_val, y_val, INFERENCE_BATCH)
        return model.fit(train, validation_data=val, epochs=epochs, verbose=0)
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     batch_size=BATCH_SIZE, epochs=epochs, verbose=0)


def predict(tf, model, X, pipeline):
    if pipeline:
        dataset = make_dataset(tf, X, np.zeros(len(X), dtype=np.int32), INFERENCE_BATCH)
        return model.predict(dataset.map(lambda x, _: x), verbose=0)
    return model.predict(X, batch_size=INFERENCE_BATCH, verbose=0)


def measure_transfers(tf, array):
    with tf.device("/CPU:0"):
        host = tf.constant(array)
    tf.test.experimental.sync_devices()
    start = time.perf_counter()
    with tf.device("/GPU:0"):
        device = tf.identity(host)
    tf.test.experimental.sync_devices()
    host_to_device = time.perf_counter() - start

    start = time.perf_counter()
    device.numpy()
    device_to_host = time.perf_counter() - start
    del device
    megabytes = array.nbytes / 2**20
    return {
        "megabytes": megabytes,
        "h2d_seconds": host_to_device,
        "d2h_seconds": device_to_host,
        "h2d_gb_per_s": megabytes / 1024 / host_to_device,
        "d2h_gb_per_s": megabytes / 1024 / device_to_host,
    }
