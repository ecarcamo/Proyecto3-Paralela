import os
from multiprocessing import get_context, shared_memory

import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE

_worker_state = {}


def _attach_shared_buffer(name, shape):
    buffer = shared_memory.SharedMemory(name=name)
    _worker_state["buffer"] = buffer
    _worker_state["pixels"] = np.ndarray(shape, dtype=np.uint8, buffer=buffer.buf)


def _decode_chunk(task):
    start, paths = task
    pixels = _worker_state["pixels"]
    for offset, path in enumerate(paths):
        with Image.open(path) as img:
            pixels[start + offset] = np.asarray(img.resize(IMAGE_SIZE))
    return len(paths)


def default_workers():
    return os.cpu_count() or 1


def load_pixels_parallel(paths, workers=None, chunk_size=64, start_method="fork"):
    workers = workers or default_workers()
    shape = (len(paths), *IMAGE_SIZE, 3)
    buffer = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
    try:
        tasks = [(start, paths[start:start + chunk_size]) for start in range(0, len(paths), chunk_size)]
        context = get_context(start_method)
        with context.Pool(workers, initializer=_attach_shared_buffer, initargs=(buffer.name, shape)) as pool:
            loaded = sum(pool.imap_unordered(_decode_chunk, tasks))
        if loaded != len(paths):
            raise RuntimeError(f"se cargaron {loaded} de {len(paths)} imágenes")
        return np.ndarray(shape, dtype=np.uint8, buffer=buffer.buf).copy()
    finally:
        buffer.close()
        buffer.unlink()


def normalize(pixels):
    return np.divide(pixels, np.float32(255.0), dtype=np.float32)


def load_splits_parallel(splits, classes, workers=None, as_float=True, start_method="fork"):
    class_to_index = {name: i for i, name in enumerate(classes)}
    order = [name for name in ("train", "val", "test") if name in splits]
    all_paths = [path for name in order for path in splits[name][0]]
    pixels = load_pixels_parallel(all_paths, workers, start_method=start_method)

    data, start = {}, 0
    for name in order:
        paths, labels = splits[name]
        block = pixels[start:start + len(paths)]
        start += len(paths)
        X = normalize(block) if as_float else block
        y = np.fromiter((class_to_index[label] for label in labels), dtype=np.int32, count=len(labels))
        data[name] = (X, y)
    return data
