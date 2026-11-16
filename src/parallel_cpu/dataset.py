from multiprocessing import get_context, shared_memory

import numpy as np
import psutil
from PIL import Image

from src.config import IMAGE_SIZE

_worker_state = {}


def _attach(name, shape):
    if _worker_state.get("name") == name:
        return _worker_state["pixels"]
    previous = _worker_state.pop("buffer", None)
    if previous is not None:
        previous.close()
    buffer = shared_memory.SharedMemory(name=name)
    _worker_state.update(name=name, buffer=buffer, pixels=np.ndarray(shape, dtype=np.uint8, buffer=buffer.buf))
    return _worker_state["pixels"]


def _decode_chunk(task):
    name, shape, start, paths = task
    pixels = _attach(name, shape)
    for offset, path in enumerate(paths):
        with Image.open(path) as img:
            pixels[start + offset] = np.asarray(img.resize(IMAGE_SIZE))
    return len(paths)


def default_workers():
    return psutil.cpu_count(logical=False) or 1


def normalize(pixels):
    return np.divide(pixels, np.float32(255.0), dtype=np.float32)


class ParallelImageLoader:
    def __init__(self, workers=None, chunk_size=64, start_method="fork"):
        self.workers = workers or default_workers()
        self.chunk_size = chunk_size
        self.pool = get_context(start_method).Pool(self.workers)

    def load_pixels(self, paths):
        shape = (len(paths), *IMAGE_SIZE, 3)
        buffer = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        try:
            tasks = [(buffer.name, shape, start, paths[start:start + self.chunk_size])
                     for start in range(0, len(paths), self.chunk_size)]
            loaded = sum(self.pool.imap_unordered(_decode_chunk, tasks))
            if loaded != len(paths):
                raise RuntimeError(f"se cargaron {loaded} de {len(paths)} imágenes")
            return np.ndarray(shape, dtype=np.uint8, buffer=buffer.buf).copy()
        finally:
            buffer.close()
            buffer.unlink()

    def load_splits(self, splits, classes, as_float=True):
        class_to_index = {name: i for i, name in enumerate(classes)}
        order = [name for name in ("train", "val", "test") if name in splits]
        pixels = self.load_pixels([path for name in order for path in splits[name][0]])
        data, start = {}, 0
        for name in order:
            paths, labels = splits[name]
            block = pixels[start:start + len(paths)]
            start += len(paths)
            y = np.fromiter((class_to_index[label] for label in labels), dtype=np.int32, count=len(labels))
            data[name] = (normalize(block) if as_float else block, y)
        return data

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_pixels_parallel(paths, workers=None, chunk_size=64, start_method="fork"):
    with ParallelImageLoader(workers, chunk_size, start_method) as loader:
        return loader.load_pixels(paths)


def load_splits_parallel(splits, classes, workers=None, as_float=True, start_method="fork"):
    with ParallelImageLoader(workers, start_method=start_method) as loader:
        return loader.load_splits(splits, classes, as_float)
