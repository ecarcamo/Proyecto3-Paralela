import math

import numpy as np
from numba import cuda
from numpy.lib.stride_tricks import sliding_window_view

from src.gpu import kernels

MAX_BATCH = 256
CONV_LAYERS = ("conv1", "conv2", "conv3")


def export_weights(model):
    layers = [layer for layer in model.layers if layer.get_weights()]
    names = ["conv1", "conv2", "conv3", "dense1", "dense2"]
    weights = {}
    for name, layer in zip(names, layers):
        kernel, bias = layer.get_weights()
        weights[f"{name}_w"] = kernel.astype(np.float32)
        weights[f"{name}_b"] = bias.astype(np.float32)
    return weights


def _blocks(width, height, planes):
    block = (min(kernels.TILE, width), min(kernels.TILE, height))
    grid = (math.ceil(width / block[0]), math.ceil(height / block[1]), planes)
    return grid, block


class NumbaCnn:
    def __init__(self, weights, image_size=64, tiled=True):
        self.size = image_size
        self.tiled = tiled
        self.weights = {name: cuda.to_device(np.ascontiguousarray(value)) for name, value in weights.items()}
        self.channels = [3] + [weights[f"{name}_w"].shape[3] for name in CONV_LAYERS]
        self.classes = weights["dense2_w"].shape[1]
        self._allocate(MAX_BATCH)

    def _allocate(self, batch):
        size, channels = self.size, self.channels
        self.buffers = {"input": cuda.device_array((batch, size, size, 3), np.float32),
                        "chw": cuda.device_array((batch, 3, size, size), np.float32)}
        for i, name in enumerate(CONV_LAYERS):
            side = size >> i
            self.buffers[f"{name}_out"] = cuda.device_array((batch, channels[i + 1], side, side), np.float32)
            self.buffers[f"{name}_pool"] = cuda.device_array((batch, channels[i + 1], side // 2, side // 2), np.float32)
        features = self.weights["dense1_w"].shape[0]
        hidden = self.weights["dense1_w"].shape[1]
        self.buffers["flat"] = cuda.device_array((batch, features), np.float32)
        self.buffers["hidden"] = cuda.device_array((batch, hidden), np.float32)
        self.buffers["logits"] = cuda.device_array((batch, self.classes), np.float32)
        self.buffers["probabilities"] = cuda.device_array((batch, self.classes), np.float32)

    def _conv_blocks(self, side, batch, channels_out):
        if not self.tiled:
            return _blocks(side, side, batch * channels_out)
        tiles = math.ceil(side / kernels.TILE)
        return (tiles, tiles, batch * channels_out // kernels.CHANNELS_PER_THREAD), (kernels.TILE, kernels.TILE)

    def launch_plan(self, batch):
        size, plan = self.size, []
        plan.append(("to_channels_first", *_blocks(size, size, batch * 3)))
        for i, name in enumerate(CONV_LAYERS):
            side = size >> i
            plan.append((f"{name}_relu", *self._conv_blocks(side, batch, self.channels[i + 1])))
            plan.append((f"{name}_maxpool", *_blocks(side // 2, side // 2, batch * self.channels[i + 1])))
        features = self.weights["dense1_w"].shape[0]
        plan.append(("flatten", (math.ceil(features / 256), batch), (256, 1)))
        hidden = self.weights["dense1_w"].shape[1]
        plan.append(("dense1_relu", (math.ceil(hidden / kernels.DENSE_THREADS), batch), (kernels.DENSE_THREADS,)))
        plan.append(("dense2", (math.ceil(self.classes / kernels.DENSE_THREADS), batch), (kernels.DENSE_THREADS,)))
        plan.append(("softmax", (math.ceil(batch / 128),), (128,)))
        return plan

    def _run_kernels(self, batch, marks=None):
        b, w = self.buffers, self.weights
        plan = self.launch_plan(batch)
        current = b["chw"][:batch]
        conv = kernels.conv3x3_relu_tiled if self.tiled else kernels.conv3x3_relu
        steps = [lambda g, k, o=current: kernels.to_channels_first[g, k](b["input"][:batch], o)]
        for name in CONV_LAYERS:
            source = current
            conv_out = b[f"{name}_out"][:batch]
            pooled = b[f"{name}_pool"][:batch]
            steps.append(lambda g, k, s=source, o=conv_out, n=name:
                         conv[g, k](s, w[f"{n}_w"], w[f"{n}_b"], o))
            steps.append(lambda g, k, s=conv_out, o=pooled: kernels.maxpool2x2[g, k](s, o))
            current = pooled
        steps.append(lambda g, k, s=current: kernels.flatten_channels_last[g, k](s, b["flat"][:batch]))
        steps.append(lambda g, k: kernels.dense[g, k](b["flat"][:batch], w["dense1_w"], w["dense1_b"],
                                                       b["hidden"][:batch], True))
        steps.append(lambda g, k: kernels.dense[g, k](b["hidden"][:batch], w["dense2_w"], w["dense2_b"],
                                                       b["logits"][:batch], False))
        steps.append(lambda g, k: kernels.softmax_rows[g, k](b["logits"][:batch], b["probabilities"][:batch]))
        for (name, grid, block), step in zip(plan, steps):
            step(grid, block)
            if marks is not None:
                event = cuda.event()
                event.record()
                marks.append((name, event))

    def profile(self, images):
        images = np.ascontiguousarray(images[:MAX_BATCH], dtype=np.float32)
        batch = len(images)
        self.buffers["input"][:batch].copy_to_device(images)
        start = cuda.event()
        start.record()
        marks = []
        self._run_kernels(batch, marks)
        marks[-1][1].synchronize()
        profile, previous = [], start
        for (name, grid, block), (_, event) in zip(self.launch_plan(batch), marks):
            threads = math.prod(grid) * math.prod(block)
            profile.append({"kernel": name, "grid": str(grid), "block": str(block), "threads": threads,
                            "ms": cuda.event_elapsed_time(previous, event)})
            previous = event
        return profile

    def predict(self, images):
        images = np.ascontiguousarray(images, dtype=np.float32)
        output = np.empty((len(images), self.classes), dtype=np.float32)
        timings = {"h2d_ms": 0.0, "kernels_ms": 0.0, "d2h_ms": 0.0}
        events = [cuda.event() for _ in range(4)]
        for start in range(0, len(images), MAX_BATCH):
            chunk = images[start:start + MAX_BATCH]
            batch = len(chunk)
            events[0].record()
            self.buffers["input"][:batch].copy_to_device(chunk)
            events[1].record()
            self._run_kernels(batch)
            events[2].record()
            self.buffers["probabilities"][:batch].copy_to_host(output[start:start + batch])
            events[3].record()
            events[3].synchronize()
            timings["h2d_ms"] += cuda.event_elapsed_time(events[0], events[1])
            timings["kernels_ms"] += cuda.event_elapsed_time(events[1], events[2])
            timings["d2h_ms"] += cuda.event_elapsed_time(events[2], events[3])
        return output, timings


def predict_numpy(weights, images, batch=256):
    outputs = []
    for start in range(0, len(images), batch):
        x = images[start:start + batch].astype(np.float32)
        for name in CONV_LAYERS:
            padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
            windows = sliding_window_view(padded, (3, 3), axis=(1, 2))
            x = np.einsum("nhwcij,ijco->nhwo", windows, weights[f"{name}_w"], optimize=True)
            x = np.maximum(x + weights[f"{name}_b"], 0)
            n, h, w, c = x.shape
            x = x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))
        x = x.reshape(len(x), -1)
        x = np.maximum(x @ weights["dense1_w"] + weights["dense1_b"], 0)
        logits = x @ weights["dense2_w"] + weights["dense2_b"]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        outputs.append(exp / exp.sum(axis=1, keepdims=True))
    return np.concatenate(outputs)
