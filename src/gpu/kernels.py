import math

from numba import cuda, float32

TILE = 16
HALO = TILE + 2
CHANNELS_PER_THREAD = 8
DENSE_THREADS = 128


@cuda.jit(fastmath=True)
def conv3x3_relu(inputs, weights, bias, outputs):
    x, y = cuda.grid(2)
    plane = cuda.blockIdx.z
    channels_out = outputs.shape[1]
    n = plane // channels_out
    co = plane % channels_out
    height = inputs.shape[2]
    width = inputs.shape[3]
    if x >= width or y >= height:
        return
    acc = float32(bias[co])
    for ci in range(inputs.shape[1]):
        for ky in range(3):
            iy = y + ky - 1
            if 0 <= iy < height:
                for kx in range(3):
                    ix = x + kx - 1
                    if 0 <= ix < width:
                        acc += inputs[n, ci, iy, ix] * weights[ky, kx, ci, co]
    outputs[n, co, y, x] = acc if acc > 0 else float32(0.0)


@cuda.jit(fastmath=True)
def conv3x3_relu_tiled(inputs, weights, bias, outputs):
    tile = cuda.shared.array((HALO, HALO), float32)
    acc = cuda.local.array(CHANNELS_PER_THREAD, float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x = cuda.blockIdx.x * TILE + tx
    y = cuda.blockIdx.y * TILE + ty
    groups = outputs.shape[1] // CHANNELS_PER_THREAD
    n = cuda.blockIdx.z // groups
    co0 = (cuda.blockIdx.z % groups) * CHANNELS_PER_THREAD
    height = inputs.shape[2]
    width = inputs.shape[3]
    inside = x < width and y < height
    for k in range(CHANNELS_PER_THREAD):
        acc[k] = bias[co0 + k]
    for ci in range(inputs.shape[1]):
        for index in range(ty * TILE + tx, HALO * HALO, TILE * TILE):
            ly = index // HALO
            lx = index % HALO
            gy = cuda.blockIdx.y * TILE + ly - 1
            gx = cuda.blockIdx.x * TILE + lx - 1
            if 0 <= gy < height and 0 <= gx < width:
                tile[ly, lx] = inputs[n, ci, gy, gx]
            else:
                tile[ly, lx] = float32(0.0)
        cuda.syncthreads()
        if inside:
            for ky in range(3):
                for kx in range(3):
                    value = tile[ty + ky, tx + kx]
                    for k in range(CHANNELS_PER_THREAD):
                        acc[k] += value * weights[ky, kx, ci, co0 + k]
        cuda.syncthreads()
    if inside:
        for k in range(CHANNELS_PER_THREAD):
            outputs[n, co0 + k, y, x] = acc[k] if acc[k] > 0 else float32(0.0)


@cuda.jit(fastmath=True)
def maxpool2x2(inputs, outputs):
    x, y = cuda.grid(2)
    plane = cuda.blockIdx.z
    channels = outputs.shape[1]
    n = plane // channels
    c = plane % channels
    if x >= outputs.shape[3] or y >= outputs.shape[2]:
        return
    iy = 2 * y
    ix = 2 * x
    best = inputs[n, c, iy, ix]
    best = max(best, inputs[n, c, iy, ix + 1])
    best = max(best, inputs[n, c, iy + 1, ix])
    best = max(best, inputs[n, c, iy + 1, ix + 1])
    outputs[n, c, y, x] = best


@cuda.jit
def flatten_channels_last(inputs, outputs):
    i, n = cuda.grid(2)
    channels = inputs.shape[1]
    height = inputs.shape[2]
    width = inputs.shape[3]
    if n >= inputs.shape[0] or i >= channels * height * width:
        return
    c = i % channels
    rest = i // channels
    x = rest % width
    y = rest // width
    outputs[n, i] = inputs[n, c, y, x]


@cuda.jit(fastmath=True)
def dense(inputs, weights, bias, outputs, apply_relu):
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    n = cuda.blockIdx.y
    tile = cuda.shared.array(DENSE_THREADS, float32)
    features = inputs.shape[1]
    acc = float32(0.0)
    for start in range(0, features, DENSE_THREADS):
        k = start + cuda.threadIdx.x
        tile[cuda.threadIdx.x] = inputs[n, k] if k < features else float32(0.0)
        cuda.syncthreads()
        if j < outputs.shape[1]:
            limit = min(DENSE_THREADS, features - start)
            for t in range(limit):
                acc += tile[t] * weights[start + t, j]
        cuda.syncthreads()
    if j < outputs.shape[1]:
        acc += bias[j]
        if apply_relu and acc < 0:
            acc = float32(0.0)
        outputs[n, j] = acc


@cuda.jit(fastmath=True)
def softmax_rows(logits, probabilities):
    n = cuda.grid(1)
    if n >= logits.shape[0]:
        return
    classes = logits.shape[1]
    peak = logits[n, 0]
    for j in range(1, classes):
        peak = max(peak, logits[n, j])
    total = float32(0.0)
    for j in range(classes):
        value = math.exp(logits[n, j] - peak)
        probabilities[n, j] = value
        total += value
    for j in range(classes):
        probabilities[n, j] /= total


@cuda.jit
def to_channels_first(images, outputs):
    x, y = cuda.grid(2)
    plane = cuda.blockIdx.z
    channels = outputs.shape[1]
    n = plane // channels
    c = plane % channels
    if x < outputs.shape[3] and y < outputs.shape[2]:
        outputs[n, c, y, x] = images[n, y, x, c]
