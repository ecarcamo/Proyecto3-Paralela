# 4. Diseño del algoritmo paralelo con CUDA y 5. Implementación con CUDA Python

**Responsable:** Esteban Cárcamo (`ecarcamo`)

## 4.1 Qué se le delega a la GPU

Según `docs/03_paralelizacion_cpu.md`, a la GPU solo le corresponde la **matemática de la red neuronal**: las
convoluciones, el max-pooling, las capas densas, el softmax, la retropropagación y la actualización de Adam. Los
archivos se quedan en el CPU. Se hizo en dos niveles:

1. **Entrenamiento con TensorFlow sobre CUDA** (`src/gpu/train.py`). TensorFlow ejecuta cada capa con kernels de
   NVIDIA: cuDNN para convoluciones y pooling, y cuBLAS para las densas. Es la forma realista de entrenar en GPU.
2. **Kernels propios con Numba CUDA** (`src/gpu/kernels.py`, `src/gpu/numba_forward.py`). La misma CNN entrenada se
   ejecuta hacia adelante con kernels escritos por nosotros en CUDA Python. Aquí sí decidimos cómo se reparte el
   trabajo entre hilos, bloques y grids.

Librerías que permiten trabajar con la GPU:

| Librería | Para qué se usó |
|---|---|
| TensorFlow 2.21 (`tensorflow[and-cuda]`) | Entrenamiento e inferencia en `/GPU:0`; trae CUDA 12.5, cuDNN 9 y cuBLAS como paquetes de pip |
| cuDNN / cuBLAS | Kernels optimizados de convolución, pooling y producto de matrices que TensorFlow llama |
| Keras `mixed_precision` | Política `mixed_float16`: cálculos en FP16 en los *tensor cores* y pesos en FP32 |
| `tf.data` | Pipeline en CPU que prepara el siguiente batch mientras la GPU entrena (`prefetch`) |
| Numba 0.68 + `numba-cuda` 0.30 | `@cuda.jit`, `cuda.grid`, `cuda.shared.array`, `cuda.syncthreads`, `cuda.to_device`, `cuda.event` |
| `nvidia-smi` | Monitoreo de uso de núcleos, memoria, potencia y reloj de la GPU cada 250 ms (`src/benchmark/gpu_monitor.py`) |

## 4.2 Hardware objetivo

| Recurso | RTX 4050 Laptop |
|---|---|
| Multiprocesadores (SM) | 20 (compute capability 8.9, arquitectura Ada) |
| Hilos por warp | 32 |
| Máximo de hilos por bloque | 1 024 |
| Memoria compartida por bloque | 48 KB |
| Memoria global | 6 GB GDDR6 |
| Tensor cores | Sí (FP16/BF16), aprovechados con precisión mixta |

## 4.3 Distribución del trabajo en los kernels propios

Convención: los mapas de activación se guardan en la GPU en formato **NCHW** (imagen, canal, fila, columna). Así los
hilos vecinos en `x` leen direcciones contiguas de memoria y los accesos se agrupan (*coalescing*). Se procesan
batches de **B = 256 imágenes**. El límite de 65 535 bloques en el eje `z` del grid es lo que fija ese máximo.

| Kernel | Qué calcula un hilo | Bloque | Grid (B = 256) | Hilos lanzados |
|---|---|---|---|---|
| `to_channels_first` | 1 valor: reordena NHWC → NCHW | 16×16 | (4, 4, B·3) | 3.1 M |
| `conv3x3_relu_tiled` capa 1 | 1 píxel × **8 canales de salida** | 16×16 | (4, 4, B·32/8) | 4.2 M |
| `maxpool2x2` capa 1 | 1 píxel de salida (máximo de 2×2) | 16×16 | (2, 2, B·32) | 8.4 M |
| `conv3x3_relu_tiled` capa 2 | 1 píxel × 8 canales | 16×16 | (2, 2, B·64/8) | 2.1 M |
| `maxpool2x2` capa 2 | 1 píxel | 16×16 | (1, 1, B·64) | 4.2 M |
| `conv3x3_relu_tiled` capa 3 | 1 píxel × 8 canales | 16×16 | (1, 1, B·128/8) | 1.0 M |
| `maxpool2x2` capa 3 | 1 píxel | 8×8 | (1, 1, B·128) | 2.1 M |
| `flatten_channels_last` | 1 elemento: NCHW → orden de Keras (H, W, C) | 256 | (32, B) | 2.1 M |
| `dense` capa 1 (+ReLU) | 1 neurona de 128 para 1 imagen | 128 | (1, B) | 32 768 |
| `dense` capa 2 | 1 logit de 29 para 1 imagen | 128 | (1, B) | 32 768 |
| `softmax_rows` | 1 imagen completa (29 clases) | 128 | (2,) | 256 |

Detalles de diseño:

* **Grid de 3 dimensiones en las capas 2D.** `blockIdx.x` y `blockIdx.y` recorren los mosaicos de 16×16 píxeles de
  la imagen. `blockIdx.z` codifica a la vez la imagen y el grupo de canales (`z = n · grupos + grupo`). Cada bloque
  trabaja un solo canal o grupo de canales de una sola imagen, así que todos sus hilos leen los **mismos pesos**:
  la caché los difunde (*broadcast*) en vez de pedirlos 256 veces.
* **Convolución con memoria compartida (`conv3x3_relu_tiled`).** Por cada canal de entrada, los 256 hilos del bloque
  copian entre todos un mosaico de 18×18 (16×16 más un borde de 1 píxel para el filtro 3×3) a `cuda.shared.array`,
  se sincronizan con `cuda.syncthreads()` y luego cada hilo lee sus 9 vecinos desde la memoria compartida, no desde
  la global. Además cada hilo acumula **8 canales de salida en registros** (`cuda.local.array`): cada valor que lee
  lo reutiliza 8 veces. Así se cargan 8 veces menos datos de entrada por operación.
* **Capas densas con mosaicos en memoria compartida.** Los 128 hilos de un bloque calculan las 128 neuronas de una
  imagen. Cargan la entrada en trozos de 128 valores a memoria compartida, y cada hilo lee la fila de pesos que le toca.
  Los pesos `W[k, j]` para `j` consecutivos están contiguos, así que el acceso es coalescente.
* **Guardas de borde** (`if x >= width ...`) en todos los kernels: los mosaicos de la última fila o columna pueden
  quedar parcialmente fuera de la imagen.
* **Medición con `cuda.event`.** Se registran eventos antes y después de la copia CPU→GPU, de los kernels y de la
  copia GPU→CPU, para separar el tiempo de transferencia del de cómputo, como pide el enunciado.

Se implementó primero una versión **ingenua** (`conv3x3_relu`: 1 hilo = 1 píxel de 1 canal, todo desde memoria
global) y después la de memoria compartida, para medir el efecto del diseño.

## 4.4 Distribución del trabajo en el entrenamiento con TensorFlow

En TensorFlow no se eligen bloques a mano: cuDNN escoge el algoritmo de convolución y su configuración de lanzamiento
(*autotuning*) la primera vez que ve cada forma de tensor. Esa es parte del costo de «calentamiento» medido aparte.
Lo que sí se diseñó es el flujo entre CPU y GPU:

```
CPU (24 procesos)            CPU (hilos tf.data)               GPU (CUDA)
──────────────────           ─────────────────────             ──────────────────────────────────
JPG → 64×64 uint8   ──►      barajar, batch de 32,     ──►     conv/pool/dense FP16 (tensor cores)
(una vez)                    uint8 → float32 ÷255,             retropropagación y Adam
                             prefetch (cola de batches)        32 pasos por cada llamada de Python
```

Optimizaciones aplicadas, una por variante medida:

| Variante | Qué cambia | Por qué |
|---|---|---|
| `gpu_fp32` | El código original, solo que en GPU | Línea base de GPU: mismo `model.fit` con arreglos NumPy |
| `gpu_fp32_tfdata` | Pipeline `tf.data` con `prefetch` | El CPU prepara el siguiente batch mientras la GPU calcula |
| `gpu_mixed_tfdata` | Precisión mixta `mixed_float16` | Activa los tensor cores y reduce a la mitad la memoria por activación |
| `gpu_mixed_tfdata_spe` | `steps_per_execution=32` | Python despacha 32 pasos de entrenamiento por llamada en vez de 1 |

Ninguna cambia el algoritmo: siguen siendo batches de 32, Adam con lr = 1e-3, 5 épocas y validación por época. Por
eso la accuracy queda en el mismo rango que la versión secuencial.

## 5. Implementación con CUDA Python: resultados

Mediciones de la corrida final (`scripts/run_all_benchmarks.sh`), con los mismos datos, el mismo split, las mismas
5 épocas y el mismo batch que la versión secuencial. Cada configuración se corrió 5 veces en su propio proceso. Antes
de la primera corrida se hace un calentamiento no medido, para separar el costo único de inicializar CUDA/cuDNN.

### 5.1 Entrenamiento

| Configuración | Entrenamiento (s) | s/época | Speedup vs. CPU original | Accuracy prueba |
|---|---|---|---|---|
| CPU, 32 hilos (Lab 3 original) | 41.312 ± 1.697 | 8.26 | 1.00 | 0.933 ± 0.006 |
| GPU fp32, arreglos NumPy | 16.286 ± 1.913 | 3.26 | 2.54 | 0.924 ± 0.010 |
| GPU fp32 + `tf.data` | 14.941 ± 1.763 | 2.99 | 2.76 | 0.936 ± 0.006 |
| GPU precisión mixta + `tf.data` | 12.691 ± 1.883 | 2.54 | 3.26 | 0.934 ± 0.014 |
| **GPU precisión mixta + `tf.data` + 32 pasos/llamada** | **12.086 ± 1.521** | **2.42** | **3.42** | **0.931 ± 0.011** |

* Costo único de inicializar CUDA/cuDNN en un proceso nuevo: **7.7–9.5 s**. No está en la tabla porque se paga una
  vez por programa, no por entrenamiento.
* La primera corrida de cada configuración sigue siendo ~3 s más lenta (14.8 s contra 11.4 s promedio de las demás en
  la versión final): TensorFlow traza el grafo y cuDNN prueba algoritmos para cada forma nueva de tensor, como el
  último batch incompleto de 20 imágenes. Sin esa corrida, el speedup estable es 40.57 / 11.41 = **×3.56**.
* La accuracy queda en el mismo rango que en CPU. La precisión mixta no degrada el modelo, porque los pesos y la
  pérdida siguen en FP32.

Con el **dataset completo** (60 900 imágenes de entrenamiento, 3 corridas en GPU y 2 en CPU, ambos con `tf.data`):

| Configuración | Entrenamiento (s) | s/época | Speedup | Accuracy prueba |
|---|---|---|---|---|
| CPU, 32 hilos | 193.280 ± 2.976 | 38.66 | 1.00 | 0.982 |
| GPU final | 39.085 ± 3.414 | 7.82 | **4.95** | 0.988 |

Con más datos el speedup sube de ×3.4 a ×5.0: los costos fijos (trazado, autotuning, validación) pesan menos y la GPU
pasa más tiempo calculando. El uso promedio de la GPU durante el entrenamiento sube de 36 % a 55 %, y la memoria de
2.2 GB a 4.2 GB (`logs/gpu_training_*.csv`).

### 5.2 Transferencias CPU ↔ GPU

`measure_transfers` copia el conjunto de entrenamiento completo a la GPU y de regreso, y sincroniza los dispositivos
antes y después (`tf.test.experimental.sync_devices`):

| Datos | Tamaño | CPU → GPU | GPU → CPU |
|---|---|---|---|
| `float32` normalizado | 570.9 MB | 0.096 s (5.8 GB/s) | 1.018 s (0.56 GB/s) |
| `uint8` sin normalizar | 142.7 MB | 0.028 s (5.5 GB/s) | 0.157 s (0.96 GB/s) |

Subir todo el conjunto cuesta menos del 1 % del tiempo de entrenamiento. La bajada es más lenta porque TensorFlow
copia a memoria paginable y luego crea el arreglo de NumPy, pero en entrenamiento casi no se baja nada: solo la
pérdida y las métricas. En la inferencia con Numba, las copias son 17.5 ms de subida y 0.4 ms de bajada, contra
271 ms de kernels.

### 5.3 Kernels propios en Numba CUDA (inferencia de las 2 610 imágenes de prueba)

| Implementación | Tiempo (s) | Speedup vs. NumPy |
|---|---|---|
| NumPy en CPU (misma matemática, vectorizada) | 18.744 ± 1.219 | 1 |
| Numba CUDA ingenuo | 0.566 ± 0.001 | 33 |
| **Numba CUDA con memoria compartida** | **0.289 ± 0.002** | **65** |
| Keras sobre cuDNN (GPU, ya trazado) | 0.207 ± 0.042 | 90 |

Tiempo por kernel para un batch de 256 imágenes (promedio de 20 mediciones con `cuda.event`,
`results/kernels_profile.csv`):

| Kernel | Ingenuo (ms) | Memoria compartida (ms) | Mejora |
|---|---|---|---|
| conv1 + ReLU | 4.96 | 2.18 | ×2.3 |
| conv2 + ReLU | 22.14 | 9.98 | ×2.2 |
| conv3 + ReLU | 21.95 | 9.87 | ×2.2 |
| dense1 + ReLU | 2.29 | 2.27 | — |
| max-pooling (3 capas) | 1.69 | 1.69 | — |
| reordenar, aplanar, dense2, softmax | 0.57 | 0.54 | — |

* **Validación:** las probabilidades de los kernels difieren de las de Keras en 2.7·10⁻³ como máximo, y la clase
  predicha coincide en 2 609 de 2 610 imágenes (accuracy 0.9337 contra 0.9333). Contra la versión NumPy en FP32 la
  diferencia es de 10⁻⁵. La de Keras es mayor porque cuDNN usa TF32 en los *tensor cores* de Ada, que redondea las
  multiplicaciones a 10 bits de mantisa.
* **Las convoluciones son más del 80 % del tiempo** y ahí la memoria compartida duplica el rendimiento. Cada valor de
  entrada se lee una vez de la memoria global por bloque y se reutiliza 9 × 8 = 72 veces desde la compartida y los
  registros.
* **cuDNN sigue siendo más rápido.** Usa algoritmos más avanzados (Winograd/GEMM implícito), *tensor cores* y kernels
  fusionados, ajustados para cada arquitectura. Nuestra versión llega a 1.4 veces su tiempo con 40 líneas de CUDA
  Python, lo que muestra el valor de diseñar bien los bloques y la memoria, y también por qué en la práctica se
  entrena con las librerías de NVIDIA.
* La primera llamada (compilación JIT de los kernels a PTX y luego a SASS) tarda 1.7 s. Es un costo único.

Evidencia: `evidence/04_optimizada_entrenamiento_gpu.png` es una captura de `btop` y `nvidia-smi` durante el
entrenamiento optimizado. Se ve la GPU al 76 %, 52 W, 708 MB en uso y el proceso `.venv/bin/python` con memoria
reservada en la GPU, mientras los núcleos del CPU quedan casi libres (4 %).
