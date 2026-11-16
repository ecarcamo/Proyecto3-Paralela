# Proyecto 3 — Mejorando algoritmos con CUDA

**CC3069 Computación Paralela y Distribuida · Universidad del Valle de Guatemala · Semestre II 2026**
Diego López (`dlop6`) · Nicolás Concuá (`NicoCT04`) · Esteban Cárcamo (`ecarcamo`) · 15 de noviembre de 2026

> Algoritmo acelerado: **Laboratorio 3 de Data Science, clasificador del alfabeto de señas americano (ASL)** con una
> red neuronal convolucional en TensorFlow/Keras. Repositorio: `github.com/ecarcamo/Proyecto3-Paralela`.

## 1. Resumen

El pipeline original lee 17 400 fotografías una por una en un solo hilo de Python y entrena la CNN en el CPU. Se
separó en dos mundos: los **archivos** se procesan con los **24 núcleos físicos del CPU** (`multiprocessing` con
memoria compartida) y la **matemática de la red neuronal** se ejecuta en la **GPU con CUDA**, tanto con TensorFlow
(cuDNN/cuBLAS, precisión mixta) como con kernels propios escritos en CUDA Python (Numba).

| Fase | Secuencial (s) | Optimizada (s) | Speedup |
|---|---|---|---|
| Preparación de datos (CPU) | 10.348 ± 2.124 | 1.224 ± 0.255 | **×8.45** |
| Entrenamiento, 5 épocas (GPU) | 41.312 ± 1.697 | 12.086 ± 1.521 | **×3.42** |
| Inferencia, 2 610 imágenes (GPU) | 0.698 ± 0.083 | 0.668 ± 0.247 | ×1.05 |
| **Total del pipeline** | **52.358 ± 2.648** | **14.007 ± 1.203** | **×3.74** |

Con el dataset completo (5 veces más imágenes), el entrenamiento acelera **×4.95** y la preparación **×9.47**. Los
kernels propios en Numba CUDA ejecutan la inferencia **65 veces** más rápido que la misma matemática en NumPy. La
accuracy no cambia: 0.933 en CPU y 0.931 en GPU.

## 2. Datos, hardware y procedimiento

**Datos.** *ASL Alphabet* de Kaggle (`grassknoted/asl-alphabet`): 87 000 imágenes JPG de 200×200 RGB en 29 clases
(A–Z, `del`, `nothing`, `space`). Igual que en el laboratorio se usa una submuestra de 600 imágenes por clase con
semilla 42 (17 400 imágenes, `setup_dataset.py`), reducidas a 64×64 y separadas 70/15/15 de forma estratificada
(12 180 de entrenamiento, 2 610 de validación y 2 610 de prueba). Para medir escalabilidad también se usó el dataset
completo (60 900 / 13 050 / 13 050).

**Modelo.** La CNN base que ganó en el laboratorio: Conv(32)–Pool–Conv(64)–Pool–Conv(128)–Pool–Dense(128)–Dense(29),
con 1 145 693 parámetros y 42.3 M multiplicaciones-sumas por imagen. Adam con lr = 1e-3, batch de 32.

| Hardware | |
|---|---|
| CPU | Intel Core i9-13980HX: 24 núcleos (8P + 16E), 32 hilos |
| GPU | NVIDIA RTX 4050 Laptop: 20 SM, 6 GB, compute capability 8.9 |
| Software | Python 3.11, TensorFlow 2.21 (CUDA 12.5, cuDNN 9), Numba 0.68 + numba-cuda 0.30, driver 620.70 |

**Procedimiento.**

* Todas las mediciones salen de una sola ejecución de `scripts/run_all_benchmarks.sh`, con la laptop conectada a la
  corriente y sin otras cargas pesadas.
* En la submuestra hay 5 corridas por configuración (3 en el dataset completo) y se reporta promedio ± desviación
  estándar.
* El entrenamiento usa siempre **5 épocas fijas** sin paro temprano, para que todas las versiones hagan el mismo
  trabajo.
* Antes de medir se calienta la caché de disco (una carga sin medir) y, en GPU, se hace un entrenamiento corto sin
  medir que absorbe la inicialización de CUDA/cuDNN (7.7–9.5 s, que se reporta aparte).
* Durante cada corrida, un hilo registra el uso de los 32 núcleos lógicos (`psutil`) y otro el de la GPU
  (`nvidia-smi` cada 250 ms). Todo queda en `logs/`.

**Decisiones tomadas:**

* La referencia en CPU es el código original con TensorFlow usando los 32 hilos, que es la versión más rápida. La
  versión estrictamente secuencial (1 hilo) también se midió: es ×4.0 más lenta, pero compararse contra ella inflaría
  el speedup.
* El speedup se calcula por fase y para el pipeline completo.
* El total optimizado suma la transferencia explícita del conjunto de entrenamiento a la GPU.

## 3. Análisis de la solución secuencial (pasos 1 y 2)

Detalle completo en `docs/01_analisis_secuencial.md`. El programa se descompone en: listado de archivos, split
estratificado, lectura + decodificación JPEG, redimensionamiento, normalización ÷255, codificación de etiquetas,
propagación hacia adelante, pérdida, retropropagación, actualización Adam, validación por época e inferencia.

| Operación repetitiva | Datos que procesa | Repeticiones por corrida |
|---|---|---|
| Lectura + decodificación + resize | JPG de 200×200×3 → 64×64×3 | 17 400 |
| Normalización ÷255 | 12 288 valores por imagen | 213.8 M divisiones |
| Paso de entrenamiento (adelante + atrás + Adam) | Batch de 32 imágenes, 1.1 M parámetros | 381 por época × 5 = 1 905 |
| Propagación (convoluciones + densas) | 42.3 M MAC por imagen | 60 900 de entrenamiento + 13 050 de validación + 2 610 de prueba |

| Fase (original, CPU) | Tiempo (s) | % del total | Núcleos ocupados |
|---|---|---|---|
| Preparación de datos | 10.348 ± 2.124 | 20 % | ≈ 1.6 de 32 |
| Entrenamiento | 41.312 ± 1.697 | 79 % | ≈ 19 de 32 |
| Inferencia | 0.698 ± 0.083 | 1 % | ≈ 15 de 32 |

La preparación es un bucle de 17 400 iteraciones independientes que corre en un solo núcleo. El entrenamiento domina
el tiempo y TensorFlow ya lo reparte en hilos, pero de 1 a 32 hilos solo acelera ×4.0 (164.5 s → 41.3 s). Factores
observados: temperatura (CPU entre 90 y 96 °C), caché de disco, GIL de Python, núcleos híbridos P/E y batches pequeños.

## 4. Oportunidades de paralelización (paso 3)

Detalle en `docs/03_paralelizacion_cpu.md`.

| Operación | Paralelismo | Dónde | Por qué |
|---|---|---|---|
| Decodificar + redimensionar JPG | De tareas: cada imagen es independiente | CPU, 24 procesos | Trabajo irregular con ramas (Huffman, libjpeg) y archivos en memoria del host; el GIL obliga a usar procesos |
| Normalizar, barajar, armar batches | Vectorizado y en tubería | CPU (`tf.data`) | Barato; se solapa con el cómputo de la GPU gracias a `prefetch` |
| Convoluciones, densas, softmax, retropropagación, Adam | De datos: millones de MAC idénticas | GPU (CUDA) | Miles de hilos haciendo la misma operación; *tensor cores* para FP16 |

## 5. Diseño del algoritmo paralelo (paso 4)

Detalle en `docs/04_diseno_cuda.md`.

**CPU.** El proceso principal crea un bloque de `multiprocessing.shared_memory` de N×64×64×3 bytes y reparte tareas
de 64 rutas con `imap_unordered` (balanceo dinámico entre núcleos P y E). Cada trabajador decodifica, redimensiona y
escribe los píxeles directamente en el bloque. El pool se crea con `fork` **antes** de inicializar TensorFlow y se
reutiliza. Con `spawn`, 32 procesos tardaban 9.6 s, porque cada uno volvía a importar las librerías.

**GPU con TensorFlow.** `tf.data` en el CPU (barajar, batch de 32, `uint8 → float32`, `prefetch`) alimenta a la GPU.
Ahí la política `mixed_float16` usa los *tensor cores* y `steps_per_execution=32` despacha 32 pasos de
entrenamiento por cada llamada de Python. cuDNN elige la configuración de cada kernel de convolución.

**GPU con kernels propios (CUDA Python / Numba).** Batches de 256 imágenes en formato NCHW:

| Kernel | Un hilo calcula | Bloque | Grid |
|---|---|---|---|
| `conv3x3_relu_tiled` (3 capas) | 1 píxel × 8 canales de salida, con un mosaico de 18×18 en memoria compartida | 16×16 | (⌈W/16⌉, ⌈H/16⌉, B·C_out/8) |
| `maxpool2x2` | 1 píxel de salida | 16×16 (8×8 en la última) | (⌈W/16⌉, ⌈H/16⌉, B·C) |
| `dense` (+ReLU) | 1 neurona de 1 imagen; la entrada en mosaicos de 128 en memoria compartida | 128 | (⌈N_out/128⌉, B) |
| `softmax_rows` | 1 imagen completa | 128 | (⌈B/128⌉) |

El eje `z` del grid codifica imagen y canal, así que todos los hilos de un bloque usan los mismos pesos (difusión desde
la caché) y los hilos vecinos en `x` leen memoria contigua (accesos agrupados). Los tiempos se miden con `cuda.event`,
separando las copias CPU→GPU, los kernels y las copias GPU→CPU.

## 6. Resultados (paso 5)

### 6.1 Tabla comparativa de tiempos y speedup por fase

![Tiempo por fase](../figures/01_tiempo_por_fase.png)

| Fase | Secuencial (s) | Optimizada (s) | Speedup | Dataset completo |
|---|---|---|---|---|
| Preparación de datos | 10.348 ± 2.124 | 1.224 ± 0.255 | ×8.45 | ×9.47 (35.70 → 3.77 s) |
| Transferencia CPU→GPU del conjunto | — | 0.028 | — | 0.34 s |
| Entrenamiento (5 épocas) | 41.312 ± 1.697 | 12.086 ± 1.521 | ×3.42 | ×4.95 (193.28 → 39.09 s) |
| Inferencia | 0.698 ± 0.083 | 0.668 ± 0.247 | ×1.05 | — |
| **Total** | **52.358** | **14.007** | **×3.74** | — |

* La preparación optimizada (1.224 s) es la medida dentro del pipeline completo. Sola, en su benchmark, tarda
  1.042 s (**×9.93**).
* En entrenamiento, sin la primera corrida (trazado del grafo y autotuning de cuDNN para formas nuevas) el speedup
  estable es **×3.56**.
* La inferencia de 2 610 imágenes casi no mejora de punta a punta, porque la domina el costo fijo de preparar el grafo
  en cada `predict`. Con el modelo ya trazado, Keras en GPU tarda 0.207 s.

### 6.2 Preparación de datos en el CPU

![Escalabilidad de la carga](../figures/02_escalabilidad_carga_cpu.png)

El óptimo son 24 procesos, uno por núcleo físico. Con 32 (*hyper-threading*) no mejora o empeora. La eficiencia cae
porque los núcleos E son más lentos y porque crear el pool, copiar y normalizar quedan en el proceso principal (ley de
Amdahl). Los píxeles son **idénticos bit a bit** a los de la versión original (`results/cpu_parallel_equivalence.json`).

### 6.3 Entrenamiento en la GPU

![Entrenamiento CPU vs GPU](../figures/03_entrenamiento_cpu_vs_gpu.png)

Cada optimización suma: mover el mismo código a la GPU da ×2.54, `tf.data` con `prefetch` ×2.76, la precisión mixta
×3.26 y `steps_per_execution` ×3.42. La GPU estuvo ocupada en promedio solo el 36 % del entrenamiento con la
submuestra y el 55 % con el dataset completo. El modelo es pequeño y la GPU espera a que el CPU le despache cada paso.

### 6.4 Kernels propios en CUDA Python

![Kernels Numba](../figures/06_kernels_numba.png)

| Implementación (2 610 imágenes) | Tiempo (s) | Speedup vs. NumPy |
|---|---|---|
| NumPy en CPU | 18.744 ± 1.219 | 1 |
| Numba CUDA ingenuo | 0.566 | 33 |
| **Numba CUDA con memoria compartida** | **0.289** | **65** |
| Keras sobre cuDNN | 0.207 | 90 |

La versión con memoria compartida y 8 canales por hilo duplica la velocidad de las convoluciones, que son más del
80 % del tiempo. Las predicciones coinciden con Keras en 2 609 de 2 610 imágenes; la diferencia máxima de probabilidad
es de 2.7·10⁻³, por el TF32 que usa cuDNN. De las 0.289 s, solo 17.5 ms son copia a la GPU y 0.4 ms copia de regreso.

### 6.5 Transferencias CPU ↔ GPU

| Conjunto de entrenamiento | Tamaño | CPU → GPU | GPU → CPU |
|---|---|---|---|
| `float32` | 570.9 MB | 0.096 s (5.8 GB/s) | 1.018 s (0.56 GB/s) |
| `uint8` | 142.7 MB | 0.028 s (5.5 GB/s) | 0.157 s (0.96 GB/s) |

## 7. Evidencia del uso de recursos

Capturas de `btop` (los 32 núcleos y la GPU) y de `nvidia-smi`, tomadas mientras corría cada fase
(`scripts/demo_workload.py`):

**Carga secuencial: un solo núcleo trabaja** (C10 al 53 %, el resto en reposo).
![Carga secuencial](../evidence/01_secuencial_carga_un_nucleo.png)

**Carga paralela con 24 procesos: todos los núcleos trabajan.**
![Carga paralela](../evidence/02_paralela_carga_24_procesos.png)

**Entrenamiento original en CPU: los 32 hilos ocupados y la GPU en 0 %.**
![Entrenamiento CPU](../evidence/03_secuencial_entrenamiento_cpu.png)

**Entrenamiento optimizado: la GPU al 76 %, 52 W y el proceso `python` con memoria en la GPU; el CPU al 4 %.**
![Entrenamiento GPU](../evidence/04_optimizada_entrenamiento_gpu.png)

Los monitores de las corridas medidas confirman lo mismo con todas las muestras:

![Uso de núcleos](../figures/04_uso_nucleos_cpu.png)

![Uso de GPU](../figures/05_uso_gpu_entrenamiento.png)

## 8. Discusión técnica
