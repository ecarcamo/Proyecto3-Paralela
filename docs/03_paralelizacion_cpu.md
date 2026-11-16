# 3. Identificación de oportunidades de paralelización

**Responsable de esta sección y de la preparación de datos en paralelo:** Nicolás Concuá (`NicoCT04`)

## 3.1 Qué se puede ejecutar de forma simultánea

Se tomaron las operaciones de `docs/01_analisis_secuencial.md` y se clasificaron según el tipo de trabajo y el
hardware que mejor le queda:

| Operación | ¿Independiente? | Tipo de trabajo | Dónde se ejecuta | Justificación |
|---|---|---|---|---|
| Listado y split | — | Metadatos (0.02 s) | CPU, 1 hilo | Es tan barato que repartirlo costaría más que hacerlo |
| Lectura + decodificación JPEG | Sí, cada imagen es independiente | Entrada/salida y descompresión con ramas, tablas Huffman y punteros | **CPU, varios procesos** | Código de bibliotecas de C (libjpeg) con mucho control de flujo: la GPU no lo ejecuta bien y el archivo vive en la memoria del CPU |
| Redimensionamiento 64×64 | Sí | Filtro bicúbico sobre una imagen pequeña | **CPU, varios procesos** | Viene pegado a la decodificación; mandar 120 000 bytes por imagen a la GPU para reducirlos costaría más que hacerlo aquí |
| Normalización ÷255 | Sí | Aritmética vectorizada | **CPU** (NumPy o `tf.data`) | Una sola pasada vectorizada sobre memoria contigua; tarda décimas de segundo |
| Convoluciones, densas, softmax, retropropagación y Adam | Sí, dentro de cada batch (píxeles, canales, imágenes) | Millones de multiplicaciones-sumas idénticas | **GPU con CUDA** | Paralelismo de datos masivo y regular: miles de hilos haciendo la misma operación sobre datos distintos |
| Validación e inferencia | Sí | Igual que la propagación hacia adelante | **GPU con CUDA** | Mismo motivo |

Hay dos niveles de paralelismo distintos:

1. **Paralelismo de tareas (CPU):** 17 400 imágenes que se procesan sin depender unas de otras. Cada tarea es
   pesada (~0.47 ms), irregular y vive en archivos del sistema. Encaja con procesos sobre los 24 núcleos físicos.
2. **Paralelismo de datos (GPU):** dentro de cada paso de la red hay millones de operaciones idénticas. Encaja con los
   miles de hilos CUDA de la GPU (ver `docs/04_diseno_cuda.md`).

El GIL de Python impide que hilos de Python decodifiquen imágenes en paralelo de verdad. Por eso la preparación usa
**procesos** (`multiprocessing`) y no hilos.

## 3.2 Diseño de la preparación de datos en paralelo

Implementación: `src/parallel_cpu/dataset.py`.

```
proceso principal                                      procesos trabajadores (N)
─────────────────                                      ──────────────────────────
crea SharedMemory de N_img × 64 × 64 × 3 bytes   ──►   cada uno se conecta al bloque (initializer)
reparte tareas de 64 rutas (start, rutas)        ──►   abre JPG → resize 64×64 → escribe en pixels[start + i]
imap_unordered: el primero libre toma la siguiente     devuelve solo cuántas imágenes procesó
copia los uint8 y normaliza ÷255 en float32
```

Decisiones:

* **Memoria compartida (`multiprocessing.shared_memory`)**: los trabajadores escriben los píxeles directamente en un
  bloque compartido y no devuelven arreglos. Así se evita serializar con `pickle` 214 MB de píxeles de vuelta al
  proceso principal.
* **Tareas de 64 imágenes con `imap_unordered`**: balanceo dinámico. Los núcleos E (más lentos) toman menos tareas
  que los P sin que nadie espere a nadie, y cada mensaje entre procesos amortiza su costo en 64 imágenes.
* **`uint8` en la memoria compartida**: 4 veces menos memoria que `float32`. La normalización queda al final y es
  vectorizada.
* **`fork` en lugar de `spawn`**: ver 3.4. Es el hallazgo más importante de esta parte.
* **Resultado idéntico**: `results/cpu_parallel_equivalence.json` compara contra `load_set` original. Los tres
  conjuntos tienen `max_abs_diff = 0.0`, con píxeles y etiquetas idénticos bit a bit.

Para el entrenamiento, `src/parallel_cpu/pipeline.py` arma un `tf.data.Dataset` que vive en el CPU
(`tf.device("/CPU:0")`). Baraja, arma batches y convierte `uint8 → float32 ÷255` con `num_parallel_calls=AUTOTUNE`,
y con `prefetch` prepara el siguiente batch mientras la GPU entrena el actual. Así el CPU y la GPU trabajan en paralelo
en vez de turnarse. Además, el conjunto completo se guarda en memoria como `uint8` (150 MB en vez de 600 MB) y cada
batch se normaliza justo antes de enviarse. Se verificó que entrega exactamente los mismos tensores que la
normalización con NumPy.

## 3.3 Resultados

`benchmarks/bench_cpu_parallel.py`: 5 corridas por cantidad de procesos, con la misma pasada de calentamiento de la
caché de disco que la medición secuencial. El tiempo incluye crear el pool, decodificar, copiar y normalizar.

### Submuestra del laboratorio (17 400 imágenes)

| Procesos | Tiempo (s) | Speedup vs. 1 proceso | Eficiencia | Speedup vs. `load_set` secuencial (8.125 s) |
|---|---|---|---|---|
| 1 | 7.840 ± 0.418 | 1.00 | 1.00 | 1.04 |
| 2 | 4.125 ± 0.109 | 1.90 | 0.95 | 1.97 |
| 4 | 2.327 ± 0.054 | 3.37 | 0.84 | 3.49 |
| 8 | 1.468 ± 0.015 | 5.34 | 0.67 | 5.53 |
| 12 | 1.190 ± 0.008 | 6.59 | 0.55 | 6.83 |
| 16 | 1.032 ± 0.012 | 7.60 | 0.47 | 7.87 |
| **24** | **0.898 ± 0.097** | **8.73** | **0.36** | **9.05** |
| 32 | 1.187 ± 0.209 | 6.60 | 0.21 | 6.84 |

### Dataset completo (87 000 imágenes)

| Procesos | Tiempo (s) | Speedup vs. 1 proceso |
|---|---|---|
| 1 | 39.836 ± 1.415 | 1.00 |
| 4 | 12.663 ± 0.747 | 3.15 |
| 8 | 7.718 ± 0.132 | 5.16 |
| 16 | 5.217 ± 0.116 | 7.64 |
| **24** | **4.323 ± 0.085** | **9.22** |
| 32 | 4.673 ± 0.359 | 8.52 |

Uso del CPU durante la carga (monitor de núcleos, `logs/cpu_parallel_subset.csv`):

| Procesos | Uso promedio por núcleo | Núcleos con más de 50 % |
|---|---|---|
| 1 | 4.4 % | 0 |
| 8 | 23.4 % | 6 |
| 24 | 53.7 % | 21 |
| 32 | 64.7 % | 32 |

## 3.4 Análisis

* **El óptimo está en 24 procesos, uno por núcleo físico.** Pasar a 32 empeora: los 8 hilos extra son
  *hyper-threads* de los núcleos P, que comparten unidades de ejecución y caché, y la contención es mayor que la
  ganancia.
* **La eficiencia cae porque los núcleos no son iguales y hay trabajo que no se reparte.** Los 16 núcleos E rinden
  bastante menos que los 8 P, así que 24 procesos no equivalen a 24 «núcleos P». Además, crear el pool, copiar el
  bloque compartido y normalizar ocurren en el proceso principal (ley de Amdahl). Con el dataset completo el trabajo
  fijo pesa menos y el speedup sube a 9.2×.
* **`spawn` contra `fork`.** La primera versión usaba `spawn`, y 32 procesos tardaban **9.2 s**, más que la versión
  secuencial (`results/cpu_parallel_subset_spawn.csv`). Con `spawn`, cada trabajador arranca un intérprete nuevo y
  vuelve a importar el módulo principal, con scikit-learn, NumPy y PIL incluidos. Ese costo crece con el número de
  procesos y se come la ganancia: el mejor caso era 8 procesos con 4.8 s. Con `fork`, los trabajadores heredan la
  memoria del padre ya inicializada y arrancan en milisegundos. En Linux esa es la opción correcta para este trabajo.

| Procesos | `spawn` (s) | `fork` (s) |
|---|---|---|
| 1 | 9.874 | 7.840 |
| 8 | 4.781 | 1.468 |
| 24 | 6.849 | 0.898 |
| 32 | 9.165 | 1.187 |

Conclusión de esta parte: procesar los archivos con los núcleos del CPU reduce la preparación de datos de 8.1 s a
0.9 s (**×9.05**). El resultado es idéntico al original y deja los datos listos para que la GPU se encargue
únicamente de la red neuronal.
