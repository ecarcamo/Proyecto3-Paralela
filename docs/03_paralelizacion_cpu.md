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

`benchmarks/bench_cpu_parallel.py`, ejecutado en la corrida final de `scripts/run_all_benchmarks.sh`: 5 corridas por
cantidad de procesos (3 en el dataset completo), con la misma pasada de calentamiento de la caché de disco que la
medición secuencial. El tiempo incluye crear el pool, decodificar, copiar y normalizar.

### Submuestra del laboratorio (17 400 imágenes)

| Procesos | Tiempo (s) | Speedup vs. 1 proceso | Eficiencia | Speedup vs. `load_set` secuencial (10.348 s) |
|---|---|---|---|---|
| 1 | 7.681 ± 0.239 | 1.00 | 1.00 | 1.35 |
| 2 | 4.257 ± 0.059 | 1.80 | 0.90 | 2.43 |
| 4 | 2.463 ± 0.071 | 3.12 | 0.78 | 4.20 |
| 8 | 1.658 ± 0.096 | 4.63 | 0.58 | 6.24 |
| 12 | 1.402 ± 0.064 | 5.48 | 0.46 | 7.38 |
| 16 | 1.287 ± 0.119 | 5.97 | 0.37 | 8.04 |
| **24** | **1.042 ± 0.045** | **7.37** | **0.31** | **9.93** |
| 32 | 1.016 ± 0.031 | 7.56 | 0.24 | 10.19 |

### Dataset completo (87 000 imágenes)

| Procesos | Tiempo (s) | Speedup vs. 1 proceso |
|---|---|---|
| 1 | 35.704 ± 2.029 | 1.00 |
| 4 | 11.177 ± 0.183 | 3.19 |
| 8 | 6.930 ± 0.235 | 5.15 |
| 16 | 4.687 ± 0.200 | 7.62 |
| **24** | **3.772 ± 0.106** | **9.47** |
| 32 | 4.048 ± 0.149 | 8.82 |

Uso del CPU durante la carga (monitor de núcleos, `logs/cpu_parallel_subset.csv`; captura de `btop` en
`evidence/02_paralela_carga_24_procesos.png`):

| Procesos | Uso promedio por núcleo | Núcleos con más de 50 % |
|---|---|---|
| 1 | 4.2 % | 0 |
| 8 | 21.2 % | 3 |
| 24 | 49.5 % | 21 |
| 32 | 59.8 % | 32 |

Dentro del pipeline completo (`bench_training`), el pool se crea una vez al inicio del programa, antes de cargar
TensorFlow, y se reutiliza. Ahí la preparación entrega `uint8` sin normalizar (la normalización la hace `tf.data`) y
tarda 1.224 ± 0.255 s.

## 3.4 Análisis

* **El óptimo está en 24 procesos, uno por núcleo físico.** Con 32 el tiempo es prácticamente igual en la submuestra
  (1.016 contra 1.042 s, dentro de la desviación) y peor en el dataset completo (4.05 contra 3.77 s). Los 8 hilos
  extra son *hyper-threads* de los núcleos P, que comparten unidades de ejecución y caché: agregan contención casi sin
  agregar cómputo. Por eso `default_workers()` usa la cantidad de núcleos físicos.
* **La eficiencia cae porque los núcleos no son iguales y hay trabajo que no se reparte.** Los 16 núcleos E rinden
  bastante menos que los 8 P, así que 24 procesos no equivalen a 24 «núcleos P». Además, crear el pool, copiar el
  bloque compartido y normalizar ocurren en el proceso principal (ley de Amdahl). Con el dataset completo el trabajo
  fijo pesa menos y el speedup sube a 9.5×.
* **Hay variación por temperatura.** El CPU estuvo entre 90 y 96 °C durante las mediciones; la carga secuencial tuvo
  una corrida de 14.1 s. Por eso el speedup contra la versión secuencial (×9.93) es algo mayor que contra el mismo
  código con 1 proceso (×7.37).
* **`spawn` contra `fork`.** La primera versión usaba `spawn`, y 32 procesos tardaban **9.6 s**, igual que la versión
  secuencial (`results/cpu_parallel_subset_spawn.csv`). Con `spawn`, cada trabajador arranca un intérprete nuevo y
  vuelve a importar el módulo principal, con scikit-learn, NumPy y PIL incluidos. Ese costo crece con el número de
  procesos y se come la ganancia: el mejor caso era 8 procesos con 5.0 s. Con `fork`, los trabajadores heredan la
  memoria del padre ya inicializada y arrancan en milisegundos. En Linux esa es la opción correcta para este trabajo,
  siempre que el pool se cree **antes** de inicializar TensorFlow y CUDA: un `fork` de un proceso que ya tiene el
  contexto CUDA copia un proceso enorme y, en nuestras pruebas, la carga subió a entre 1.3 y 5 s.

| Procesos | `spawn` (s) | `fork` (s) |
|---|---|---|
| 1 | 9.796 | 7.681 |
| 8 | 4.984 | 1.658 |
| 24 | 7.102 | 1.042 |
| 32 | 9.554 | 1.016 |

Conclusión de esta parte: procesar los archivos con los núcleos del CPU reduce la preparación de datos de 10.3 s a
1.0 s (**×9.9**). El resultado es idéntico al original y deja los datos listos para que la GPU se encargue
únicamente de la red neuronal.
