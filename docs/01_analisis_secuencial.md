# 1. Análisis de la solución secuencial existente

**Responsable:** Diego López (`dlop6`)

## 1.1 Partes del programa original

El Laboratorio 3 de Data Science (`notebooks/ASL_Clasificador.ipynb`, rama `lab3` del repositorio del curso) clasifica
las 29 clases del dataset *ASL Alphabet* de Kaggle (26 letras más `del`, `nothing` y `space`). Su código se trasladó sin
cambiar la lógica a `setup_dataset.py` y `src/sequential/`:

| Parte | Archivo | Función |
|---|---|---|
| Submuestreo del dataset | `setup_dataset.py` | Del dataset completo (87 000 imágenes de 200×200 RGB) copia 600 imágenes por clase con semilla 42: 17 400 imágenes |
| Listado y split | `src/sequential/dataset.py` (`list_images`, `split_dataset`) | Recorre las carpetas, arma la lista de rutas y etiquetas y hace un split estratificado 70/15/15 |
| Carga y preprocesamiento | `src/sequential/dataset.py` (`load_set`) | Un `for` de Python que, imagen por imagen, abre el JPG con PIL, lo reduce a 64×64, lo convierte a arreglo y lo divide entre 255 |
| Modelo | `src/sequential/models.py` (`build_cnn_base`) | La CNN base que ganó en el laboratorio: 3 bloques Conv2D+ReLU+MaxPool (32, 64 y 128 filtros), Dense(128) y softmax de 29 clases. 1 145 693 parámetros |
| Entrenamiento | `src/sequential/run_sequential.py` | `model.fit` con Adam (lr = 1e-3), batch 32, entropía cruzada y validación al final de cada época |
| Evaluación | `src/sequential/run_sequential.py` | `model.evaluate` / `model.predict` sobre el conjunto de prueba y accuracy |

Las únicas diferencias con el notebook son de reproducibilidad: las rutas salen de `src/config.py` y las listas de
archivos se ordenan antes de barajar y de hacer el split, para que el resultado no dependa del orden en que el sistema de
archivos devuelve los nombres.

## 1.2 Operaciones, desde la recepción de las imágenes hasta el resultado

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Listado de archivos | `os.listdir` por clase; produce 17 400 rutas y sus etiquetas de texto |
| 2 | Split estratificado | `train_test_split` dos veces (70 % y luego 50/50 del resto): 12 180 entrenamiento, 2 610 validación, 2 610 prueba |
| 3 | Lectura y decodificación JPEG | `Image.open` lee el archivo y decodifica los 200×200×3 píxeles comprimidos |
| 4 | Redimensionamiento | `resize((64, 64))` con filtro bicúbico: cada píxel de salida combina una vecindad de la imagen original |
| 5 | Conversión y normalización | `np.array(img) / 255.0`: 12 288 valores por imagen pasan de enteros 0–255 a flotantes 0–1 |
| 6 | Codificación de etiquetas | Diccionario `clase → índice` (0–28) |
| 7 | Propagación hacia adelante | Por cada batch de 32: 3 convoluciones 3×3, ReLU, max-pooling 2×2, aplanado, 2 capas densas y softmax |
| 8 | Cálculo de pérdida | Entropía cruzada categórica dispersa contra la etiqueta |
| 9 | Retropropagación | Gradientes de la pérdida respecto a los 1.1 M de parámetros (convoluciones transpuestas y productos matriz-vector) |
| 10 | Actualización Adam | Actualiza cada parámetro con sus momentos primero y segundo |
| 11 | Validación por época | Propagación hacia adelante de las 2 610 imágenes de validación |
| 12 | Inferencia y métrica | `predict` sobre las 2 610 imágenes de prueba, `argmax` y accuracy |

## 1.3 Operaciones repetitivas

Cifras para una corrida de medición (17 400 imágenes, 5 épocas, batch 32). Las multiplicaciones-acumulaciones (MAC)
por imagen en la propagación hacia adelante son 3.54 M (conv1) + 18.87 M (conv2) + 18.87 M (conv3) + 1.05 M (densas)
= **42.3 M MAC**. La retropropagación cuesta aproximadamente el doble de la propagación hacia adelante.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Lectura + decodificación JPEG | Archivo JPG de 200×200×3 (~14 KB) | 17 400 por corrida (una por imagen) |
| Redimensionamiento a 64×64 | 120 000 valores de entrada → 12 288 de salida | 17 400 por corrida |
| Normalización ÷255 | 12 288 valores por imagen | 17 400 × 12 288 = 213.8 M divisiones |
| Convolución 3×3 (3 capas) | Mapas de 64×64×3, 32×32×32 y 16×16×64 | 41.3 M MAC por imagen, por cada propagación |
| Producto matriz-vector (densas) | Vector de 8 192 → 128 → 29 | 1.05 M MAC por imagen, por cada propagación |
| Paso de entrenamiento (adelante + atrás + Adam) | Batch de 32 imágenes, 1 145 693 parámetros | 381 pasos por época × 5 épocas = 1 905 |
| Propagación de entrenamiento | 12 180 imágenes por época | 60 900 imágenes ≈ 7.7 TMAC entre adelante y atrás |
| Validación | 2 610 imágenes por época | 13 050 propagaciones ≈ 0.55 TMAC |
| Inferencia de prueba | 2 610 imágenes | 2 610 propagaciones ≈ 0.11 TMAC |

La carga de imágenes es un bucle de 17 400 iteraciones independientes entre sí. El entrenamiento está dominado por las
convoluciones: millones de productos punto pequeños e independientes dentro de cada batch. Las dos son candidatas
naturales a paralelizarse, aunque con hardware distinto (ver `docs/03_paralelizacion_cpu.md` y `docs/04_diseno_cuda.md`).

# 2. Medición del tiempo de ejecución secuencial

## 2.1 Procedimiento

`benchmarks/bench_sequential.py` ejecuta el pipeline original completo varias veces dentro de un mismo proceso, con
la GPU oculta (`CUDA_VISIBLE_DEVICES=-1`):

1. Una pasada de calentamiento de la carga, sin medir, para que los 17 400 JPG queden en la caché de páginas del
   sistema operativo. Así todas las corridas miden lo mismo: decodificar y procesar, no esperar al disco.
2. En cada corrida, con `time.perf_counter`, se miden cuatro fases: listado + split, preparación de datos (`load_set`
   sobre los tres conjuntos), entrenamiento de una CNN nueva y la inferencia sobre prueba.
3. El entrenamiento usa una cantidad fija de **5 épocas** sin paro temprano, para que todas las implementaciones hagan
   exactamente el mismo trabajo. La semilla cambia por corrida (42 + corrida) y la accuracy se reporta como control.
4. Un hilo aparte (`src/benchmark/cpu_monitor.py`) registra el uso de cada uno de los 32 núcleos lógicos cada 0.25 s.
   Así se ve cuántos núcleos trabaja cada fase (`logs/cpu_sequential*.csv`).

Se corrieron dos configuraciones:

* **Original (5 corridas):** tal cual el laboratorio. TensorFlow decide sus hilos y usa los 32 núcleos lógicos en
  las operaciones de la red (oneDNN).
* **Estrictamente secuencial (3 corridas):** `--threads 1` limita a TensorFlow a un hilo de operación
  (`intra_op = inter_op = 1`). Es la referencia de «un solo núcleo» para todo el programa.

```bash
.venv/bin/python -m benchmarks.bench_sequential --repeats 5 --epochs 5
.venv/bin/python -m benchmarks.bench_sequential --repeats 3 --epochs 5 --threads 1
```

## 2.2 Resultados

Tiempos en segundos (promedio ± desviación estándar) de la corrida final de `scripts/run_all_benchmarks.sh`. Fuente:
`results/sequential_runs.csv` y `results/sequential_runs_1thread.csv`.

| Fase | Original (32 hilos TF, 5 corridas) | Estrictamente secuencial (1 hilo, 3 corridas) |
|---|---|---|
| Listado + split | 0.026 ± 0.007 | 0.028 ± 0.008 |
| Preparación de datos (17 400 imágenes) | 10.348 ± 2.124 | 9.699 ± 2.861 |
| Entrenamiento (5 épocas, 12 180 imágenes) | 41.312 ± 1.697 | 164.541 ± 1.922 |
| Inferencia (2 610 imágenes) | 0.698 ± 0.083 | 2.241 ± 0.320 |
| **Total** | **52.38 ± 2.65** | **176.51 ± 2.93** |
| Accuracy en prueba | 0.933 ± 0.006 | 0.922 ± 0.017 |

Uso del CPU medido por el monitor (promedio de los 32 núcleos lógicos durante la fase, `logs/cpu_sequential*.csv`):

| Fase | Uso promedio por núcleo | Núcleos con más de 50 % | Núcleos equivalentes ocupados |
|---|---|---|---|
| Preparación de datos | 4.9 % | 0 | ≈ 1.6 |
| Entrenamiento (original) | 59.6 % | 32 | ≈ 19 |
| Entrenamiento (1 hilo) | 5.1 % | 0 | ≈ 1.6 |

`evidence/01_secuencial_carga_un_nucleo.png` y `evidence/03_secuencial_entrenamiento_cpu.png` son capturas de `btop`
tomadas mientras corrían esas fases. En la primera solo un núcleo trabaja; en la segunda trabajan todos.

Con una corrida completa sin límite de épocas (`run_sequential.py`, paro temprano con paciencia 3, 18 épocas) la CNN
llega a **0.9751** de accuracy en prueba (`logs/sequential_original_run.log`). Es coherente con el 0.9467 reportado en
el laboratorio, que se había corrido en otra máquina.

## 2.3 Análisis

* **La preparación de datos usa un solo núcleo.** El `for` de `load_set` procesa una imagen a la vez en el hilo
  principal de Python. El monitor muestra el equivalente a 1.6 núcleos ocupados de 32: uno por el bucle y el resto por
  ruido del sistema. Cuesta ~0.6 ms por imagen, sobre todo por la decodificación JPEG y el redimensionamiento
  bicúbico. Son 10 s que no dependen de la red neuronal y que los otros 23 núcleos físicos podrían repartirse.
* **El entrenamiento domina el tiempo total** (79 % en la versión original y 93 % con un hilo). Son 1 905 pasos de
  optimización, cada uno con ~127 M MAC por imagen entre propagación hacia adelante y hacia atrás.
* **TensorFlow ya paraleliza en CPU.** Pasar de 1 a 32 hilos acelera el entrenamiento ×4.0 (164.5 s → 41.3 s), muy
  lejos de ×32: los batches de 32 imágenes son pequeños, los 16 núcleos E son más lentos que los 8 P y la memoria es
  compartida. Por eso, para la comparación final contra la GPU se usa la versión original (32 hilos) como
  referencia: es la más rápida en CPU y no infla el speedup.

Factores que influyen en el rendimiento medido:

| Factor | Efecto observado o esperado |
|---|---|
| Temperatura y energía de la laptop | El CPU trabajó entre 90 y 96 °C. En una sesión de prueba previa, con la máquina más fría, el entrenamiento tardó 36.8 s en lugar de 41.3 s, y una corrida de carga llegó a 14.1 s: de ahí la desviación de ±2.1 s |
| Caché de páginas del disco | La primera lectura de los JPG es más lenta; por eso hay una pasada de calentamiento |
| Decodificación JPEG y `resize` en Python | Trabajo de CPU con el GIL tomado: no se paraleliza con hilos, sí con procesos |
| Hilos de TensorFlow (oneDNN) | Aceleran el entrenamiento ×4.0, pero escalan mal en batches pequeños |
| Núcleos híbridos 8P + 16E | Los núcleos E tienen menos rendimiento; el reparto no es homogéneo |
| Procesos de fondo | Se midió con la máquina sin otras cargas pesadas; aun así hay ~1 núcleo ocupado en reposo |
