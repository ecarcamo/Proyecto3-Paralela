# Proyecto 3 — Mejorando algoritmos con CUDA

CC3069 Computación Paralela y Distribuida · Universidad del Valle de Guatemala · Semestre II 2026

Aceleración del **Laboratorio 3 de Data Science: clasificador de letras del alfabeto de señas americano (ASL)**.
El laboratorio original carga las imágenes una por una en un solo hilo y entrena una CNN en TensorFlow/Keras.
Este proyecto separa el pipeline en dos mundos:

* **CPU con varios núcleos** para todo lo que es manipulación de archivos: leer los JPG, reducirlos a 64×64 y normalizarlos.
* **GPU con CUDA** exclusivamente para la matemática de la red neuronal: entrenamiento e inferencia.

## Equipo

| Integrante | Usuario | Responsabilidad |
|---|---|---|
| Diego López | `dlop6` | Línea base secuencial, medición y análisis comparativo |
| Nicolás Concuá | `NicoCT04` | Preparación de datos en paralelo con los núcleos del CPU |
| Esteban Cárcamo | `ecarcamo` | Entorno CUDA, entrenamiento en GPU y kernels propios con Numba CUDA |

Todo se ejecutó en una sola máquina (autorizado por el catedrático, porque las computadoras de dos integrantes se dañaron).

## Hardware

| Componente | Detalle |
|---|---|
| CPU | Intel Core i9-13980HX · 24 núcleos físicos (8P + 16E) · 32 hilos |
| GPU | NVIDIA GeForce RTX 4050 Laptop · 20 SM · 6 GB GDDR6 · compute capability 8.9 |
| RAM | 16 GB |
| Software | Python 3.11 · TensorFlow 2.21 (CUDA 12.5, cuDNN 9) · Numba 0.68 + numba-cuda 0.30 · driver 620.70 |

`python -m src.env_check` imprime este inventario y lo guarda en `results/environment.json`.

## Instalación

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m src.env_check
```

TensorFlow 2.21 trae sus librerías CUDA 12 como paquetes de pip. Para que encuentre `libcusolver`, el módulo
`src/gpu_runtime.py` la precarga antes de importar TensorFlow; por eso todo el código obtiene TensorFlow con
`load_tensorflow("gpu")` o `load_tensorflow("cpu")` (esta última oculta la GPU con `CUDA_VISIBLE_DEVICES=-1`).
Numba usa las mismas librerías CUDA 12 (`numba-cuda[cu12]`), así que ambos conviven en un mismo proceso.

## Entregables

| Entregable | Archivo |
|---|---|
| Código secuencial original y versión optimizada (notebook ejecutado) | `notebooks/proyecto3.ipynb` |
| Informe de rendimiento y análisis | `report/informe.pdf` (fuente: `report/informe.md`) |
| Detalle por paso | `docs/01_analisis_secuencial.md` (pasos 1–2), `docs/03_paralelizacion_cpu.md` (paso 3), `docs/04_diseno_cuda.md` (pasos 4–5) |
| Tablas y speedup | `results/summary.md`, `results/*.csv` |
| Figuras | `figures/` |
| Capturas de `btop` y `nvidia-smi` | `evidence/` |
| Logs de cada corrida y monitores de CPU/GPU | `logs/` |

## Resultados principales

| Fase | Secuencial (s) | Optimizada (s) | Speedup |
|---|---|---|---|
| Preparación de datos (CPU, 24 procesos) | 10.348 | 1.224 | ×8.45 |
| Entrenamiento, 5 épocas (GPU) | 41.312 | 12.086 | ×3.42 |
| Inferencia (GPU) | 0.698 | 0.668 | ×1.05 |
| **Total** | **52.358** | **14.007** | **×3.74** |

Con el dataset completo (87 000 imágenes) el entrenamiento acelera ×4.95. Los kernels propios en Numba CUDA hacen la
inferencia ×65 más rápido que NumPy.

## Estructura

```
setup_dataset.py            submuestra de 600 imágenes por clase (igual que el Lab 3)
src/config.py               rutas e hiperparámetros
src/gpu_runtime.py          carga TensorFlow en CPU o GPU
src/env_check.py            inventario de hardware
src/sequential/             código original del Lab 3 (carga con PIL y CNN base)
src/parallel_cpu/           carga paralela con multiprocessing + pipeline tf.data
src/gpu/train.py            entrenamiento en GPU (precisión mixta, tf.data, steps_per_execution)
src/gpu/kernels.py          kernels CUDA en Numba (convolución, pooling, densas, softmax)
src/gpu/numba_forward.py    CNN hacia adelante con esos kernels
src/benchmark/              medición por fases y monitores de CPU y GPU
benchmarks/                 benchmarks de cada fase
scripts/run_all_benchmarks.sh   corrida final de todas las mediciones
scripts/demo_workload.py    carga sostenida para tomar capturas de los monitores
analysis/compare.py         tablas de speedup y figuras
```

## Reproducir

```bash
.venv/bin/python setup_dataset.py
./scripts/run_all_benchmarks.sh
.venv/bin/python -m analysis.compare
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto3.ipynb
.venv/bin/python report/build_report.py
```
