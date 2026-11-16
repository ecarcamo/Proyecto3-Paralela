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
