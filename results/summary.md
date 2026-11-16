# Resumen de resultados

Generado por `python -m analysis.compare`. Promedio ± desviación estándar.

## Tiempo por fase y speedup

Secuencial: código original del Lab 3 en CPU. Optimizada: carga con 24 procesos + entrenamiento e inferencia en GPU con precisión mixta, `tf.data` y 32 pasos por llamada. El total optimizado suma además la transferencia explícita CPU→GPU del conjunto de entrenamiento.

| fase | secuencial_s | optimizada_s | speedup |
|---|---|---|---|
| Preparación de datos (CPU) | 10.348 ± 2.124 | 1.224 ± 0.255 | 8.45 |
| Entrenamiento (5 épocas) | 41.312 ± 1.697 | 12.086 ± 1.521 | 3.42 |
| Inferencia (2 610 imágenes) | 0.698 ± 0.083 | 0.668 ± 0.247 | 1.05 |
| Total del pipeline | 52.358 ± 2.648 | 14.007 ± 1.203 | 3.74 |

## Escalabilidad de la preparación de datos en CPU

| workers | tiempo_s | speedup_vs_1_proceso | eficiencia | speedup_vs_secuencial |
|---|---|---|---|---|
| 1 | 7.681 ± 0.239 | 1.0 | 1.0 | 1.35 |
| 2 | 4.257 ± 0.059 | 1.8 | 0.9 | 2.43 |
| 4 | 2.463 ± 0.071 | 3.12 | 0.78 | 4.2 |
| 8 | 1.658 ± 0.096 | 4.63 | 0.58 | 6.24 |
| 12 | 1.402 ± 0.064 | 5.48 | 0.46 | 7.38 |
| 16 | 1.287 ± 0.119 | 5.97 | 0.37 | 8.04 |
| 24 | 1.042 ± 0.045 | 7.37 | 0.31 | 9.93 |
| 32 | 1.016 ± 0.031 | 7.56 | 0.24 | 10.19 |

### Dataset completo (87 000 imágenes)

| workers | tiempo_s | speedup_vs_1_proceso | eficiencia |
|---|---|---|---|
| 1 | 35.704 ± 2.029 | 1.0 | 1.0 |
| 4 | 11.177 ± 0.183 | 3.19 | 0.8 |
| 8 | 6.930 ± 0.235 | 5.15 | 0.64 |
| 16 | 4.687 ± 0.200 | 7.62 | 0.48 |
| 24 | 3.772 ± 0.106 | 9.47 | 0.39 |
| 32 | 4.048 ± 0.149 | 8.82 | 0.28 |

## Entrenamiento: CPU vs. GPU

Speedup respecto al entrenamiento original en CPU (TensorFlow usa los 32 hilos).

| configuracion | entrenamiento_s | por_epoca_s | speedup_vs_cpu | accuracy_prueba |
|---|---|---|---|---|
| CPU, 1 hilo (estrictamente secuencial) | 164.541 ± 1.922 | 32.91 | 0.25 | 0.9217 ± 0.0171 |
| CPU, 32 hilos (Lab 3 original) | 41.312 ± 1.697 | 8.26 | 1.0 | 0.9332 ± 0.0063 |
| GPU fp32, arreglos NumPy | 16.286 ± 1.913 | 3.26 | 2.54 | 0.9242 ± 0.0099 |
| GPU fp32 + tf.data | 14.941 ± 1.763 | 2.99 | 2.76 | 0.9356 ± 0.0057 |
| GPU precisión mixta + tf.data | 12.691 ± 1.883 | 2.54 | 3.26 | 0.9336 ± 0.0136 |
| GPU precisión mixta + tf.data + 32 pasos/llamada | 12.086 ± 1.521 | 2.42 | 3.42 | 0.9306 ± 0.0112 |

Costo único de inicialización de CUDA/cuDNN por proceso (calentamiento, no incluido arriba): GPU fp32: 7.7 s; GPU fp32 + tf.data: 9.5 s; GPU mixta + tf.data: 9.4 s; GPU final: 9.4 s.

### Dataset completo (60 900 imágenes de entrenamiento)

| configuracion | entrenamiento_s | por_epoca_s | speedup | accuracy_prueba |
|---|---|---|---|---|
| CPU (32 hilos) + tf.data | 193.280 ± 2.976 | 38.66 | 1.0 | 0.9819 |
| GPU final | 39.085 ± 3.414 | 7.82 | 4.95 | 0.9877 |

## Transferencias CPU ↔ GPU (conjunto de entrenamiento completo)

| dtype | megabytes | h2d_seconds | h2d_gb_per_s | d2h_seconds | d2h_gb_per_s |
|---|---|---|---|---|---|
| float32 | 570.938 | 0.096 | 5.841 | 1.018 | 0.562 |
| uint8 | 142.734 | 0.028 | 5.495 | 0.157 | 0.957 |

## Kernels propios en Numba CUDA (inferencia)

| implementation | seconds | h2d_s | kernels_s | d2h_s |
|---|---|---|---|---|
| keras_gpu | 0.2074 | nan | nan | nan |
| numba_cuda_naive | 0.5655 | 0.0182 | 0.5465 | 0.0005 |
| numba_cuda_tiled | 0.2888 | 0.0175 | 0.2707 | 0.0004 |
| numpy_cpu | 18.7443 | nan | nan | nan |

Diferencia máxima de probabilidades vs. Keras: 2.73e-03; coincidencia de clase predicha: 0.9996.
