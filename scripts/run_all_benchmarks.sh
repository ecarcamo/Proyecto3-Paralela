#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FILTER='oneDNN|absl|^WARNING|cuda_dnn|computation placer|cuda_blas|cuda_fft|Unable to register|cuInit'

run() {
    local log=$1
    shift
    echo "== $(date +%T) $*"
    "$PY" -u "$@" 2>&1 | grep -vE "$FILTER" > "logs/$log.log"
}

rm -f results/sequential_runs*.csv results/cpu_parallel_*.csv results/training_*.csv results/transfers_*.csv \
      results/kernels_*.csv results/kernels_validation.json

run env_check -m src.env_check
run bench_sequential -m benchmarks.bench_sequential --repeats 5 --epochs 5
run bench_sequential_1thread -m benchmarks.bench_sequential --repeats 3 --epochs 5 --threads 1
run bench_cpu_parallel_subset -m benchmarks.bench_cpu_parallel --repeats 5
run bench_cpu_parallel_subset_spawn -m benchmarks.bench_cpu_parallel --start-method spawn --repeats 3
run bench_cpu_parallel_full -m benchmarks.bench_cpu_parallel --full --workers 1,4,8,16,24,32 --repeats 3
run bench_training_gpu_fp32_subset -m benchmarks.bench_training --variant gpu_fp32 --save-model
run bench_training_gpu_fp32_tfdata_subset -m benchmarks.bench_training --variant gpu_fp32_tfdata
run bench_training_gpu_mixed_tfdata_subset -m benchmarks.bench_training --variant gpu_mixed_tfdata
run bench_training_gpu_mixed_tfdata_spe_subset -m benchmarks.bench_training --variant gpu_mixed_tfdata_spe
run bench_training_gpu_mixed_tfdata_spe_full -m benchmarks.bench_training --variant gpu_mixed_tfdata_spe --dataset full --repeats 3
run bench_training_cpu_tfdata_full -m benchmarks.bench_training --variant cpu_tfdata --dataset full --repeats 2
run bench_kernels -m benchmarks.bench_kernels
echo "== $(date +%T) listo"
