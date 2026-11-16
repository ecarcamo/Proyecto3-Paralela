import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR, ensure_dirs

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

PHASES = ["preparacion_datos", "entrenamiento", "inferencia"]
FINAL_VARIANT = "gpu_mixed_tfdata_spe"
PHASE_LABELS = {
    "preparacion_datos": "Preparación de datos (CPU)",
    "entrenamiento": "Entrenamiento (5 épocas)",
    "inferencia": "Inferencia (2 610 imágenes)",
    "total": "Total del pipeline",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def stats(frame, phase):
    values = frame.loc[frame["phase"] == phase, "seconds"]
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def totals(frame, phases):
    subset = frame[frame["phase"].isin(phases)]
    per_run = subset.groupby("run")["seconds"].sum()
    return per_run.mean(), per_run.std(ddof=1)


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def phase_table(sequential, optimized):
    rows = []
    for phase in PHASES + ["total"]:
        if phase == "total":
            seq_mean, seq_std = totals(sequential, PHASES)
            opt_mean, opt_std = totals(optimized, PHASES + ["transferencia_h2d"])
        else:
            seq_mean, seq_std = stats(sequential, phase)
            opt_mean, opt_std = stats(optimized, phase)
        rows.append({
            "fase": PHASE_LABELS[phase],
            "secuencial_s": fmt(seq_mean, seq_std),
            "optimizada_s": fmt(opt_mean, opt_std),
            "speedup": round(seq_mean / opt_mean, 2),
            "seq_mean": seq_mean,
            "opt_mean": opt_mean,
        })
    return pd.DataFrame(rows)


def plot_phases(table):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(table))
    height = 0.36
    ax.barh(y - height / 2 - 0.01, table["seq_mean"], height, color=BLUE, label="Secuencial (Lab 3 original, CPU)")
    ax.barh(y + height / 2 + 0.01, table["opt_mean"], height, color=ORANGE, label="Optimizada (CPU paralela + GPU CUDA)")
    for i, row in table.iterrows():
        ax.text(row["seq_mean"], i - height / 2, f" {row['seq_mean']:.2f} s", va="center", color=TEXT, fontsize=9)
        ax.text(row["opt_mean"], i + height / 2, f" {row['opt_mean']:.2f} s  (×{row['speedup']})",
                va="center", color=TEXT, fontsize=9)
    ax.set_yticks(y, table["fase"])
    ax.invert_yaxis()
    ax.set_xlabel("Tiempo promedio por corrida (s)")
    ax.set_title("Tiempo por fase: secuencial vs. optimizada")
    ax.set_xlim(0, table["seq_mean"].max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    save(fig, "01_tiempo_por_fase.png")


def scaling_table(parallel, sequential_prep):
    grouped = parallel.groupby("workers")["seconds"].agg(["mean", "std"]).reset_index()
    base = grouped.loc[grouped["workers"] == 1, "mean"].iloc[0]
    grouped["speedup_vs_1_proceso"] = base / grouped["mean"]
    grouped["eficiencia"] = grouped["speedup_vs_1_proceso"] / grouped["workers"]
    grouped["speedup_vs_secuencial"] = sequential_prep / grouped["mean"]
    return grouped


def plot_scaling(grouped, name, title):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    left.plot(grouped["workers"], grouped["mean"], color=BLUE, linewidth=2, marker="o", markersize=7)
    left.set_xlabel("Procesos")
    left.set_ylabel("Tiempo (s)")
    left.set_title("Tiempo de preparación")
    right.plot(grouped["workers"], grouped["workers"], color=MUTED, linewidth=1.5, linestyle="--", label="Ideal (lineal)")
    right.plot(grouped["workers"], grouped["speedup_vs_1_proceso"], color=ORANGE, linewidth=2, marker="o",
               markersize=7, label="Medido")
    right.axvline(24, color=GRID, linewidth=1.5)
    right.text(24.3, 1, "24 núcleos físicos", color=MUTED, fontsize=8)
    last = grouped.iloc[-1]
    right.annotate(f"×{last['speedup_vs_1_proceso']:.1f}", (last["workers"], last["speedup_vs_1_proceso"]),
                   textcoords="offset points", xytext=(-28, 6), color=TEXT)
    right.set_xlabel("Procesos")
    right.set_ylabel("Speedup vs. 1 proceso")
    right.set_title("Escalabilidad")
    right.legend(loc="upper left")
    for ax in (left, right):
        ax.set_xticks(grouped["workers"])
    fig.suptitle(title, color=TEXT, fontweight="bold")
    save(fig, name)


def training_table(frames, cpu_reference):
    rows = []
    for label, frame in frames:
        if frame is None:
            continue
        mean, std = stats(frame, "entrenamiento")
        accuracy = frame.groupby("run")["test_accuracy"].first()
        rows.append({
            "configuracion": label,
            "entrenamiento_s": fmt(mean, std),
            "por_epoca_s": round(mean / frame["epochs"].iloc[0], 2),
            "speedup_vs_cpu": round(cpu_reference / mean, 2),
            "accuracy_prueba": f"{accuracy.mean():.4f} ± {accuracy.std(ddof=1):.4f}",
            "mean": mean,
        })
    return pd.DataFrame(rows)


def plot_training(table):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    colors = [BLUE, BLUE, ORANGE, ORANGE, ORANGE, ORANGE][: len(table)]
    y = np.arange(len(table))
    ax.barh(y, table["mean"], 0.6, color=colors)
    for i, row in table.iterrows():
        ax.text(row["mean"], i, f" {row['mean']:.1f} s  (×{row['speedup_vs_cpu']})", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(y, table["configuracion"])
    ax.invert_yaxis()
    ax.set_xlim(0, table["mean"].max() * 1.3)
    ax.set_xlabel("Tiempo de entrenamiento, 5 épocas (s)")
    ax.set_title("Entrenamiento de la CNN base: CPU vs. GPU")
    ax.grid(axis="y", visible=False)
    save(fig, "03_entrenamiento_cpu_vs_gpu.png")


def plot_core_heatmaps():
    sources = [
        ("cpu_sequential.csv", "preparacion_datos", "Carga secuencial (Lab 3 original)"),
        ("cpu_parallel_subset.csv", "preparacion_datos_24p", "Carga paralela, 24 procesos"),
        (f"cpu_training_{FINAL_VARIANT}_subset.csv", "entrenamiento", "Entrenamiento en GPU (tf.data en CPU)"),
    ]
    panels = []
    for file_name, phase, title in sources:
        path = LOGS_DIR / file_name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["phase"] == phase]
        if frame.empty:
            continue
        cores = frame[[c for c in frame.columns if c.startswith("core_")]].to_numpy().T
        panels.append((title, cores[:, : min(cores.shape[1], 120)]))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (title, cores) in zip(axes, panels):
        image = ax.imshow(cores, aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
        ax.set_title(f"{title} · uso promedio {cores.mean():.0f} %", loc="left")
        ax.set_ylabel("Núcleo lógico")
        ax.set_xlabel("Muestra (cada 0.25 s)")
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="% uso", pad=0.01)
    save(fig, "04_uso_nucleos_cpu.png")


def plot_gpu_usage():
    path = LOGS_DIR / f"gpu_training_{FINAL_VARIANT}_subset.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame[frame["t"] <= frame.loc[frame["phase"] != "idle", "t"].max() + 1]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    top.plot(frame["t"], frame["utilization_gpu"], color=ORANGE, linewidth=1.5)
    top.set_ylabel("Uso de núcleos GPU (%)")
    top.set_ylim(0, 105)
    bottom.plot(frame["t"], frame["memory_used"], color=BLUE, linewidth=1.5)
    bottom.set_ylabel("Memoria GPU (MiB)")
    bottom.set_xlabel("Tiempo (s)")
    for ax in (top, bottom):
        for _, span in frame[frame["phase"] == "entrenamiento"].groupby((frame["phase"] != "entrenamiento").cumsum()):
            ax.axvspan(span["t"].min(), span["t"].max(), color=GRID, alpha=0.6, linewidth=0)
    top.set_title("GPU durante las corridas de entrenamiento (sombreado: fase de entrenamiento)")
    save(fig, "05_uso_gpu_entrenamiento.png")


def plot_kernels(runs, profile):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    labels = {"numpy_cpu": "NumPy\n(CPU)", "numba_cuda_naive": "Numba\ningenuo",
              "numba_cuda_tiled": "Numba\nmem. compartida", "keras_gpu": "Keras/cuDNN\n(GPU)"}
    means = runs.groupby("implementation")["seconds"].mean().reindex(list(labels))
    colors = [BLUE, YELLOW, ORANGE, AQUA]
    left.bar([labels[k] for k in means.index], means.values, color=colors, width=0.6)
    for i, value in enumerate(means.values):
        left.text(i, value, f"{value:.3f} s", ha="center", va="bottom", color=TEXT, fontsize=9)
    left.set_yscale("log")
    left.set_ylabel("Tiempo de inferencia (s, escala log)")
    left.set_title("Inferencia de 2 610 imágenes")
    left.grid(axis="x", visible=False)
    pivot = profile.pivot_table(index="kernel", columns="variant", values="ms", sort=False)
    y = np.arange(len(pivot))
    right.barh(y - 0.2, pivot["numba_cuda_naive"], 0.38, color=YELLOW, label="Ingenuo")
    right.barh(y + 0.2, pivot["numba_cuda_tiled"], 0.38, color=ORANGE, label="Memoria compartida")
    right.set_yticks(y, pivot.index)
    right.invert_yaxis()
    right.legend(loc="lower right")
    right.set_xlabel("ms por batch de 256")
    right.set_title("Tiempo por kernel")
    right.grid(axis="y", visible=False)
    save(fig, "06_kernels_numba.png")


def markdown(frame, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[c]) for c in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, divider, *body])


def main():
    ensure_dirs()
    sequential = read("sequential_runs.csv")
    sequential_1thread = read("sequential_runs_1thread.csv")
    parallel = read("cpu_parallel_subset.csv")
    parallel_full = read("cpu_parallel_full.csv")
    gpu_fp32 = read("training_gpu_fp32_subset.csv")
    gpu_fp32_tfdata = read("training_gpu_fp32_tfdata_subset.csv")
    gpu_mixed = read("training_gpu_mixed_tfdata_subset.csv")
    gpu_final = read(f"training_{FINAL_VARIANT}_subset.csv")
    full_cpu = read("training_cpu_tfdata_full.csv")
    full_gpu = read(f"training_{FINAL_VARIANT}_full.csv")
    transfers = [read(f"transfers_{name}_subset.csv") for name in ("gpu_fp32", FINAL_VARIANT)]
    kernel_runs = read("kernels_runs.csv")
    kernel_profile = read("kernels_profile.csv")

    sections = ["# Resumen de resultados\n", "Generado por `python -m analysis.compare`. Promedio ± desviación estándar.\n"]

    phases = phase_table(sequential, gpu_final)
    plot_phases(phases)
    sections += ["## Tiempo por fase y speedup\n",
                 "Secuencial: código original del Lab 3 en CPU. Optimizada: carga con 24 procesos + entrenamiento "
                 "e inferencia en GPU con precisión mixta, `tf.data` y 32 pasos por llamada. El total optimizado "
                 "suma además la transferencia explícita CPU→GPU del conjunto de entrenamiento.\n",
                 markdown(phases, ["fase", "secuencial_s", "optimizada_s", "speedup"]), ""]

    seq_prep = stats(sequential, "preparacion_datos")[0]
    scaling = scaling_table(parallel, seq_prep)
    plot_scaling(scaling, "02_escalabilidad_carga_cpu.png", "Carga de 17 400 imágenes con multiprocessing")
    scaling_view = scaling.assign(
        tiempo_s=[fmt(m, s) for m, s in zip(scaling["mean"], scaling["std"])],
        speedup_vs_1_proceso=scaling["speedup_vs_1_proceso"].round(2),
        eficiencia=scaling["eficiencia"].round(2),
        speedup_vs_secuencial=scaling["speedup_vs_secuencial"].round(2),
    )
    sections += ["## Escalabilidad de la preparación de datos en CPU\n",
                 markdown(scaling_view, ["workers", "tiempo_s", "speedup_vs_1_proceso", "eficiencia",
                                         "speedup_vs_secuencial"]), ""]
    if parallel_full is not None:
        full_scaling = scaling_table(parallel_full, np.nan)
        plot_scaling(full_scaling, "02b_escalabilidad_carga_cpu_completo.png",
                     "Carga del dataset completo (87 000 imágenes)")
        full_view = full_scaling.assign(
            tiempo_s=[fmt(m, s) for m, s in zip(full_scaling["mean"], full_scaling["std"])],
            speedup_vs_1_proceso=full_scaling["speedup_vs_1_proceso"].round(2),
            eficiencia=full_scaling["eficiencia"].round(2),
        )
        sections += ["### Dataset completo (87 000 imágenes)\n",
                     markdown(full_view, ["workers", "tiempo_s", "speedup_vs_1_proceso", "eficiencia"]), ""]

    cpu_reference = stats(sequential, "entrenamiento")[0]
    training = training_table([
        ("CPU, 1 hilo (estrictamente secuencial)", sequential_1thread),
        ("CPU, 32 hilos (Lab 3 original)", sequential),
        ("GPU fp32, arreglos NumPy", gpu_fp32),
        ("GPU fp32 + tf.data", gpu_fp32_tfdata),
        ("GPU precisión mixta + tf.data", gpu_mixed),
        ("GPU precisión mixta + tf.data + 32 pasos/llamada", gpu_final),
    ], cpu_reference)
    plot_training(training)
    warmups = [(label, frame["warmup_seconds"].iloc[0]) for label, frame in
               (("GPU fp32", gpu_fp32), ("GPU fp32 + tf.data", gpu_fp32_tfdata), ("GPU mixta + tf.data", gpu_mixed),
                ("GPU final", gpu_final)) if frame is not None and "warmup_seconds" in frame]
    sections += ["## Entrenamiento: CPU vs. GPU\n",
                 "Speedup respecto al entrenamiento original en CPU (TensorFlow usa los 32 hilos).\n",
                 markdown(training, ["configuracion", "entrenamiento_s", "por_epoca_s", "speedup_vs_cpu",
                                     "accuracy_prueba"]), "",
                 "Costo único de inicialización de CUDA/cuDNN por proceso (calentamiento, no incluido arriba): "
                 + "; ".join(f"{label}: {value:.1f} s" for label, value in warmups) + ".", ""]

    if full_cpu is not None and full_gpu is not None:
        full_rows = []
        for label, frame in (("CPU (32 hilos) + tf.data", full_cpu), ("GPU final", full_gpu)):
            mean, std = stats(frame, "entrenamiento")
            full_rows.append({"configuracion": label, "entrenamiento_s": fmt(mean, std),
                              "por_epoca_s": round(mean / frame["epochs"].iloc[0], 2), "mean": mean,
                              "accuracy_prueba": f"{frame.groupby('run')['test_accuracy'].first().mean():.4f}"})
        full_table = pd.DataFrame(full_rows)
        full_table["speedup"] = (full_table["mean"].iloc[0] / full_table["mean"]).round(2)
        sections += ["### Dataset completo (60 900 imágenes de entrenamiento)\n",
                     markdown(full_table, ["configuracion", "entrenamiento_s", "por_epoca_s", "speedup",
                                           "accuracy_prueba"]), ""]

    transfer_rows = pd.concat([t for t in transfers if t is not None])
    transfer_view = transfer_rows.groupby("dtype")[["megabytes", "h2d_seconds", "h2d_gb_per_s", "d2h_seconds",
                                                    "d2h_gb_per_s"]].mean().round(3).reset_index()
    sections += ["## Transferencias CPU ↔ GPU (conjunto de entrenamiento completo)\n",
                 markdown(transfer_view, list(transfer_view.columns)), ""]

    if kernel_runs is not None:
        plot_kernels(kernel_runs, kernel_profile)
        validation = json.loads((RESULTS_DIR / "kernels_validation.json").read_text())
        kernel_summary = kernel_runs.groupby("implementation")[["seconds", "h2d_s", "kernels_s", "d2h_s"]].mean()
        kernel_summary = kernel_summary.round(4).reset_index()
        sections += ["## Kernels propios en Numba CUDA (inferencia)\n",
                     markdown(kernel_summary, list(kernel_summary.columns)), "",
                     f"Diferencia máxima de probabilidades vs. Keras: "
                     f"{validation['max_abs_diff_numba_vs_keras']:.2e}; coincidencia de clase predicha: "
                     f"{validation['argmax_agreement_numba_vs_keras']:.4f}.", ""]

    plot_core_heatmaps()
    plot_gpu_usage()
    phases.drop(columns=["seq_mean", "opt_mean"]).to_csv(RESULTS_DIR / "speedup_por_fase.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))


if __name__ == "__main__":
    main()
