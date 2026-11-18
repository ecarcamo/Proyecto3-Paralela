import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, ensure_dirs

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
HEADER = "#f3f2ef"
ACCENT = ("#e8f1fb", "#fdeee8", "#e8f6ef", "#fff4e5")


def phase_mean(frame, phase):
    values = frame.loc[frame["phase"] == phase, "seconds"]
    return values.mean()


def main():
    ensure_dirs()
    sequential = pd.read_csv(RESULTS_DIR / "sequential_runs.csv")
    parallel = pd.read_csv(RESULTS_DIR / "cpu_parallel_subset.csv")
    gpu = pd.read_csv(RESULTS_DIR / "training_gpu_mixed_tfdata_spe_subset.csv")

    prep_seq = phase_mean(sequential, "preparacion_datos")
    prep_par = parallel.loc[parallel["workers"] == 24, "seconds"].mean()
    train_cpu = phase_mean(sequential, "entrenamiento")
    train_gpu = phase_mean(gpu, "entrenamiento")

    rows = [
        (
            "1",
            "Carga secuencial (Lab 3)",
            "Un hilo lee y redimensiona JPG",
            f"{prep_seq:.2f} s",
            "solo preparación",
            "evidence/01_secuencial_carga_un_nucleo.png",
        ),
        (
            "2",
            "Carga paralela (24 procesos)",
            "Memoria compartida + multiprocessing",
            f"{prep_par:.2f} s",
            "benchmark de carga",
            "evidence/02_paralela_carga_24_procesos.png",
        ),
        (
            "3",
            "Entrenamiento en CPU",
            "CNN en TF, hasta 32 hilos",
            f"{train_cpu:.2f} s",
            "5 épocas",
            "evidence/03_secuencial_entrenamiento_cpu.png",
        ),
        (
            "4",
            "Entrenamiento en GPU (final)",
            "CUDA + mixed precision + tf.data + SPE",
            f"{train_gpu:.2f} s",
            "5 épocas",
            "evidence/04_optimizada_entrenamiento_gpu.png",
        ),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    ax.set_title(
        "Cuatro escenarios del proyecto · subset 17 400 imágenes · tiempos promedio",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    headers = ["#", "Escenario", "Qué corre", "Tiempo", "Nota", "Captura"]
    table = ax.table(
        cellText=[[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows],
        colLabels=headers,
        loc="upper center",
        cellLoc="left",
        colWidths=[0.04, 0.22, 0.28, 0.1, 0.14, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=TEXT)
        if row == 0:
            cell.set_facecolor(HEADER)
            cell.set_text_props(fontweight="bold", color=TEXT)
        elif col == 0:
            cell.set_facecolor(ACCENT[row - 1])
            cell.set_text_props(fontweight="bold")
        elif 1 <= row <= 4:
            cell.set_facecolor(ACCENT[row - 1] if col == 1 else SURFACE)

    fig.text(
        0.02,
        0.02,
        "Pipeline completo secuencial vs optimizado: ver figures/01_tiempo_por_fase.png  "
        "(total 52.4 s → 14.0 s, ×3.74).",
        color=MUTED,
        fontsize=8.5,
    )

    out = FIGURES_DIR / "00_cuatro_escenarios.png"
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()