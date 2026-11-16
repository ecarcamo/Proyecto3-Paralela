from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

KAGGLE_DATASET = "grassknoted/asl-alphabet"
IMAGES_PER_CLASS = 600
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 29
SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def full_dataset_dir() -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    nested = root / "asl_alphabet_train" / "asl_alphabet_train"
    return nested if nested.exists() else root / "asl_alphabet_train"


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, LOGS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
