import argparse
import time

import numpy as np

from src.config import BATCH_SIZE, DATA_DIR, LEARNING_RATE, MODELS_DIR, SEED, ensure_dirs
from src.gpu_runtime import load_tensorflow
from src.sequential.dataset import list_images, load_splits, split_dataset
from src.sequential.models import build_cnn_base, compile_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    tf = load_tensorflow("cpu")
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    ensure_dirs()

    start = time.perf_counter()
    classes, files, labels = list_images(DATA_DIR / "subset")
    splits = split_dataset(files, labels)
    data = load_splits(splits, classes)
    preparation = time.perf_counter() - start
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    print("Entrenamiento:", X_train.shape, "Validación:", X_val.shape, "Prueba:", X_test.shape)

    model = compile_model(build_cnn_base(), LEARNING_RATE)
    callbacks = [] if args.no_early_stop else [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    ]
    start = time.perf_counter()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
              epochs=args.epochs, callbacks=callbacks, verbose=2)
    training = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = model.predict(X_test, batch_size=256, verbose=0)
    inference = time.perf_counter() - start
    accuracy = float((probabilities.argmax(axis=1) == y_test).mean())

    print(f"CNN base en CPU, accuracy en prueba: {accuracy:.4f}")
    print(f"Tiempos (s): preparación {preparation:.2f} · entrenamiento {training:.2f} · inferencia {inference:.2f}")
    model.save(MODELS_DIR / "cnn_base_cpu.keras")


if __name__ == "__main__":
    main()
