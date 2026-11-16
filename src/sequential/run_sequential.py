import argparse

import numpy as np

from src.config import BATCH_SIZE, DATA_DIR, LEARNING_RATE, MODELS_DIR, SEED, ensure_dirs
from src.gpu_runtime import load_tensorflow
from src.sequential.dataset import list_images, load_splits, split_dataset
from src.sequential.models import build_cnn_base, compile_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    tf = load_tensorflow("cpu")
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    ensure_dirs()

    classes, files, labels = list_images(DATA_DIR / "subset")
    splits = split_dataset(files, labels)
    data = load_splits(splits, classes)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    print("Entrenamiento:", X_train.shape, "Validación:", X_val.shape, "Prueba:", X_test.shape)

    model = compile_model(build_cnn_base(), LEARNING_RATE)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
              epochs=args.epochs, callbacks=[early_stop], verbose=2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN base en CPU, accuracy en prueba: {test_acc:.4f}, loss: {test_loss:.4f}")
    model.save(MODELS_DIR / "cnn_base_cpu.keras")


if __name__ == "__main__":
    main()
