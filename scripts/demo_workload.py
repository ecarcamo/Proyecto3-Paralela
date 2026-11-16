import argparse
import time

from src.config import BATCH_SIZE, DATA_DIR, LEARNING_RATE
from src.parallel_cpu.dataset import ParallelImageLoader
from src.sequential.dataset import list_images, load_splits, split_dataset

PHASES = ("sequential_load", "parallel_load", "cpu_train", "gpu_train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--seconds", type=float, default=40)
    args = parser.parse_args()

    classes, files, labels = list_images(DATA_DIR / "subset")
    splits = split_dataset(files, labels)
    loader = ParallelImageLoader()
    deadline = time.perf_counter() + args.seconds

    if args.phase in ("sequential_load", "parallel_load"):
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            if args.phase == "sequential_load":
                load_splits(splits, classes)
            else:
                loader.load_splits(splits, classes)
            print(f"{args.phase}: {time.perf_counter() - start:.2f} s", flush=True)
        loader.close()
        return

    data = loader.load_splits(splits, classes, as_float=args.phase == "cpu_train")
    loader.close()
    from src.gpu_runtime import load_tensorflow

    if args.phase == "cpu_train":
        tf = load_tensorflow("cpu")
        from src.sequential.models import build_cnn_base, compile_model

        model = compile_model(build_cnn_base(), LEARNING_RATE)
        X_train, y_train = data["train"]
        while time.perf_counter() < deadline:
            model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=2)
        return

    tf = load_tensorflow("gpu")
    from src.gpu.train import VARIANTS, build_model
    from src.parallel_cpu.pipeline import make_dataset

    variant = VARIANTS["gpu_mixed_tfdata_spe"]
    model = build_model(tf, variant["mixed"], 0, variant["steps_per_execution"])
    train = make_dataset(tf, *data["train"], BATCH_SIZE, shuffle=True)
    while time.perf_counter() < deadline:
        model.fit(train, epochs=1, verbose=2)


if __name__ == "__main__":
    main()
