import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import IMAGE_SIZE, SEED


def list_images(base_path):
    classes = sorted(os.listdir(base_path))
    files, labels = [], []
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        for image_name in sorted(os.listdir(class_path)):
            files.append(os.path.join(class_path, image_name))
            labels.append(class_name)
    return classes, files, labels


def split_dataset(files, labels):
    files_train, files_temp, labels_train, labels_temp = train_test_split(
        files, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    files_val, files_test, labels_val, labels_test = train_test_split(
        files_temp, labels_temp, test_size=0.50, stratify=labels_temp, random_state=SEED
    )
    return {
        "train": (files_train, labels_train),
        "val": (files_val, labels_val),
        "test": (files_test, labels_test),
    }


def load_set(paths, text_labels, class_to_index):
    X = np.zeros((len(paths), *IMAGE_SIZE, 3), dtype=np.float32)
    y = np.zeros(len(paths), dtype=np.int32)
    for i, (path, label) in enumerate(zip(paths, text_labels)):
        img = Image.open(path).resize(IMAGE_SIZE)
        X[i] = np.array(img) / 255.0
        y[i] = class_to_index[label]
    return X, y


def load_splits(splits, classes):
    class_to_index = {name: i for i, name in enumerate(classes)}
    return {name: load_set(paths, labels, class_to_index) for name, (paths, labels) in splits.items()}
