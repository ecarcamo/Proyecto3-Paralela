import os
import random
import shutil

from src.config import DATA_DIR, IMAGES_PER_CLASS, SEED, full_dataset_dir


def main():
    train_dir = full_dataset_dir()
    print(f"Dataset completo en: {train_dir}")

    dest_dir = DATA_DIR / "subset"
    os.makedirs(dest_dir, exist_ok=True)

    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    print(f"{len(classes)} clases encontradas")
    random.seed(SEED)

    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        dest_class_path = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        images = sorted(img for img in os.listdir(class_path) if img.endswith(".jpg") or img.endswith(".png"))
        random.shuffle(images)
        sampled = images[:IMAGES_PER_CLASS]

        for img in sampled:
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_class_path, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"{class_name}: {len(sampled)} imágenes")

    print("Submuestra lista")


if __name__ == "__main__":
    main()
