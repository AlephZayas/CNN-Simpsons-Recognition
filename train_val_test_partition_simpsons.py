import os
import random
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img


def get_files(source_dir):
    return [f for f in os.listdir(source_dir) if f.endswith(".jpg")]


def get_top_characters(files, top_n=18):
    characters = [f.split("_pic_")[0] for f in files]
    counts = Counter(characters)
    return [c for c, _ in counts.most_common(top_n)]


def create_directories(base_dir, characters):
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for c in characters:
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

    return train_dir, val_dir, test_dir


def group_images(files, characters):
    char_files = {c: [] for c in characters}

    for f in files:
        character = f.split("_pic_")[0]
        if character in characters:
            char_files[character].append(f)

    return char_files


def process_and_split(
    char_files,
    source_dir,
    train_dir,
    val_dir,
    test_dir,
    target_size=(150, 150),
    train_ratio=0.7,
    val_ratio=0.15
):
    """
    Divide en train/val/test y redimensiona
    test_ratio se infiere como: 1 - train - val
    """

    for character, images in char_files.items():
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        splits = [
            (train_imgs, train_dir),
            (val_imgs, val_dir),
            (test_imgs, test_dir)
        ]

        for subset, subset_dir in splits:
            for img_name in subset:
                try:
                    img = load_img(
                        os.path.join(source_dir, img_name),
                        target_size=target_size
                    )

                    img.save(
                        os.path.join(subset_dir, character, img_name)
                    )

                except Exception as e:
                    print(f"Error con {img_name}: {e}")


def main():
    source_dir = r"D:\Diplomado\TecnicasCognitivasIntroduccionABD\Prácticas\Práctica II\CNN_Simpsons\simpsons\simpsons"
    output_dir = r"D:\Diplomado\TecnicasCognitivasIntroduccionABD\Prácticas\Práctica II\CNN_Simpsons\simpsons_train_val_test"
    target_size = (150, 150)

    files = get_files(source_dir)
    top_characters = get_top_characters(files)

    print("Top personajes:")
    print(top_characters)

    train_dir, val_dir, test_dir = create_directories(output_dir, top_characters)
    char_files = group_images(files, top_characters)

    process_and_split(
        char_files,
        source_dir,
        train_dir,
        val_dir,
        test_dir,
        target_size=target_size,
        train_ratio=0.7,
        val_ratio=0.15
    )

    print(f"Dataset creado (train/val/test) y estandarizado a {target_size}")


if __name__ == "__main__":
    main()