import os
import random
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img


def get_files(source_dir):
    """Obtiene todas las imágenes .jpg"""
    return [f for f in os.listdir(source_dir) if f.endswith(".jpg")]


def get_top_characters(files, top_n=18):
    """Obtiene los personajes más frecuentes"""
    characters = [f.split("_pic_")[0] for f in files]
    counts = Counter(characters)
    return [c for c, _ in counts.most_common(top_n)]


def create_directories(base_dir, characters):
    """Crea estructura train/val"""
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    for c in characters:
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)

    return train_dir, val_dir


def group_images(files, characters):
    """Agrupa imágenes por personaje"""
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
    target_size=(150, 150),
    split_ratio=0.8
):
    """Divide en train/val y redimensiona"""

    for character, images in char_files.items():
        random.shuffle(images)
        split = int(len(images) * split_ratio)

        train_imgs = images[:split]
        val_imgs = images[split:]

        for subset, subset_dir in [(train_imgs, train_dir), (val_imgs, val_dir)]:
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
    # CONFIGURACIÓN
    source_dir = r"D:\Diplomado\TecnicasCognitivasIntroduccionABD\Prácticas\Práctica II\CNN_Simpsons\simpsons\simpsons"
    output_dir = r"D:\Diplomado\TecnicasCognitivasIntroduccionABD\Prácticas\Práctica II\output_check"
    target_size = (150, 150)

    # Pipeline
    files = get_files(source_dir)
    top_characters = get_top_characters(files)

    print("Top personajes:")
    print(top_characters)

    train_dir, val_dir = create_directories(output_dir, top_characters)
    char_files = group_images(files, top_characters)

    process_and_split(
        char_files,
        source_dir,
        train_dir,
        val_dir,
        target_size=target_size
    )

    print(f"Dataset creado y estandarizado a {target_size}")


if __name__ == "__main__":
    main()