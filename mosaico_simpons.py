import os
import random
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def obtener_imagenes_simpsons(
    ruta_base: str,
    n_por_clase: int = 5,
    seed: int = None
) -> List[Tuple[str, str]]:
    """
    Obtiene imágenes de cada clase (personaje).

    Args:
        ruta_base: Ruta a carpeta (train o val)
        n_por_clase: Número de imágenes por clase
        seed: Semilla para reproducibilidad

    Returns:
        Lista de tuplas (ruta_imagen, clase)
    """
    if seed is not None:
        random.seed(seed)

    lista = []

    for clase in os.listdir(ruta_base):
        ruta_clase = os.path.join(ruta_base, clase)

        if os.path.isdir(ruta_clase):
            imagenes = os.listdir(ruta_clase)

            if not imagenes:
                continue

            seleccionadas = random.sample(
                imagenes,
                min(n_por_clase, len(imagenes))
            )

            for img in seleccionadas:
                lista.append((os.path.join(ruta_clase, img), clase))

    return lista


def crear_mosaico_simpsons(
    lista_imagenes: List[Tuple[str, str]],
    columnas: int = 5,
    ancho_celda: int = 150,
    alto_celda: int = 150
) -> Image.Image:
    """
    Crea un mosaico con imágenes y su clase.

    Returns:
        Imagen PIL
    """
    filas = (len(lista_imagenes) + columnas - 1) // columnas
    ancho_mosaico = columnas * ancho_celda
    alto_mosaico = filas * alto_celda

    mosaico = Image.new('RGB', (ancho_mosaico, alto_mosaico), color='white')
    draw = ImageDraw.Draw(mosaico)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for i, (img_path, clase) in enumerate(lista_imagenes):
        fila = i // columnas
        col = i % columnas

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img.thumbnail((ancho_celda, alto_celda - 20))

                x_offset = col * ancho_celda + (ancho_celda - img.width) // 2
                y_offset = fila * alto_celda + (alto_celda - img.height - 20) // 2

                mosaico.paste(img, (x_offset, y_offset))

                titulo = clase
                ancho_texto = draw.textlength(titulo, font=font)

                pos_x = col * ancho_celda + (ancho_celda - ancho_texto) // 2
                pos_y = fila * alto_celda + alto_celda - 20

                draw.text((pos_x, pos_y), titulo, font=font, fill="black")

        except Exception as e:
            print(f"Error con {img_path}: {e}")

    return mosaico