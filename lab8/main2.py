import typing as t
import sys
from pathlib import Path
from collections import Counter
import json
import numpy as np
import cv2

def prepare_classes(caltech_dir: Path) -> dict[str, list[Path]]:
    result = {}
    # ищем все подкаталоги caltech, не начинающиеся с точки
    all_items = caltech_dir.glob("*")
    for item in all_items:
        if item.is_dir() and not item.name.startswith("."):
            result[item.name] = []

    # заполняем списки файлов в каждой категории
    for name, files in result.items():
        category = caltech_dir / name
        files.extend(category.glob("*.png"))
        files.extend(category.glob("*.jpg"))
        files.extend(category.glob("*.jpeg"))

    # удаляем категории без единого файла
    for name, files in list(result.items()): # для избежания параллельного изменения и итерирования
        if not files:
            del result[name]

    return result

def prepare_image(img: Path) -> np.ndarray:
    data = np.fromfile(img, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
    return gray

def get_visual_words(classes: dict[str, list[Path]], N_words: int) -> tuple[np.ndarray, list]:
    detector = cv2.SIFT_create()
    all_descriptors: list[np.ndarray] = []
    image_descriptors: list[tuple[
        str, # имя класса
        str, # имя файла
        int, # начальный индекс в ll_descriptors
        int # конечный
        ]] = []

    for name, images in classes.items():
        index = 0
        total = len(images)
        print(f"{name:20}: ", end="")
        print(f"{index:5d}/{total:5d}", end="", flush=True)
        # перебираем изображения в классе
        for imfile in images:
            try:
                img = prepare_image(imfile)
            except IOError():
                index += 1
            else:
                # файл картинки загрузился
                pts, descriptors = detector.detectAndCompute(img, None)
                before = len(all_descriptors)
                all_descriptors.extend(descriptors)
                after = len(all_descriptors)
                image_descriptors.append(
                    (
                        name,
                        str(imfile),
                        before,
                        after
                     )
                )
                index += 1
            print("\x08" * 11, end="")
            print(f"{index:5d}/{total:5d}", end="", flush=True)
        print()

if __name__ == '__main__':
    SCRIPT = Path(sys.argv[0]) # путь к скрипту
    SCRIPT_DIR = SCRIPT.parent.resolve() # абсолютный путь к каталогу скрипта
    CALTECH_DIR = SCRIPT_DIR / "caltech101" # путь к каталогу CALTECH
    classes = prepare_classes(CALTECH_DIR)

    print(list(classes.keys()))