import time
import numpy
import cv2
import sys
import typing as t
from pathlib import Path
from collections import Counter
import json

def prepare_classes(caltech_dir: Path) -> dict[str, list[Path]]:
    result = {}
    # ищем все подкаталоги caltech, не начинающиеся с точки
    all_items = caltech_dir.glob('*')
    for item in all_items:
        if item.is_dir() and not item.name.startswith('.'):
            result[item.name] = []
    # заполняем списки файлов в каждой категории
    for name, files in result.items():
        category = caltech_dir / name
        files.extend(category.glob('*.png'))
        files.extend(category.glob('*.jpg'))
        files.extend(category.glob('*.jpeg'))
    # удаляем категории без единого файла
    for name, files in list(result.items()):
        if not files:
            del result[name]
    return result


def prepare_image(img: Path) -> numpy.ndarray:
    data = numpy.fromfile(img, numpy.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
    return gray


def get_visual_words(classes: dict[str, list[Path]], N_words: int) -> tuple[numpy.ndarray, list]:
    detector = cv2.SIFT_create()
    all_descriptors: list[numpy.ndarray] = []
    image_descriptors: list[tuple[
        str,  # имя класса
        str,  # имя файла
        int,  # начальный индекс в all_descriptors
        int  # конечный индекс в all_descriptors
    ]] = []
    # перебираем классы
    for name, images in classes.items():
        index = 0
        total = len(images)
        print(f'{name:20}: ', end='')
        print(f'{index:5d}/{total:5d}', end='', flush=True)
        # перебираем изображения в классе
        for imfile in images:
            try:
                img = prepare_image(imfile)
            except IOError:
                index += 1
            else:
                # файл картинки загрузился
                pts, descriptors = detector.detectAndCompute(img, None)
                before = len(all_descriptors)
                all_descriptors.extend(descriptors)
                after = len(all_descriptors)
                image_descriptors.append(
                    (
                        name,  # имя класса
                        str(imfile),  # путь к файлу
                        before,
                        after
                    )
                )
                index += 1
            print('\x08' * 11, end='')
            print(f'{index:5d}/{total:5d}', end='', flush=True)
        print()
    # готовим словарь визуальных слов
    descriptor_array = numpy.array(all_descriptors, numpy.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                15,  # макс 15 итераций
                0.1  # погрешность порядка 0.1
                )
    flags = cv2.KMEANS_RANDOM_CENTERS  # случайные начальные позиции
    attempts = 5  # число попыток кластеризации
    print('Clusterizing...')
    start_time = time.time()
    _compact, labels, centers = cv2.kmeans(
        data=descriptor_array,  # данные для кластеризации
        K=N_words,  # размер словаря
        bestLabels=None,  # нет допущений о принадлежности слов
        criteria=criteria,
        attempts=attempts,
        flags=flags,  # случайные начальные позиции центров
        centers=None  # у нас нет допущений о начальных позициях
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for k-means clustering: {elapsed_time} seconds")
    # составляем гистограммы слов для изображений
    word_histograms: t.List[t.Dict] = []
    for catname, imfile, before, after in image_descriptors:
        # номера визуальных слов в изображении imfile
        words = labels[before:after]
        cnt = Counter(words[:, 0])
        histogram = [cnt[x] for x in range(N_words)]
        word_histograms.append(
            {
                'class': catname,  # класс изображения
                'image': imfile,  # файл изображения
                'histogram': histogram,  # гистограмма слов
            }
        )
    return centers, word_histograms


if __name__ == '__main__':
    SCRIPT = Path(sys.argv[0])  # путь к скрипту
    SCRIPT_DIR = SCRIPT.parent.resolve()  # абсолютный путь к каталогу скрипта
    CALTECH_DIR = SCRIPT_DIR / 'caltech101'  # путь к каталогу CALTECH
    classes = prepare_classes(CALTECH_DIR)
    print(list(classes.keys()))
    N_words = 128
    vocab, histograms = get_visual_words(classes, N_words)
    output = {
        'word_count': N_words,
        'classes': list(classes.keys()),
        'histograms': histograms,
        'centers': vocab.tolist()
    }
    with (SCRIPT_DIR / 'words.json').open('wt', encoding='utf-8') as dst:
        json.dump(  # сохраняем объект в JSON пишем его в файл
            output,  # что записываем
            dst,  # куда записываем
            indent=2,  # переносы строк и отступы 2 пробела
            ensure_ascii=False,  # не-ASCII символы оставить как есть
        )

