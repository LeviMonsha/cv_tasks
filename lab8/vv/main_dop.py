import numpy
import cv2
import sys
import typing as t
from pathlib import Path
from collections import Counter
import json
from main2 import prepare_image, prepare_classes

with open('words.json', 'rt', encoding='utf-8') as src:
    data = json.load(src)

N_words = data['word_count']
classes = data['classes']
vocab = numpy.array(data['centers'], numpy.float32)
histograms = data['histograms']
# обучаем классификатор
image_descriptors = numpy.array([  # описания изображений
    item['histogram'] for item in histograms
], numpy.float32)
image_labels = numpy.array([  # номера классов изображений
    classes.index(item['class']) for item in histograms
], numpy.int32)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria(
    (
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6
    )
)
print('Training...')
svm.train(  # обучаем классификатор
    image_descriptors,  # описания объектов
    cv2.ml.ROW_SAMPLE,  # объекты по строкам
    image_labels  # номера классов объектов
)
if not svm.isTrained():
    print('Training failed!')
    sys.exit(1)
print('    done.')


# функция принимает путь к изображению и возвращает имя класса
def classify_one(imfile: Path) -> str:
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # грузим и нормализуем изображение
    image = prepare_image(imfile)
    # ищем локальные особенности
    _pts, descriptors = detector.detectAndCompute(image, None)
    # для каждой особенности ищем ближайший центр кластера
    matches = matcher.match(descriptors, vocab)
    # какие номера визуальных слов есть в изображении
    words = [m.trainIdx for m in matches]
    cnt = Counter(words)
    # гистограмма частот визуальных слов
    histogram = [cnt[x] for x in range(N_words)]
    # классифицируем
    query = numpy.array([histogram], numpy.float32)
    _strength, response = svm.predict(query)
    return classes[int(response[0, 0])]  # возвращаем имя класса


def full_classify() -> t.List[t.Tuple[str, float]]:
    rates = []
    caltech_dir = Path(sys.argv[0]).parent / 'caltech101'
    image_classes = prepare_classes(caltech_dir)

    total_correct = 0
    total_images = 0

    for class_name, images in image_classes.items():
        good, bad, total = 0, 0, len(images)
        print(f'Class: {class_name}')

        for idx, imfile in enumerate(images, start=1):
            print(f'Processed images: {idx}/{total}', end='\r', flush=True)
            if classify_one(imfile) == class_name:
                good += 1
            else:
                bad += 1

        total_correct += good
        total_images += total

        print(f'{class_name:20}: {good}/{bad}/{total}')

        rates.append((
            class_name,  # имя класса
            good / total  # доля успешных классификаций
        ))

    avg_success_rate = total_correct / total_images
    print(f'\nУспешность классификации: {avg_success_rate:.2f}')

    return rates


if __name__ == '__main__':
    full_classify()