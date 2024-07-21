import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

ESC   = 0x0000001B
UP    = 0x00260000
DOWN  = 0x00280000
LEFT  = 0x00250000
RIGHT = 0x00270000
SPACE = 0x00000020

def load_image(fpath: str) -> np.ndarray:
    data = np.fromfile(fpath, dtype = np.uint8) # изображение в массив
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) # цветное изображение
    if img is None:
        raise IOError('Неверный формат файла')
    return img

def display_time(arr):
    for j, i in enumerate(arr):
        arr[j] = sum(i) / len(i)
        print(f'{comp_sift[j]} - {labels[j]}')

def build_histogram(sift, kaze, orb):
    labels = ['Поиск', 'Сопоставление', 'Гомография']
    x = range(len(labels))
    fig, ax = plt.subplots()
    bar_width = 0.2
    bar1 = ax.bar(x, sift, bar_width, label='SIFT')
    bar2 = ax.bar([i + bar_width for i in x], orb, bar_width, label='ORB')
    bar3 = ax.bar([i + 2 * bar_width for i in x], kaze, bar_width, label='KAZE')
    ax.set_xticks([i + bar_width for i in x])
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylabel('Время - секунды')
    plt.title('Сравнение времени выполнения методов')
    plt.show()

def set_cmp(meth, matcher, arr_time, color):
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample_end = np.array(
        [(0, 0), (sample.shape[1] - 1, 0), (sample.shape[1] - 1, sample.shape[0] - 1), (0, sample.shape[0] - 1)],
        np.float32).reshape(-1, 1, 2)
    # ищем особенности
    time_f = time.time()
    sample_pts, sample_descriptors = meth.detectAndCompute(gray_sample, None)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_pts, frame_descriptors = meth.detectAndCompute(gray_frame, None)
    arr_time[0].append(time.time() - time_f)

    lowe_threshold = 0.7
    # ищем пары похожих точек
    time_c = time.time()
    matches = matcher.knnMatch(
        frame_descriptors,  # дескрипторы на сцене
        sample_descriptors,  # ескрипторы на лбразце
        k=2,  # ищем по  2 наилучших совпадения для каждой точки

    )
    arr_time[1].append(time.time() - time_c)
    # ищем удачные сравнения по критерию Лёвэ
    time_g = time.time()
    good_matches = []
    for m1, m2 in matches:  # объекты сравнения
        if m1.distance < lowe_threshold * m2.distance:
            # совпадение уверенное
            good_matches.append(m1)

    # ищем координаты Хороших точек
    points_sample = []
    points_frame = []
    for m in good_matches:
        points_sample.append(sample_pts[m.trainIdx].pt)  # точка на образце
        points_frame.append(frame_pts[m.queryIdx].pt)  # точка на кадре

    points_sample = np.array(points_sample, np.float32).reshape(-1, 1, 2)
    points_frame = np.array(points_frame, np.float32).reshape(-1, 1, 2)

    # ищем проективное преобразование
    matrix, ptmask = cv2.findHomography(
        points_sample,  # прообразы
        points_frame,  # образы куда
        cv2.RANSAC  # используем метод Ransac (выбросы)
    )
    # обводим книгу на кадре
    frame_ends = cv2.perspectiveTransform(sample_end, matrix)
    frame_ends = frame_ends.reshape(1, -1, 2).astype(np.int32)
    cv2.polylines(
        frame,  # где рисуем
        frame_ends,  # координаты углов
        True,  # нужно замкнуть многоугольник
        color,
        2
    )
    arr_time[2].append(time.time() - time_g)

    print(arr_time)

sift = cv2.SIFT.create()
matcher_sift = cv2.BFMatcher(cv2.NORM_L1)
kaze = cv2.KAZE.create()
matcher_kaze = cv2.BFMatcher(cv2.NORM_L1)
hamming = cv2.ORB.create()
matcher_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)

comp_sift = [[] for i in range(3)]
comp_kaze = [[] for j in range(3)]
comp_orb = [[] for k in range(3)]

sample = load_image('book1.jpg')
video = cv2.VideoCapture('bookz.mp4')
try:
    while True:
        global frame
        success, frame = video.read()
        if not success:
            print('Не удалось прочитать видео')
            break
        set_cmp(hamming, matcher_hamming, comp_orb, (64, 255, 64))
        set_cmp(sift, matcher_sift, comp_sift, (255, 255, 64))
        set_cmp(kaze, matcher_kaze, comp_kaze, (64, 64, 255))
        cv2.imshow('Result', frame)
        key = cv2.waitKey(40)
        if key == 27:
            print('Остановка видео')
            break
finally:
    cv2.destroyAllWindows()
    video.release() # освобождаем источник видео
    labels = ['Поиск', 'Сопоставление', 'Гомография']
    total = [comp_sift, comp_kaze, comp_orb]
    print('---sift---')
    display_time(comp_sift)
    print('---kaze---')
    display_time(comp_kaze)
    print('---orb---')
    display_time(comp_orb)
    build_histogram(comp_sift, comp_kaze, comp_orb)