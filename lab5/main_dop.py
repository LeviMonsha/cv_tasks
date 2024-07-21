from time import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

#~~~~~~~~~~~~~~~~~
def load_image(fpath: str) -> np.ndarray:
    data = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img
#~~~~~~~~~~~~~~~~~

ESC   = 0x00000018
UP    = 0x00260000
DOWN  = 0x00280000
LEFT  = 0x00250000
RIGHT = 0x00270000
SPACE = 0x00000020

def optional_operation_create():
    sample = load_image("book3.jpg")
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample_end = np.array([
        (0, 0),
        (sample.shape[1] - 1, 0),
        (sample.shape[1] - 1, sample.shape[0] - 1),
        (0, sample.shape[0] - 1)
    ], np.float32).reshape(-1, 1, 2)


    def optional_operation(feature_extraction_alg, matcher):
        video = cv2.VideoCapture("bookz.mp4")
        gl_find_points_time = 0
        gl_comparison_points_time = 0
        gl_elimination_homography_time = 0

        time_first = time()
        # ищем особенности
        sample_pts, sample_descriptors = feature_extraction_alg.detectAndCompute(gray_sample, None)
        gl_find_points_time += time() - time_first

        count = 1
        try:
            while True:
                success, frame = video.read()  # читаем кадр
                if not success:
                    print("video has ended")
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frame_pts, frame_descriptors = feature_extraction_alg.detectAndCompute(gray_frame, None)

                lowe_threshold = 1

                time_first = time()
                # ищем пары похожих точек
                matches = matcher.knnMatch(
                    frame_descriptors,  # дескрипторы на сцене
                    sample_descriptors,  # дескрипторы на образе
                    k=2,  # ищем по 2 наилучших совпадения для каждой точки
                )
                gl_comparison_points_time += time() - time_first

                # ищем удачное сравнение по критерию Лёвэ
                good_matches = []

                time_first = time()
                for m1, m2 in matches:  # m1, m2 - объекты сравнения
                    if m1.distance < lowe_threshold * m2.distance:
                        # совпадение уверенное - используем его
                        good_matches.append(m1)
                # ищем координаты "хороших" точек
                points_sample = []
                points_frame = []

                for m in good_matches:
                    points_sample.append(sample_pts[m.trainIdx].pt)  # точка на образце
                    points_frame.append(frame_pts[m.queryIdx].pt)  # точка на кадре

                points_sample = np.array(points_sample, np.float32).reshape(-1, 1, 2)
                points_frame = np.array(points_frame, np.float32).reshape(-1, 1, 2)

                # ищем проективное преобразование
                matrix, ptmask = cv2.findHomography(
                    points_sample,  # прообразы откудв
                    points_frame,  # образы куда
                    cv2.RANSAC  # используем метод RANSAC
                )
                gl_elimination_homography_time += time() - time_first

                # обводим книгу на карте
                frame_ends = cv2.perspectiveTransform(sample_end, matrix)
                frame_ends = frame_ends.reshape(1, -1, 2).astype(np.int32)
                cv2.polylines(
                    frame,  # где рисуем
                    frame_ends,  # координаты углов
                    True,  # нужно замкнуть многоугольник
                    (64, 255, 64),
                    2
                )
                count += 1

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)  # вызов с таймаутом 40 мс
                if key == 27:
                    print("stopped by user")
                    break

        finally:
            video.release()  # освобождает источник видео

        avr_find_points = gl_find_points_time / count
        avr_comparison_points = gl_comparison_points_time / count
        avr_elimination_homography = gl_elimination_homography_time / count
        res_avr = [avr_find_points, avr_comparison_points, avr_elimination_homography]
        return res_avr

    return optional_operation

def build_histogram(sift, orb, kaze):
    labels = ['Поиск', 'Сопоставление', 'Гомография']
    x = range(len(labels))
    fig, ax = plt.subplots()
    bar_width = 0.2
    ax.bar(x, sift, bar_width, label='SIFT')
    ax.bar([i + bar_width for i in x], orb, bar_width, label='ORB')
    ax.bar([i + 2 * bar_width for i in x], kaze, bar_width, label='KAZE')
    ax.set_xticks([i + bar_width for i in x])
    ax.set_xticklabels(labels)
    ax.legend()
    plt.ylabel('Время - секунды')
    plt.title('Сравнение времени выполнения методов')
    plt.show()

if __name__ == '__main__':
    opt_oper = optional_operation_create()

    sift = cv2.SIFT.create()
    matcher_sift = cv2.BFMatcher(cv2.NORM_L1)
    res_sift = opt_oper(sift, matcher_sift)

    orb = cv2.ORB.create()
    matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
    res_orb = opt_oper(orb, matcher_orb)

    kaze = cv2.KAZE.create()
    matcher_kaze = cv2.BFMatcher(cv2.NORM_L1)
    res_kaze = opt_oper(kaze, matcher_kaze)

    print(f"Алгоритм Поиск, мс\tСопоставление, мс\tГомография, мс\n"
          f"SIFT\t{res_sift[0]}\t{res_sift[1]}\t{res_sift[2]}\n"
          f"ORB\t{res_orb[0]}\t{res_orb[1]}\t{res_orb[2]}\n"
          f"KAZE\t{res_kaze[0]}\t{res_kaze[1]}\t{res_kaze[2]}\n")

    build_histogram(res_sift, res_orb, res_kaze)