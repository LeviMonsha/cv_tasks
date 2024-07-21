import sys
import typing as tp

import numpy as np
import cv2

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

if __name__ == '__main__':
    sift = cv2.SIFT.create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    sample = load_image("book3.jpg")
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample_end = np.array([
        (0, 0),
        (sample.shape[1] - 1, 0),
        (sample.shape[1] - 1, sample.shape[0] - 1),
        (0, sample.shape[0] - 1)
    ], np.float32).reshape(-1, 1, 2)

    # ищем особенности
    sample_pts, sample_descriptors = sift.detectAndCompute(gray_sample,None )

    video = cv2.VideoCapture("bookz.mp4")
    try:
        while True:
            success, frame = video.read()  # читаем кадр
            if not success:
                print("video has ended")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_pts, frame_descriptors = sift.detectAndCompute(gray_frame, None)

            lowe_threshold = 0.7

            # ищем пары похожих точек
            matches = matcher.knnMatch(
                frame_descriptors, # дескрипторы на сцене
                sample_descriptors, # дескрипторы на образе
                k=2, # ищем по 2 наилучших совпадения для каждой точки
            )

            print(f"Lowe={lowe_threshold:.2f}")
            # ищем удачное сравнение по критерию Лёвэ
            good_matches = []
            for m1, m2 in matches: # m1, m2 - объекты сравнения
                if m1.distance < lowe_threshold * m2.distance:
                    # совпадение уверенное - используем его
                    good_matches.append(m1)
            # ищем координаты "хороших" точек
            points_sample = []
            points_frame = []

            for m in good_matches:
                points_sample .append(sample_pts[m.trainIdx].pt) # точка на образче
                points_frame .append(frame_pts[m.queryIdx].pt) # точка на кадре

            points_sample = np.array(points_sample, np.float32).reshape(-1, 1, 2)
            points_frame = np.array(points_frame, np.float32).reshape(-1, 1, 2)

            # ищем проективное преобразование
            matrix, ptmask = cv2.findHomography(
                points_sample, # прообразы откудв
                points_frame, # образы куда
                cv2.RANSAC # используем метод RANSAC
            )
            # обводим книгу на карте
            frame_ends = cv2.perspectiveTransform(sample_end, matrix)
            frame_ends = frame_ends.reshape(1, -1, 2).astype(np.int32)
            cv2.polylines(
                frame, # где рисуем
                frame_ends, # координаты углов
                True, # нужно замкнуть многоугольник
                (64, 255, 64),
                2
            )
            cv2.imshow("frame", frame)
            key = cv2.waitKey(10)  # вызов с таймаутом 10 мс
            if key == 27:
                print("stopped by user")
                break

    finally:
        video.release()  # освобождает источник видео