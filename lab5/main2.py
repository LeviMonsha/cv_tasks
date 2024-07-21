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

    sample = load_image("book1.jpg")
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    # ищем особенности
    sample_pts, sample_descriptors = sift.detectAndCompute(gray_sample,None )

    frame = load_image("bookz.jpg")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_pts, frame_descriptors = sift.detectAndCompute(gray_frame, None)

    lowe_threshold = 0.9

    # ищем пары похожих точек
    matches = matcher.knnMatch(
        frame_descriptors, # дескрипторы на сцене
        sample_descriptors, # дескрипторы на образе
        k=2, # ищем по 2 наилучших совпадения для каждой точки
    )

    while True:
        print(f"Lowe={lowe_threshold:.2f}")
        # ищемудачное сравнение по критерию Лёвэ
        good_matches = []
        for m1, m2 in matches: # m1, m2 - объекты сравнения
            if m1.distance < lowe_threshold * m2.distance:
                # совпадение уверенное - используем его
                good_matches.append(m1)

        # показываем связи между точками
        W = sample.shape[1] + frame.shape[1]
        H = max(sample.shape[0], frame.shape[0])
        result = np.zeros((H, W, 3), np.uint8)
        cv2.drawMatches(
            frame, # изображение-сценв
            frame_pts, # позиции особенностей на сцене
            sample, # изображение-образец
            sample_pts, # позиции особенностей на образе
            good_matches, # список сопоставлений точек
            result, # где рисовать результат
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # только точки с парой
        )
        cv2.imshow("matches", result)
        key = cv2.waitKeyEx()

        if key == ESC:
            break
        elif key == UP:
            lowe_threshold += 0.05
        elif key == DOWN:
            lowe_threshold -= 0.05

        lowe_threshold = max(0.05, min(1.0, lowe_threshold))