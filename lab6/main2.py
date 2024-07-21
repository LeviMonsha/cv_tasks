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
    img = load_image("coins.jpg")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur( # размытие
        gray_image,
        (3, 3) # размер окрестности для усреднения
    )

    # параметры поиска окружностей
    # сколько пикселей должно проголосовать за окружность
    threshold = 80
    # минимальный радиус
    minRadius = 40
    # максимальный радиус
    maxRadius = 0 # без ограничений
    # допустимое расстояние между центрами
    minDist = 100

    while True:
        print(f"T={threshold}, R={minRadius}, D={minDist}")
        circles = cv2.HoughCircles( # поиск окружностей
            blurred, # изображениеddddddddd
            method=cv2.HOUGH_GRADIENT, # используемый градиентный метод
            dp=1, # шаг при поиске центра
            param1=50, # для градиентного метода - порог Кэнни
            param2=threshold, # для градиентного метода - число голосов
            minDist=minDist,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        copy = img.copy()
        if circles is not None:
            print(circles.shape, circles.dtype)
            for x, y, r in circles[0, :].astype(np.int32):
                cv2.circle(
                    copy,
                    (x, y), r,
                    (64, 255, 64), 2
                )
        cv2.imshow("res", copy)
        key = cv2.waitKey()

        if key == 27:
            break
        elif key == ord('q'):
            threshold += 5
        elif key == ord('a'):
            threshold -= 5
        elif key == ord('w'):
            minRadius += 5
        elif key == ord('s'):
            minRadius -= 5
        elif key == ord('e'):
            minDist += 5
        elif key == ord('d'):
            minDist -= 5
        threshold = max(5, min(500, threshold))
        minRadius = max(5, min(500, minRadius))
        minDist = max(5, min(500, minDist))