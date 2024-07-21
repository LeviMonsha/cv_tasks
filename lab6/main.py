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
    img = load_image("road.jpg")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur( # размытие
        gray_image,
        (3, 3) # размер окрестности для усреднения
    )
    canny = cv2.Canny( # поиск границ
        blurred,
        50, 150
    )

    # параметры преобразования Хафа
    # минимальное число за прямую
    threshold = 100
    # минимальная длина отрезка прямой
    minLineLength = 100
    # аксимальная длина разрыва в прямой
    maxLineGap = 100

    while True:
        print(f"T={threshold}, L={minLineLength}, G={maxLineGap}")

        lines = cv2.HoughLinesP( # поиск отрезков
            canny, # изображение для поиска
            theta=np.radians(1), # шаг угла в радианах
            rho=1, # шаг смещения в пикселях
            threshold=threshold,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap
        )
        print(lines.shape, lines.dtype)
        copy = img.copy()
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(
                copy,
                (x1, y1),
                (x2, y2),
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
            minLineLength += 5
        elif key == ord('s'):
            minLineLength -= 5
        elif key == ord('e'):
            maxLineGap += 5
        elif key == ord('d'):
            maxLineGap -= 5
        threshold = max(5, min(500, threshold))
        minLineLength = max(5, min(500, minLineLength))
        maxLineGap = max(5, min(500, maxLineGap))