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
    img = load_image("sils.png")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sample = load_image("sil1.png")
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    ght =cv2.createGeneralizedHoughBallard()
    ght.setLevels(360)
    ght.setMinDist(50) # минимальное расстояние между фигурами
    ght.setDp(2) # шаг поиска позиций
    ght.setVotesThreshold(50) # порог голосов
    ght.setTemplate(gray_sample) # фигура, которую ищем

    copy = img.copy()
    # выполняем поиск
    positions, votes = ght.detect(gray_image)
    if positions is not None:
        R = max(gray_sample.shape) // 2
        color = (64, 255, 64)
        for row in positions[0, :]:
            x, y = row[0:2].astype(np.int32)
            cv2.circle(
                copy,
                (x, y), R, color, 2
            )
    cv2.imshow("res", copy)
    cv2.waitKey()