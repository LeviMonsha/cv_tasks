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
    sample = load_image("book1.jpg")
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    quality = 0.2 # углы с откликом 20% наилучшего - не считаются
    min_distance = 20 # минимальное расстояние между углами
    max_corners = 100 # сколько углов хотим получить

while True:
        # ищем углы на изображении
        sample_features = cv2.goodFeaturesToTrack(
            image=gray_sample, # анализируем изображение
            maxCorners=max_corners,
            qualityLevel=quality, # насколько "сильные" углы нужны
            minDistance=min_distance,
            useHarrisDetector=False # детектор Ши-Томаси
        )
        # print(sample_features.shape, sample_features.dtype)
        copy = sample.copy()
        sf = sample_features[:, 0, :].astype(np.int32)
        for x, y in sf:
            cv2.circle(copy, (x, y), 4, (128, 255, 255), 1)
            cv2.circle(copy, (x, y), min_distance, (255, 255, 128), 1)
        cv2.imshow("features", copy)
        key = cv2.waitKeyEx()
        if key == ESC:
            break
        elif key == UP:
            quality += 0.05
        elif key == DOWN:
            quality -= 0.05
        elif key == RIGHT:
            min_distance += 5
        elif key == LEFT:
            min_distance -= 5
        quality = max(0.05, min(0.95, quality))
        min_distance = max(5, min(500, min_distance))
        print(f"Q = {quality:.2f} R = {min_distance}")