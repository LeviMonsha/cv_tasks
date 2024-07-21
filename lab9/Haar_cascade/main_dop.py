from pathlib import Path
import numpy as np
import cv2

ESC   = 0x00000018
UP    = 0x00260000
DOWN  = 0x00280000

if __name__ == '__main__':
    LOGO = "training/cascade.xml"
    cascade = cv2.CascadeClassifier(str(LOGO))
    if cascade.empty():
        raise IOError("каскад загрузить не удалось")

    # параметры метода Виолы-Джонса
    # во сколько раз уменьшаем изображение на каждом шаге
    scale_per_stage = 1.1
    # колько соседних окон тоже дало положительный отклик
    min_neighbours = 1

    video = cv2.VideoCapture("video_cur.mp4")

    while True:
        success, img = video.read()
        if not success:
            print("видео закончилось")
            break
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"min_neighbours: {min_neighbours}")
        copy = img.copy()

        rois = cascade.detectMultiScale( # вызываем каскад
            gray_img, # обрабатываемое изображение
            scaleFactor=scale_per_stage,
            minNeighbors=min_neighbours
        )

        for x, y, w, h in rois:
            cv2.rectangle(
                copy,
                (x, y), (x+w, y+h),
                (64, 255, 255),
                2
            )
        cv2.imshow("logo", copy)
        key = cv2.waitKeyEx(10)

        if key == ESC:
            break
        elif key == UP:
            min_neighbours += 1
        elif key == DOWN:
            min_neighbours -= 1
        elif key == 32:
            cv2.waitKey()
        min_neighbours = max(0, min(10, min_neighbours))