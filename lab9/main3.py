from pathlib import Path
import numpy as np
import cv2

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

if __name__ == '__main__':
    CASCAD_DIR = Path(cv2.data.haarcascades)
    FACE = CASCAD_DIR / 'haarcascade_frontalface_default.xml'
    EYES = CASCAD_DIR / 'haarcascade_eye.xml'
    cascade = cv2.CascadeClassifier(str(FACE))
    cascade_eyes = cv2.CascadeClassifier(str(EYES))
    if cascade.empty():
        raise IOError('Каскад загрузить не удалось')

    #параметры каскада
    # во сколько раз уменьшаетмся изображение
    scale_per_stage = 1.3
    # сколько соседних окон тоже дало положительный отклик
    min_neighbours = 1

    video = cv2.VideoCapture('face.mp4')
    while True:
        success, img = video.read()
        if not success:
            print("видео закончилось")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"min_neighbours: {min_neighbours}")
        copy = img.copy()
        rois = cascade.detectMultiScale(
            gray, # обрабатываемое изображение
            scaleFactor=scale_per_stage,
            minNeighbors=min_neighbours
        )
        for x, y, w, h in rois:
            cv2.rectangle(copy, (x, y), (x + w, y + h), (64, 255, 255), 2)
            roi_color_head = copy[y:y + h, x:x + w]
            rois_eyes = cascade_eyes.detectMultiScale(
                roi_color_head,
                scaleFactor=scale_per_stage,
                minNeighbors=min_neighbours
            )
            for ex, ey, ew, eh in rois_eyes:
                cv2.circle(roi_color_head, (ex+int(ew / 2), ey+int(eh / 2)), 10, (64, 64, 64), 2)
        cv2.imshow('face', copy)
        key = cv2.waitKeyEx(10)
        if key == ESC:
            break
        elif key == UP:
            min_neighbours += 1
        elif key == DOWN:
            min_neighbours -= 1
        min_neighbours = max(0, min(10, min_neighbours))