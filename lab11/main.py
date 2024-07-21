import numpy as np
import cv2

if __name__ == '__main__':
    video = cv2.VideoCapture("bookz.mp4")
    # параметры детектора углов Харриса
    features_params = {
        "maxCorners": 100, # макс число точек
        "qualityLevel": 0.3, # допустимая сила отклика относительно максимальной
        "minDistance": 20, # мин расстояние между точками
    }
    # параметры расчета оптического потока
    lk_params = {
        "winSize": (15, 15), # размер окна расчета
        "maxLevel": 2, # число уровней пирамиды изображения
        "criteria": ( # критерии останова
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            10, 0.3
        )
    }
    # предыдущий кадр видео
    _, old_frame = video.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # особенности на предыдущем кадре
    old_features = cv2.goodFeaturesToTrack(
        old_gray, # изображение
        **features_params # параметры из словаря
    )

    while True:
        if cv2.waitKey(50) == 27:
            break
        success, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # рассчитываем оптический поток пирамидальным методом Люкаса-Канадэ
        moved_features, good, _err = cv2.calcOpticalFlowPyrLK(
            old_gray, # предыдущий кадр
            gray, # текущий кадр
            old_features, # предыдущая позиция особенностей
            None, # текущая позиция неизвестна
            **lk_params # остальные параметры
        )
        old_pts = old_features[good == 1].astype(np.int32)
        new_pts = moved_features[good == 1].astype(np.int32)
        for p1, p2 in zip(old_pts, new_pts):
            delta = (p1 - p2) * 5
            cv2.line(
                frame,
                p2, p2 + delta,
                (128, 255, 255), 1
            )
            cv2.circle(frame, p2, 2,
                       (128, 255, 255), -1)
        cv2.imshow("opt flow", frame)
        old_gray = gray
        old_features = cv2.goodFeaturesToTrack(gray, **features_params)
    video.release()