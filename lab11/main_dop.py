import numpy as np
import cv2

if __name__ == '__main__':
    video = cv2.VideoCapture("bookz.mp4")

    # араметры детектора углов Харриса
    features_params = {
        "maxCorners": 100,  # макс число точек
        "qualityLevel": 0.3,  # допустимая сила отклика относительно максимальной
        "minDistance": 20,  # мин расстояние между точками
    }
    # параметры расчета оптического потока
    lk_params = {
        "winSize": (15, 15),  # размер окна расчета
        "maxLevel": 2,  # число уровней пирамиды изображения
        "criteria": (  # критерии останова
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            10, 0.3
        )
    }

    _, old_frame = video.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # особенности на предыдущем кадре
    old_features = cv2.goodFeaturesToTrack(
        old_gray,  # изображение
        **features_params  # параметры из словаря
    )
    window = cv2.selectROI("select area", old_frame)
    x, y, w, h = window
    if w == 0 or h == 0:
        raise SystemExit(0)
    roi = old_frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # находим маску (какие пиксели используем)
    mask = cv2.inRange(
        hsv_roi,
        # нижние границы диапазона
        np.array([0.0, 60.0, 32.0]),
        # верхние границы диапазона
        np.array([180.0, 255.0, 255.0]),
    )
    # строим цветовую гистограмму
    histogram = cv2.calcHist(
        [hsv_roi], # изображение
        [0], # используем каналы оттенка
        mask, # какие пиксели игнорировать
        [180], # сколько "корзин" для каждого канала
        [0, 180], # диапазоны для каждого канала
    )

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        if cv2.waitKey(25) == 27:
            break
        success, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # рассчитываем оптический поток пирамидальным методом Люкаса-Канадэ
        moved_features, good, _err = cv2.calcOpticalFlowPyrLK(
            old_gray,  # предыдущий кадр
            gray,  # текущий кадр
            old_features,  # предыдущая позиция особенностей
            None,  # текущая позиция неизвестна
            **lk_params  # остальные параметры
        )
        old_pts = old_features[good == 1].astype(np.int32)
        new_pts = moved_features[good == 1].astype(np.int32)

        min_x = np.min(new_pts[:, 0])
        min_y = np.min(new_pts[:, 1])
        max_x = np.max(new_pts[:, 0])
        max_y = np.max(new_pts[:, 1])

        if min_x > x or min_y > y or max_x < w + x or max_y < h + y:
            print("Прямоугольник вышел за пределы кадра.")
            print(max_x, max_y)

        if not success:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # насколько пиксели кадра похожи на гистограмму
        dst = cv2.calcBackProject(
            [hsv], # изображение
            [0], # каналы оттенка
            histogram, # гистограмма - образек
            [0, 180], # диапазон значений 0..180
            1 # масштабировать кадр не надо
        )

        # сдвиг среднего
        _, window = cv2.meanShift(
            dst,  # маска похожести
            window,  # старая позиция окна
            term_crit,  # критерии останова
        )
        x, y, w, h = window

        cv2.rectangle(
            frame,
            (x, y), (x+w, y+h),
            (128, 255, 255), 2
        )

        cv2.imshow("res", frame)
        old_gray = gray
        old_features = cv2.goodFeaturesToTrack(gray, **features_params)

    video.release()