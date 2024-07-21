import numpy as np
import cv2

if __name__ == '__main__':
    video = cv2.VideoCapture("bookz.mp4")
    _, old_frame = video.read()
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
    # нормализуем гистограмму, чтобы убрать влияние размера
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        if cv2.waitKey(25) == 27:
            break
        success, frame = video.read()
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

        cv2.imshow("mean shift", frame)
        cv2.imshow("back projection", dst)

    video.release()