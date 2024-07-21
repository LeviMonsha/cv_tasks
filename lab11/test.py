import cv2
import numpy as np

# Загрузка видео
video = cv2.VideoCapture('bookz.mp4')

# параметры детектора углов Харриса
features_params = {
    "maxCorners": 100, # макс число точек
    "qualityLevel": 0.3, # допустимая сила отклика относительно максимальной
    "minDistance": 20, # мин расстояние между точками
}


# Читаем первый кадр видео
ret, frame = video.read()

# Выбираем область, которую хотим отслеживать
bbox = cv2.selectROI('Frame', frame)
x, y, w, h = bbox

# параметры расчета оптического потока
lk_params = {
    "winSize": (w, h), # размер окна расчета
    "maxLevel": 2, # число уровней пирамиды изображения
    "criteria": ( # критерии останова
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        10, 0.3
    )
}

# Определение начальных точек для отслеживания в выбранной области
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# особенности на предыдущем кадре
old_features = cv2.goodFeaturesToTrack(
    old_gray, # изображение
    **features_params # параметры из словаря
)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Оптический поток на основе Lucas-Kanade
    moved_features, good, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                         frame_gray,
                                                         old_features,
                                                         None,
                                                         **lk_params)

    # Выбираем только точки, которые были успешно отслежены
    good_new = moved_features[good == 1]

    # Обновляем координаты прямоугольной области
    x, y = good_new[0]
    bbox = (x - w // 2, y - h // 2, w, h)

    # Рисуем прямоугольную область и оптический поток
    cv2.rectangle(frame,
                  (int(x), int(y)),
                  (int(x + w), int(y + h)),
                  (255, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(20) == 27:
        break

    old_gray = frame_gray
    # Обновляем точки
    old_features = cv2.goodFeaturesToTrack(frame_gray, **features_params)

video.release()
cv2.destroyAllWindows()


# import numpy as np
# import cv2
#
# if __name__ == '__main__':
#     video = cv2.VideoCapture("bookz.mp4")
#
#     # араметры детектора углов Харриса
#     features_params = {
#         "maxCorners": 100,  # макс число точек
#         "qualityLevel": 0.3,  # допустимая сила отклика относительно максимальной
#         "minDistance": 20,  # мин расстояние между точками
#     }
#     # параметры расчета оптического потока
#     lk_params = {
#         "winSize": (15, 15),  # размер окна расчета
#         "maxLevel": 2,  # число уровней пирамиды изображения
#         "criteria": (  # критерии останова
#             cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
#             10, 0.3
#         )
#     }
#
#     _, old_frame = video.read()
#     old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#     # особенности на предыдущем кадре
#     old_features = cv2.goodFeaturesToTrack(
#         old_gray,  # изображение
#         **features_params  # параметры из словаря
#     )
#     window = cv2.selectROI("select area", old_frame)
#     x, y, w, h = window
#     if w == 0 or h == 0:
#         raise SystemExit(0)
#     roi = old_frame[y:y+h, x:x+w]
#     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     # находим маску (какие пиксели используем)
#     mask = cv2.inRange(
#         hsv_roi,
#         # нижние границы диапазона
#         np.array([0.0, 60.0, 32.0]),
#         # верхние границы диапазона
#         np.array([180.0, 255.0, 255.0]),
#     )
#     # строим цветовую гистограмму
#     histogram = cv2.calcHist(
#         [hsv_roi], # изображение
#         [0], # используем каналы оттенка
#         mask, # какие пиксели игнорировать
#         [180], # сколько "корзин" для каждого канала
#         [0, 180], # диапазоны для каждого канала
#     )
#
#     term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#
#     while True:
#         if cv2.waitKey(25) == 27:
#             break
#         success, frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # рассчитываем оптический поток пирамидальным методом Люкаса-Канадэ
#         moved_features, good, _err = cv2.calcOpticalFlowPyrLK(
#             old_gray,  # предыдущий кадр
#             gray,  # текущий кадр
#             old_features,  # предыдущая позиция особенностей
#             None,  # текущая позиция неизвестна
#             **lk_params  # остальные параметры
#         )
#         old_pts = old_features[good == 1].astype(np.int32)
#         new_pts = moved_features[good == 1].astype(np.int32)
#
#         if not success:
#             break
#
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         # насколько пиксели кадра похожи на гистограмму
#         dst = cv2.calcBackProject(
#             [hsv], # изображение
#             [0], # каналы оттенка
#             histogram, # гистограмма - образек
#             [0, 180], # диапазон значений 0..180
#             1 # масштабировать кадр не надо
#         )
#
#         # сдвиг среднего
#         _, window = cv2.meanShift(
#             dst,  # маска похожести
#             window,  # старая позиция окна
#             term_crit,  # критерии останова
#         )
#         x, y, w, h = window
#
#         for p1, p2 in zip(old_pts, new_pts):
#             delta = (p1 - p2) * 5
#             cv2.line(
#                 frame,
#                 p2, p2 + delta,
#                 (128, 255, 255), 1
#             )
#             cv2.circle(frame, p2, 2,
#                        (128, 255, 255), -1)
#         cv2.rectangle(
#             frame,
#             (x, y), (x+w, y+h),
#             (128, 255, 255), 2
#         )
#
#         cv2.imshow("res", frame)
#         old_gray = gray
#         old_features = cv2.goodFeaturesToTrack(gray, **features_params)
#
#     video.release()