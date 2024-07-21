import cv2
import numpy as np

# Чтение видео-файла
video = cv2.VideoCapture('chessboard.mp4')

# Ждем пока будут найдены углы шахматной доски
found_corners = False
frame = None
PATTERN_SIZE = (7, 3)  # размер шаблона (7х3)

while not found_corners:
    ret, frame = video.read()
    if not ret:
        break

    # Поиск углов шахматной доски на кадре видео
    ret, corners = cv2.findChessboardCorners(
            frame,
            PATTERN_SIZE,
            flags=cv2.CALIB_CB_FILTER_QUADS
        )

    if ret:
        found_corners = True
        pts1 = corners.reshape(-1, 1, 2)  # Первые найденные углы становятся точками для преобразования перспективы

# Размер окна
h, w, _ = frame.shape
window_size = (w, h)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Применяем преобразование перспективы, если углы шахматной доски найдены
    ret, corners = cv2.findChessboardCorners(
            frame,
            PATTERN_SIZE,
            flags=cv2.CALIB_CB_FILTER_QUADS
        )
    if ret:
        pts2 = corners.reshape(-1, 1, 2)
        if pts2.shape[0] == 4 and pts1.shape[0] == 4:  # Убедимся, что массивы содержат 4 точки
            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            warped_frame = cv2.warpPerspective(frame, matrix, window_size)

            # Отображаем преобразованный кадр
            cv2.imshow('Warped Frame', warped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
