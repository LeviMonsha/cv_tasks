import numpy as np
import cv2

PATTERN_SIZE = (7, 3) # размер шаблона (число пересечений 2*2)

#~~~~~~~~~~~~~~~~~
def load_image(fpath: str) -> np.ndarray:
    data = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img
#~~~~~~~~~~~~~~~~~

def get_pattern(shape: tuple) -> np.ndarray:
    pts = []
    for r in range(1, PATTERN_SIZE[1] + 1):
        for c in range(1, PATTERN_SIZE[0] + 1):
            x = c * shape[1] // (PATTERN_SIZE[0] + 1)
            y = r * shape[0] // (PATTERN_SIZE[1] + 1)
            pts.append((x, y))
    pts = np.array(pts, np.float32)
    pts = pts.reshape(-1, 1, 2)
    return pts

if __name__ == '__main__':
    insert = load_image("times-square.jpg")

    insert_pts = get_pattern(insert.shape)
    ih, iw = insert.shape[:2]
    insert_ends = np.array([
        (0, 0),
        (iw - 1, 0),
        (iw - 1, ih - 1),
        (0, ih - 1)
    ], np.float32).reshape(-1, 1, 2)
    # cv2.drawChessboardCorners(insert, PATTERN_SIZE, insert_pts, True)
    # cv2.imshow("insert", insert)

    video = cv2.VideoCapture("chessboard.mp4")

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warped = np.zeros((h, w, 3), np.uint8) # искаженная вставка
    mask = np.zeros((h, w), np.uint8)

    try:
        while True:
            success, frame = video.read() # читаем кадр
            if not success:
                print("video has ended")
                break

            success, corners = cv2.findChessboardCorners( # ищем координаты шахматной доски
                frame, # где ищем
                PATTERN_SIZE, # размер шаблона (число углов 2*2)
                flags=cv2.CALIB_CB_FILTER_QUADS # флаги алгоритма
            )
            # print(corners.shape, corners.dtype)
            # print(corners)

            # corners = corners.astype(np.int32).reshape(-1, 2) # 21, 2
            # for x, y in corners:
            #     cv2.circle(frame, (x, y), 5, (128, 255, 255), 2)

            if success: # нашли шаблон
                if corners[0, 0, 0] > corners[-1, 0, 0]:
                    corners = corners[::-1, ...]
                cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, success)

            # ищем перспективное преобразование
            matrix, _ptmask = cv2.findHomography(
                insert_pts, # точки-прообразы (откуда)
                corners, # точки-образы (куда)
                cv2.RANSAC # используем алгоритм
            )

            # преобразуем изображение вставку
            warped.fill(0)
            cv2.warpPerspective(
                insert, # преобразуемое изображение
                matrix, # матрица преобразования
                (w, h), # размеры целевого изображения
                dst=warped # записывем результат в массив warped
            )

            # создаем маску переноса пикселей
            frame_ends = cv2.perspectiveTransform( # преобразуем координаты точек
                insert_ends, # преобразуемые точки
                matrix, # описание преобразования
            )
            # for item in frame_ends:
            #     x, y = item[0].astype(np.int32)
            #     cv2.circle(frame, (x, y), 5, (128, 255, 255), 2)

            mask.dtype = np.uint8 # рисование работает только с uint8
            mask.fill(0)
            cv2.fillPoly(
                mask, # рисуем на маске
                frame_ends.reshape(1, -1, 2).astype(np.int32),
                (1, ) # цвет
            )
            mask.dtype = np.bool_ # делаем массив снова логическим
            # переносим пиксели
            frame[mask] = warped[mask]

            # cv2.imshow("warped", warped)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(20) # вызов с таймаутом 20 мс
            if key == 27:
                print("stopped by user")
                break
    finally:
        video.release() # освобождает источник видео