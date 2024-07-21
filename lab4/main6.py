import sys
import typing as tp
import numpy as np
import cv2

PATTERN_SIZE = (7,3) # размер шаблона (2х2)

def load_image(fpath: str) -> np.ndarray:
    data = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img

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
    web_camera = cv2.VideoCapture("chessboard.mp4")
    iw = int(web_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(web_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    insert_ends = np.array([(0,0), (iw-1, 0), (iw-1, ih-1), (0, ih-1)], np.float32).reshape(-1, 1, 2)
    #cv2.drawChessboardCorners(insert, PATTERN_SIZE, insert_pts, True)
    #cv2.imshow('Insert', insert)

    video = cv2.VideoCapture(0)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warped = np.zeros((h,w,3), np.uint8) # искажённый 'вставыш'
    mask =  np.zeros((h,w), np.uint8)
    try:
        while True:
            success_chess, frame = video.read()
            success_camera, insert = web_camera.read()
            insert_pts = get_pattern(insert.shape)
            if not success_camera or not success_chess:
                print('Не удалось прочитать видео')
                break

            success, corners = cv2.findChessboardCorners(
                frame,
                PATTERN_SIZE,
                flags = cv2.CALIB_CB_FILTER_QUADS
            )
            #print(corners.shape, corners.dtype)
            #print(corners)
            if success:
                if corners[0, 0, 0] > corners[-1,0,0]:
                    corners = corners[::-1, ...]

                cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, success)

            # ищем перспективное рпеобрразование
            matrix, _ptsmask = cv2.findHomography(
                insert_pts, # откуда
                corners, # точки образы
                cv2.RANSAC #используемы алгоритм
            )
            warped.fill(0)
            cv2.warpPerspective(
                insert, # преобразуем изображение
                matrix, # матрица преобразования
                (w, h), # целевое изображение
                dst = warped # записать результат в warped
            )
            frame_ends = cv2.perspectiveTransform( # преобразуем коордианаты точек
                insert_ends, # преобразуемы точки
                matrix  # описание преобразования
            )
            print(frame_ends)
            mask.dtype = np.uint8
            mask.fill(0)
            cv2.fillPoly(
                mask,
                frame_ends.reshape(1, -1, 2).astype(np.int32),
                (1, )
            ) # рисуем на маске
            # cv2.imshow('Mask', mask*255)
            mask.dtype = np.bool_
            # переносим пиксели
            frame[mask] = warped[mask]
            cv2.imshow('Video', frame)
            # cv2.imshow('Waped', warped)
            key = cv2.waitKey(20)
            if key == 27:
                print('Остановка видео')
                break
    finally:
        video.release() # освобождаем источник видео