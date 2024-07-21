import numpy as np
import cv2

PATTERN_SIZE = (7, 3) # размер шаблона (число пересечений 2*2)

if __name__ == '__main__':
    video = cv2.VideoCapture("chessboard.mp4")
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

            cv2.imshow("frame", frame)
            key = cv2.waitKey(40) # вызов с таймаутом 40 мс
            if key == 27:
                print("stopped by user")
                break
    finally:
        video.release() # освобождает источник видео