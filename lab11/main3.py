import numpy as np
import cv2

if __name__ == '__main__':
    video = cv2.VideoCapture("sunset.mp4")
    back_sub = cv2.createBackgroundSubtractorMOG2(
        detectShadows=False
    )

    kernel_open = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    while True:
        if cv2.waitKey(25) == 27:
            break
        success, frame = video.read()
        if not success:
            break

        mask = back_sub.apply( # обрабатываем кадр
            frame, # кадр
            # скорость обновления модели фона
            # 0 - не обновлять вообще
            # 1 - сбрасываем модель фона для каждого кадра
            learningRate=0.010
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # ищем отдельные белые пятна на маске
        result = cv2.connectedComponentsWithStats(
            mask, # маска
            ltype=cv2.CV_16U # тип метки = сколько макс пятен
        )
        _, labels, stats, _centroids = result
        for stat in stats[1:].astype(np.int32):
            x, y, w, h, area = stat
            if area > 20:
                cv2.rectangle(
                    frame,
                    (x, y), (x+w, y+h),
                    (128, 255, 255), 2
                )

        cv2.imshow("video", frame)
        cv2.imshow("mask", mask)

    video.release()