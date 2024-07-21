import numpy as np
import cv2

#~~~~~~~~~~~~~~~~~
def load_image(fpath: str) -> np.ndarray:
    data = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img
#~~~~~~~~~~~~~~~~~

ESC   = 0x00000018
UP    = 0x00260000
DOWN  = 0x00280000
LEFT  = 0x00250000
RIGHT = 0x00270000

if __name__ == '__main__':
    img = load_image('contrast.png')
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV_FULL
    )

    threshold, _ = cv2.threshold(hsv[..., 1],
                                 0, 155,
                                 cv2.THRESH_OTSU | cv2.THRESH_TOZERO)

    while True:
        _, set = cv2.threshold(hsv[..., 1],
                                 threshold, 255,
                                 cv2.THRESH_TOZERO)
        hsv_modified = np.copy(hsv)
        hsv_modified[..., 1] = set  # заменяем насыщенность
        result = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB_FULL)
        cv2.imshow('Result', result)
        key = cv2.waitKeyEx()
        if key == ESC:
            break
        elif key == UP:
            threshold += 5
        elif key == DOWN:
            threshold -= 5
        threshold = min(255, max(0, threshold))
        print(threshold)

    cv2.destroyAllWindows()
