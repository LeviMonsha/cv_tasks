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

gray = (127, 127, 127)

if __name__ == '__main__':
    img = load_image("contrast.png")
    img1 = load_image("contrast.png")
    # cистема Hue-Saturation-Value
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV_FULL
    )

    _, sat = cv2.threshold(
        hsv[..., 1], # канал насыщенности
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    img_maskGray = img.copy()

    mask_color = sat > 0
    mask_gray = sat == 0
    img[mask_color] = gray
    img_maskGray[mask_gray] = gray
    cv2.imshow("result_gray_items", img)
    cv2.imshow("result_color_items", img_maskGray)
    cv2.waitKey()
