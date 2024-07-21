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

if __name__ == '__main__':
    image = load_image("lena.png")
    h, w, ch = image.shape
    print(f"{w}x{h}, {ch} каналов")

    cv2.imshow("Image", image)
    while cv2.waitKey() not in (27, -1):
        pass