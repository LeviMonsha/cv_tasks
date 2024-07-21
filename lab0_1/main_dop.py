import numpy as np
import cv2


# ~~~~~~~~~~~~~~~~~
def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img


# ~~~~~~~~~~~~~~~~~

def save_image(fpath: str, img: np.ndarray) -> None:
    ret, data = cv2.imencode(".png", img)
    if not ret:
        raise IOError("Invalid image")
    with open(fpath, "wb") as dst:
        dst.write(data)


if __name__ == '__main__':
    image = load_image("lena.png")
    name_file = input("введите имя файла: ")
    if (name_file.isalnum()):
        x, y, w, h = cv2.selectROI("Select a rectangle", image)
        part = image[y:h + y, x:w + x]
        if w or h != 0:
            save_image(f"{name_file}.png", part)
        else: print("неверный масштаб изображения")
    else: print("ошибка в именовании файла")