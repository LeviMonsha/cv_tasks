import numpy as np
import cv2

#~~~~~~~~~~~~~~~~~
def load_image(fpath: str) -> numpy.ndarray:
    data = numpy.fromfile(fpath, dtype=numpy.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("неверный формат файла")
    return img
#~~~~~~~~~~~~~~~~~

def save_image(fpath: str, img: np.ndarray) -> None:
    ret, data = cv2.imencode(".png", img)
    if not ret:
        raise IOError("Invalid image")
    with open(fpath, "wb") as dst:
        dst.write(data)

if __name__ == '__main__':
    image = load_image("lena.png")
    h, w, ch = image.shape
    print(f"{w}x{h}, {ch} каналов")

    channels = ["blue", "green", "red"]
    # for i in range(len(channels)): chname = channels[i]
    for i, chname in enumerate(channels):
        part = image[:, :, i]
        cv2.imshow(chname, part)
        save_image(f"lena{chname}.png", part)
    cv2.waitKey()