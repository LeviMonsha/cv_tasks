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
    img = load_image("lena_2.png")
    channels = ["blue", "green", "red"]
    for i, chname in enumerate(channels):
        part = img[..., i] #img[:, :, i]
        minidx = part.argmin()
        maxidx = part.argmax()
        # пересчитывем линейные индексы в обычные
        minpos = np.unravel_index(minidx, part.shape)  # линейный индекс минимума
        maxpos = np.unravel_index(maxidx, part.shape)  # линейный индекс максимума
        print(chname)
        print(f"min at {minpos} = {part[minpos]}")
        print(f"max at {maxpos} = {part[maxpos]}")