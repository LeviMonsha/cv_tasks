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
    copy = img.copy()
    channels = ["blue", "green", "red"]
    for i, chname in enumerate(channels):
        part = img[..., i] #img[:, :, i]
        minidx = part.argmin()
        maxidx = part.argmax()
        # пересчитывем линейные индексы в обычные
        minpos = np.unravel_index(minidx, part.shape) # линейный индекс минимума
        maxpos = np.unravel_index(maxidx, part.shape) # линейный индекс максимума
        print(chname)
        print(f"min at {minpos} = {part[minpos]}")
        print(f"max at {maxpos} = {part[maxpos]}")
        color = [0, 0, 0]
        color[i] = 255
        cv2.line(
            copy,
            pt1=minpos[::-1], # начало прямой x,y
            pt2=maxpos[::-1], # конец прямой x,y
            color=color, # цвет
            thickness=2 # толщина пикселей
        ) # рисуем отрезок прямой
        cv2.circle(
            copy,
            center=maxpos[::-1], # центр (x,y)
            radius=3,
            color=color,
            thickness=-1 # толщина -1 = закрашенный круг
        )
    cv2.imshow("res", copy)
    cv2.waitKey()