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

def repaint(img: np.ndarray, k: float) -> np.ndarray:
    # in = диапазон a...b
    # out = диапазон 0...255
    # out = (in - a) * 255 / (b - a) + 0
    minval = img.min()
    # преобразуем тип элементов массива
    float_img = img.astype(np.float32)
    # result_float = (float_img - minval) * 255.0 / (maxval - minval)
    # a = a + 2 создает копию объекта
    float_img -= minval
    float_img *= k
    float_img += minval
    # обрезали диапазон значений эл мас
    float_img.clip(min=0, max=255, out=float_img)  # без out создает новый массив / изменение "на месте"

    result = float_img.astype(np.uint8)

    print(f"оригинал: {img.min()}...{img.max()}")
    print(f"результат: {result.min()}...{result.max()}")

    w = img.shape[1]
    combo = np.zeros(
        shape=(img.shape[0], 2 * w, 3),
        dtype=np.uint8
    )
    combo[:, :w, :] = img
    combo[:, w:, :] = result
    cv2.putText(  # выводим текст на изображение
        combo,
        text="before",
        # левый край, базовая линия
        org=(0, combo.shape[0] - 10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1.0,
        color=(255, 255, 255)
    )
    cv2.putText(  # выводим текст на изображение
        combo,
        text="after",
        # левый край, базовая линия
        org=(w, combo.shape[0] - 10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1.0,
        color=(255, 255, 255)
    )
    return combo

if __name__ == '__main__':
    k_str = 1
    k = float(k_str)
    img = load_image("lena_2.png")

    while True:
        combo = repaint(img, k)
        cv2.imshow("combo", combo)
        key = cv2.waitKey()

        if key == 27:
            break
        elif key == ord('w'):
            k += 0.1
            if k > 10: k = 0
        elif key == ord('s'):
            k -= 0.1
            if k < 0: k = 10
        print(f"Коэффициент k = {np.round(k, 1)}")