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
    # in = диапазон a...b
    # out = диапазон 0...255
    # out = (in - a) * 255 / (b - a) + 0
    minval = img.min()
    maxval = img.max()
    # преобразуем тип элементов массива
    float_img = img.astype(np.float32)
    # result_float = (float_img - minval) * 255.0 / (maxval - minval)
    # a = a + 2 создает копию объекта
    float_img -= minval
    float_img *= 255.0 / (maxval - minval)
    result = float_img.astype(np.uint8)

    print(f"оригинал: {img.min()}...{img.max()}")
    print(f"результат: {result.min()}...{result.max()}")

    w = img.shape[1]
    combo = np.zeros(
        shape=(img.shape[0], 2*w, 3),
        dtype=np.uint8
    )
    combo[:, :w, :] = img
    combo[:, w:, :] = result
    cv2.putText( # выводим текст на изображение
        combo,
        text="before",
        # левый край, базовая линия
        org=(0, combo.shape[0]-10),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1.0,
        color=(255,255,255)
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
    cv2.imshow("combo", combo)
    cv2.waitKey()